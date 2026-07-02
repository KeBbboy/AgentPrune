import argparse
import asyncio
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from AgentDropout.graph.graph import Graph
from AgentDropout.utils.const import AgentPrune_ROOT
from AgentDropout.utils.globals import CompletionTokens, Cost, PromptTokens, Time, CachedTokens, ReasoningTokens, BilledPromptTokens, BilledCompletionTokens, print_token_stats
from dataset.cl_bench_dataset import CLBenchDataset
from experiments.run_mmlu import get_kwargs

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="AgentPrune Experiments on CL-bench.")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default="dataset/cl_bench")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--limit_questions", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true",
                        help="Load data and build the graph without calling the model API.")
    parser.add_argument("--mode", type=str, default="DirectAnswer",
                        choices=[
                            "DirectAnswer", "FullConnected", "Random", "Chain", "Debate", "Layered", "Star", "Mesh",
                            "FakeFullConnected", "FakeRandom", "FakeChain", "FakeStar", "FakeMesh",
                            "FakeAGRandom", "FakeAGFull",
                        ])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--agent_names", nargs="+", type=str, default=["AnalyzeAgent"])
    parser.add_argument("--agent_nums", nargs="+", type=int, default=[1])
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--llm_name", type=str, default="MiniMax-M3")
    parser.add_argument("--api_base_url", type=str, default=None,
                        help="OpenAI-compatible API base URL for official CL-bench messages mode.")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI-compatible API key for official CL-bench messages mode.")
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--domain", type=str, default="cl_bench")
    parser.add_argument("--decision_method", type=str, default="FinalRefer")
    parser.add_argument("--official_messages", action="store_true", default=True,
                        help="Use each sample's official messages directly when present.")
    parser.add_argument("--optimized_spatial", action="store_true")
    parser.add_argument("--optimized_temporal", action="store_true")
    parser.add_argument("--diff", action="store_true")
    parser.add_argument("--dec", action="store_true")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--llm_enable_thinking", type=parse_bool_arg, default=True,
                        help="Enable model thinking mode: true or false")
    parser.add_argument("--llm_thinking_budget", type=int, default=8192,
                        help="Thinking token budget (unified interface): "
                             "Qwen3-style providers use this as token budget (128-32768); "
                             "MiniMax-M3 uses >0 for adaptive, 0 for disabled")
    args = parser.parse_args()

    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    if args.domain != "cl_bench":
        parser.error("run_cl_bench.py requires --domain cl_bench.")
    if args.limit_questions < 0:
        parser.error("--limit_questions must be >= 0. Use 0 to run the full split.")

    (AgentPrune_ROOT / "result").mkdir(exist_ok=True)
    return args


def _freeze_parameters(parameters):
    if hasattr(parameters, "requires_grad_"):
        parameters.requires_grad_(False)
        return
    for parameter in parameters:
        parameter.requires_grad_(False)


def _loader(dataset: CLBenchDataset, batch_size: int, limit_questions: Optional[int]) -> Iterator[List[Dict[str, Any]]]:
    records = []
    for i_record, record in enumerate(dataset):
        if limit_questions is not None and i_record >= limit_questions:
            break
        records.append(record)
        if len(records) >= batch_size:
            yield records
            records = []
    if records:
        yield records


def _default_result_file(args) -> Path:
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{AgentPrune_ROOT}/result/{args.domain}")
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir / f"{args.domain}_{args.llm_name}_{current_time}.json"


def _score_name(score: Optional[Dict[str, bool]]) -> str:
    if score is None:
        return "unscored"
    if score.get("exact_match"):
        return "exact_match"
    if score.get("contains_match"):
        return "contains_match"
    return "incorrect"


def _normalize_openai_base_url(url: Optional[str]) -> str:
    if not url:
        return ""
    normalized = url.strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    return normalized


def _official_openai_client(args) -> AsyncOpenAI:
    base_url = _normalize_openai_base_url(
        args.api_base_url
        or os.getenv("CL_BENCH_BASE_URL")
        or os.getenv("BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
    )
    api_key = (
        args.api_key
        or os.getenv("CL_BENCH_API_KEY")
        or os.getenv("API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not base_url:
        raise ValueError("Missing API base URL. Set --api_base_url, CL_BENCH_BASE_URL, BASE_URL, or OPENAI_BASE_URL.")
    if not api_key:
        raise ValueError("Missing API key. Set --api_key, CL_BENCH_API_KEY, API_KEY, or OPENAI_API_KEY.")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


async def _official_messages_completion(client: AsyncOpenAI, args, messages: List[Dict[str, Any]]) -> str:
    completion = await client.chat.completions.create(
        model=args.llm_name,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    response = completion.choices[0].message.content
    return response or ""


async def evaluate_cl_bench(graph: Graph, dataset: CLBenchDataset, args) -> Dict[str, Any]:
    print(f"Evaluating AgentDropout on {dataset.__class__.__name__} split {dataset.split}")
    _freeze_parameters(graph.spatial_logits)
    _freeze_parameters(graph.temporal_logits)

    limit_questions = args.limit_questions or None
    data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)
    num_batches = int(math.ceil(data_len / args.batch_size)) if data_len else 0
    result_file = Path(args.result_file) if args.result_file else _default_result_file(args)
    result_file.parent.mkdir(parents=True, exist_ok=True)

    results = []
    exact_correct = 0
    contains_correct = 0
    scored = 0
    total = 0
    official_client = _official_openai_client(args) if args.official_messages else None

    for _, record_batch in tqdm(enumerate(_loader(dataset, args.batch_size, limit_questions)), total=num_batches):
        print(80 * "-")
        start_ts = time.time()
        tasks = []
        for record in record_batch:
            if args.official_messages and record.get("messages"):
                tasks.append(asyncio.create_task(_official_messages_completion(official_client, args, record["messages"])))
            else:
                realized_graph = copy.deepcopy(graph)
                realized_graph.spatial_logits = graph.spatial_logits
                realized_graph.temporal_logits = graph.temporal_logits
                input_dict = dataset.record_to_input(record)
                tasks.append(asyncio.create_task(realized_graph.arun(input_dict, args.num_rounds, case=True)))

        raw_results = await asyncio.gather(*tasks)
        print(f"Batch time {time.time() - start_ts:.3f}")

        for raw_result, record in zip(raw_results, record_batch):
            if args.official_messages and record.get("messages"):
                raw_answer = raw_result
                all_answers = [{"official_messages": True}]
            else:
                raw_answer, _, all_answers = raw_result
            prediction = dataset.postprocess_answer(raw_answer)
            target = dataset.record_to_target_answer(record)
            score = dataset.score(prediction, target)

            total += 1
            if score is not None:
                scored += 1
                exact_correct += int(score["exact_match"])
                contains_correct += int(score["contains_match"])

            updated_item = {
                "id": record.get("id"),
                "idx": record.get("metadata", {}).get("task_id", record.get("id")),
                "messages": record.get("messages"),
                "metadata": record.get("metadata"),
                "task": record.get("task"),
                "context": record.get("context"),
                "answer": target,
                "rubrics": record.get("rubrics"),
                "model_output": raw_answer,
                "response": raw_answer,
                "prediction": prediction,
                "score": score,
                "score_name": _score_name(score),
                "all_answers": all_answers,
            }
            results.append(updated_item)
            print(f"ID: {record.get('id')}")
            print(f"Prediction: {prediction}")
            print(f"Target: {target}")
            print(f"Score: {_score_name(score)}")

        summary = {
            "total": total,
            "scored": scored,
            "exact_match": exact_correct / scored if scored else None,
            "contains_match": contains_correct / scored if scored else None,
            "result_file": str(result_file),
        }
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, indent=4, ensure_ascii=False)
        print(f"Summary: {summary}")
        print_token_stats()

    return {
        "total": total,
        "scored": scored,
        "exact_match": exact_correct / scored if scored else None,
        "contains_match": contains_correct / scored if scored else None,
        "result_file": str(result_file),
    }


async def main():
    args = parse_args()
    dataset = CLBenchDataset(dataset_path=args.dataset_path, dataset_dir=args.dataset_dir, split=args.split)
    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(args.mode, len(agent_names))

    graph = Graph(
        domain=args.domain,
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=args.decision_method,
        optimized_spatial=args.optimized_spatial,
        optimized_temporal=args.optimized_temporal,
        rounds=args.num_rounds,
        diff=args.diff,
        dec=args.dec,
        **kwargs,
    )

    print("Final spatial logits: ", graph.spatial_logits)
    print("Final temporal logits: ", graph.temporal_logits)
    print("Final spatial masks: ", graph.spatial_masks)
    print("Final temporal masks: ", graph.temporal_masks)
    if not args.diff:
        print("Final spatial sparsity:", graph.spatial_masks.sum() / graph.spatial_masks.numel())
        print("Final temporal sparsity:", graph.temporal_masks.sum() / graph.temporal_masks.numel())
    else:
        spatial_sparsity = torch.mean(torch.stack([mask.sum() / mask.numel() for mask in graph.spatial_masks]))
        temporal_sparsity = torch.mean(torch.stack([mask.sum() / mask.numel() for mask in graph.temporal_masks]))
        print("Spatial sparsity (mean):", spatial_sparsity)
        print("Temporal sparsity (mean):", temporal_sparsity)

    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()
    CachedTokens.instance().reset()
    ReasoningTokens.instance().reset()
    BilledPromptTokens.instance().reset()
    BilledCompletionTokens.instance().reset()
    if args.dry_run:
        print("Dry run complete. No model API calls were made.")
        return
    summary = await evaluate_cl_bench(graph, dataset, args)
    print(f"Score: {summary}")


if __name__ == "__main__":
    asyncio.run(main())
