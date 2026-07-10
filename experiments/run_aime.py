import argparse
import asyncio
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import AgentDropout.agents  # noqa: F401
import AgentDropout.prompt  # noqa: F401
from AgentDropout.graph.graph import Graph
from AgentDropout.llm.gpt_chat import configure_thinking, parse_bool_arg
from AgentDropout.utils.const import AgentPrune_ROOT
from AgentDropout.utils.globals import CompletionTokens, Cost, PromptTokens, Time, print_token_stats
from AgentDropout.utils.utils import frobenius_norm, nuclear_norm
from AgentDropout.benchmarks.agent_benchmarks import AgentBenchmarkDataset, BenchmarkRecord, iter_batches


MODES = ["DirectAnswer", "FullConnected", "Random", "Chain", "Debate", "Layered", "Star", "Mesh"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run AIME with BenchmarkAgent.")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--train_split", type=str, default=None,
                        help="Split used only for AgentPrune training.")
    parser.add_argument("--train_data_dir", type=str, default=None,
                        help="Data directory used only for AgentPrune training.")
    parser.add_argument("--train_dataset_path", type=str, default="dataset/AIME/train_extra.jsonl",
                        help="Dataset path used only for AgentPrune training.")
    parser.add_argument("--train_limit_questions", type=int, default=20,
                        help="Number of training questions for AgentPrune. Default 20.")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--limit_questions", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--llm_name", type=str, default="MiniMax-M3")
    parser.add_argument("--mode", type=str, default="FullConnected", choices=MODES)
    parser.add_argument("--lr", type=float, default=0.1, help="AgentPrune learning rate.")
    parser.add_argument("--delta", type=float, default=0.1, help="Regularization margin for fixed masks.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--agent_names", nargs="+", type=str, default=["BenchmarkAgent"])
    parser.add_argument("--agent_nums", nargs="+", type=int, default=[4])
    parser.add_argument("--num_iterations", type=int, default=10,
                        help="Number of AgentPrune optimization iterations.")
    parser.add_argument("--imp_per_iterations", type=int, default=5,
                        help="Prune every N optimization iterations.")
    parser.add_argument("--pruning_rate", type=float, default=0.25,
                        help="Rate of graph edge pruning.")
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--domain", type=str, default="aime")
    parser.add_argument("--decision_method", type=str, default="FinalRefer")
    parser.add_argument("--optimized_spatial", action="store_true")
    parser.add_argument("--optimized_temporal", action="store_true")
    parser.add_argument("--diff", action="store_true")
    parser.add_argument("--dec", action="store_true")
    parser.add_argument(
        "--enabled_tools",
        nargs="*",
        default=None,
        help=(
            "Tools to enable for BenchmarkAgent. Defaults to the AIME benchmark "
            "spec tools ['web_search', 'code_generator']. "
            "Use --enabled_tools with no values for no tools."
        ),
    )
    parser.add_argument("--direct_tool_call", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--web_tool_provider", type=str, default="tavily")
    parser.add_argument("--top_k_results", type=int, default=3)
    parser.add_argument("--max_search_limit", type=int, default=3)
    parser.add_argument("--max_tool_chars", type=int, default=6000)
    parser.add_argument("--max_tool_turns", type=int, default=1,
                        help="Max number of tool calls per agent per round. Default 1 (one-shot tool use).")
    parser.add_argument("--llm_enable_thinking", type=parse_bool_arg, default=True,
                        help="Enable model thinking mode: true or false")
    parser.add_argument("--llm_thinking_budget", type=int, default=8192,
                        help="Thinking token budget for both Qwen3 and MiniMax-M3 providers")
    args = parser.parse_args()
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    if args.limit_questions < 0:
        parser.error("--limit_questions must be >= 0.")
    if args.train_limit_questions < 0:
        parser.error("--train_limit_questions must be >= 0.")
    return args


def get_kwargs(mode: str, n_agents: int, node_kwargs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    def full_connected() -> List[List[int]]:
        return [[1 if i != j else 0 for i in range(n_agents)] for j in range(n_agents)]

    def random_graph() -> List[List[int]]:
        import random

        return [[random.randint(0, 1) if i != j else 0 for i in range(n_agents)] for j in range(n_agents)]

    def layered_graph(layer_num: int = 2) -> List[List[int]]:
        matrix = [[0] * n_agents for _ in range(n_agents)]
        base_size = n_agents // layer_num
        remainder = n_agents % layer_num
        layers = []
        for layer in range(layer_num):
            layers.extend([layer] * (base_size + (1 if layer < remainder else 0)))
        for i in range(n_agents):
            for j in range(n_agents):
                if layers[j] == layers[i] + 1:
                    matrix[i][j] = 1
        return matrix

    def star_graph() -> List[List[int]]:
        matrix = [[0] * n_agents for _ in range(n_agents)]
        for i in range(1, n_agents):
            matrix[0][i] = 1
        return matrix

    def mesh_graph() -> List[List[int]]:
        matrix = [[0] * n_agents for _ in range(n_agents)]
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                matrix[i][j] = 1
        return matrix

    if mode == "DirectAnswer":
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [node_kwargs[0] if node_kwargs else {}]
    elif mode == "FullConnected":
        fixed_spatial_masks = full_connected()
        fixed_temporal_masks = [[1 for _ in range(n_agents)] for _ in range(n_agents)]
    elif mode == "Random":
        fixed_spatial_masks = random_graph()
        fixed_temporal_masks = random_graph()
    elif mode == "Chain":
        fixed_spatial_masks = [[1 if i == j + 1 else 0 for i in range(n_agents)] for j in range(n_agents)]
        fixed_temporal_masks = [[1 if i == 0 and j == n_agents - 1 else 0 for i in range(n_agents)] for j in range(n_agents)]
    elif mode == "Debate":
        fixed_spatial_masks = [[0 for _ in range(n_agents)] for _ in range(n_agents)]
        fixed_temporal_masks = [[1 for _ in range(n_agents)] for _ in range(n_agents)]
    elif mode == "Layered":
        fixed_spatial_masks = layered_graph()
        fixed_temporal_masks = [[1 for _ in range(n_agents)] for _ in range(n_agents)]
    elif mode == "Star":
        fixed_spatial_masks = star_graph()
        fixed_temporal_masks = [[1 for _ in range(n_agents)] for _ in range(n_agents)]
    elif mode == "Mesh":
        fixed_spatial_masks = mesh_graph()
        fixed_temporal_masks = [[1 for _ in range(n_agents)] for _ in range(n_agents)]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return {
        "initial_spatial_probability": 0.5,
        "fixed_spatial_masks": fixed_spatial_masks,
        "initial_temporal_probability": 0.5,
        "fixed_temporal_masks": fixed_temporal_masks,
        "node_kwargs": node_kwargs,
    }


def default_result_file(args, dataset: AgentBenchmarkDataset) -> Path:
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{AgentPrune_ROOT}/result/{dataset.name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir / f"{dataset.name}_{dataset.split}_{args.llm_name}_{current_time}.json"


def selected_records(dataset: AgentBenchmarkDataset, limit_questions: int) -> List[BenchmarkRecord]:
    records = list(dataset)
    return records[:limit_questions] if limit_questions else records


def build_node_kwargs(
    mode: str,
    agent_names: List[str],
    base_tool_config: Dict[str, Any],
    enabled_tools: List[str],
) -> List[Dict[str, Any]]:
    if mode != "FullConnected" or any(agent_name != "BenchmarkAgent" for agent_name in agent_names):
        return [{"tool_config": dict(base_tool_config)} for _ in agent_names]

    specialists = []
    if "code_generator" in enabled_tools:
        specialists.append("Code Execution Specialist")
    specialists.extend(["Independent Solver", "Synthesis Verifier"])

    node_kwargs = []
    for idx, _ in enumerate(agent_names):
        role = specialists[idx % len(specialists)]
        tool_config = dict(base_tool_config)
        node_kwargs.append({"role": role, "tool_config": tool_config})
    return node_kwargs


def build_graph(args, dataset: AgentBenchmarkDataset) -> Graph:
    graph, _ = build_graph_with_kwargs(args, dataset)
    return graph


def build_graph_with_kwargs(args, dataset: AgentBenchmarkDataset) -> tuple[Graph, Dict[str, Any]]:
    enabled_tools = dataset.default_tools if args.enabled_tools is None else list(args.enabled_tools)
    tool_config = {
        "enabled_tools": enabled_tools,
        "direct_tool_call": args.direct_tool_call,
        "web_tool_provider": args.web_tool_provider,
        "top_k_results": args.top_k_results,
        "max_search_limit": args.max_search_limit,
        "max_tool_chars": args.max_tool_chars,
        "max_tool_turns": args.max_tool_turns,
    }
    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    if args.mode == "DirectAnswer":
        agent_names = agent_names[:1] or ["BenchmarkAgent"]
    node_kwargs = build_node_kwargs(args.mode, agent_names, tool_config, enabled_tools)
    kwargs = get_kwargs(args.mode, len(agent_names), node_kwargs=node_kwargs)
    graph = Graph(
        domain=args.domain or dataset.name,
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
    return graph, kwargs


def _agent_count(graph: Graph) -> int:
    return len(graph.agent_names)


def _mask_matrix(mask_values: List[List[int]], n_agents: int) -> torch.Tensor:
    return torch.tensor(mask_values, dtype=torch.float32).reshape((n_agents, n_agents))


def _reshape_logits(logits, n_agents: int):
    if isinstance(logits, torch.nn.ParameterList):
        return [param.reshape((n_agents, n_agents)) for param in logits]
    return logits.reshape((n_agents, n_agents))


def _regularization_loss(graph: Graph, kwargs: Dict[str, Any], args) -> torch.Tensor:
    n_agents = _agent_count(graph)
    spatial_fixed = _mask_matrix(kwargs["fixed_spatial_masks"], n_agents)
    temporal_fixed = _mask_matrix(kwargs["fixed_temporal_masks"], n_agents)
    spatial_train = _reshape_logits(graph.spatial_logits, n_agents)
    temporal_train = _reshape_logits(graph.temporal_logits, n_agents)

    if not graph.diff:
        loss_s = nuclear_norm(spatial_train)
        loss_t = nuclear_norm(temporal_train)
        frob_loss_s = frobenius_norm(spatial_fixed, spatial_train)
        frob_loss_t = frobenius_norm(temporal_fixed, temporal_train)
    else:
        loss_s = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in spatial_train]))
        loss_t = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in temporal_train]))
        frob_loss_s = torch.mean(torch.stack([frobenius_norm(spatial_fixed, matrix) for matrix in spatial_train]))
        frob_loss_t = torch.mean(torch.stack([frobenius_norm(temporal_fixed, matrix) for matrix in temporal_train]))

    return loss_s + loss_t + F.relu(frob_loss_s - args.delta) + F.relu(frob_loss_t - args.delta)


async def train_agentprune(graph: Graph, dataset: AgentBenchmarkDataset, args, kwargs: Dict[str, Any]) -> None:
    records = selected_records(dataset, args.train_limit_questions)
    if not records:
        print("No AIME records selected for AgentPrune training.")
        return
    print(f"AgentPrune training records: {len(records)} / {len(dataset)}")

    if not graph.diff:
        optimizer = torch.optim.Adam([graph.spatial_logits, graph.temporal_logits], lr=args.lr)
    else:
        optimizer = torch.optim.Adam(
            list(graph.spatial_logits.parameters()) + list(graph.temporal_logits.parameters()),
            lr=args.lr,
        )

    graph.optimized_spatial = args.optimized_spatial
    graph.optimized_temporal = args.optimized_temporal
    loader_index = 0

    for i_iter in range(args.num_iterations):
        print(f"Train {i_iter + 1}/{args.num_iterations}", 80 * "-")
        start_ts = time.time()
        tasks = []
        correct_records = []
        add_losses = []

        for _ in range(args.batch_size):
            record = records[loader_index % len(records)]
            loader_index += 1
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits
            input_dict = dataset.record_to_input(record)
            tasks.append(asyncio.create_task(realized_graph.arun(input_dict, args.num_rounds, compress_messages=False)))
            correct_records.append(record)
            add_losses.append(_regularization_loss(realized_graph, kwargs, args))

        raw_results = await asyncio.gather(*tasks)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        answers: List[str] = []

        for raw_result, add_loss, record in zip(raw_results, add_losses, correct_records):
            raw_answer, log_prob = raw_result
            answer_text = raw_answer[0] if isinstance(raw_answer, list) and raw_answer else raw_answer
            score = dataset.score(answer_text, record)
            utility = float(score.get("score", score.get("accuracy", float(score.get("correct", False)))))
            utilities.append(utility)
            answers.append(str(score.get("prediction", dataset.postprocess_answer(answer_text))))
            loss_list.append(-log_prob * utility + add_loss)
            print(f"ID: {record.question_id}")
            print(f"Prediction: {answers[-1]}")
            print(f"Target: {record.answer}")
            print(f"Correct: {score.get('correct', False)}")

        total_loss = torch.mean(torch.stack(loss_list))
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if not graph.diff:
            spatial_probs = torch.sigmoid(graph.spatial_logits)
            temporal_probs = torch.sigmoid(graph.temporal_logits)
        else:
            spatial_probs = [torch.sigmoid(logit) for logit in graph.spatial_logits]
            temporal_probs = [torch.sigmoid(logit) for logit in graph.temporal_logits]

        print("answers:", answers)
        print(f"Batch time {time.time() - start_ts:.3f}")
        print("utilities:", utilities)
        print("loss:", total_loss.item())
        print("Spatial logits:", graph.spatial_logits)
        print("Temporal logits:", graph.temporal_logits)
        print("Spatial probs:", spatial_probs)
        print("Temporal probs:", temporal_probs)
        print_token_stats()

        if (i_iter + 1) % args.imp_per_iterations == 0:
            if not graph.diff:
                spatial_masks, temporal_masks = graph.update_masks(args.pruning_rate)
            else:
                spatial_masks, temporal_masks = graph.update_masks_diff(args.pruning_rate)
            print("spatial masks:", spatial_masks)
            print("temporal masks:", temporal_masks)
            if not graph.diff:
                print("spatial sparsity:", spatial_masks.sum() / spatial_masks.numel())
                print("temporal sparsity:", temporal_masks.sum() / temporal_masks.numel())
            elif len(spatial_masks) > 0:
                print("spatial sparsity:", spatial_masks[0].sum() / spatial_masks[0].numel())
                if len(temporal_masks) > 0:
                    print("temporal sparsity:", temporal_masks[0].sum() / temporal_masks[0].numel())


async def evaluate(graph: Graph, dataset: AgentBenchmarkDataset, args) -> Dict[str, Any]:
    records = selected_records(dataset, args.limit_questions)
    result_file = Path(args.result_file) if args.result_file else default_result_file(args, dataset)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    configured_tools = dataset.default_tools if args.enabled_tools is None else list(args.enabled_tools)

    if args.dry_run:
        tool_usage = {tool: 0 for tool in configured_tools}
        tool_tokens = {tool: 0 for tool in configured_tools}
        summary = {
            "benchmark": dataset.name,
            "split": dataset.split,
            "loaded": len(dataset),
            "selected": len(records),
            "expected_samples": dataset.expected_samples(),
            "default_tools": dataset.default_tools,
            "tool_usage": tool_usage,
            "tool_tokens": tool_tokens,
            "tool_usage_by_agent": {},
            "tool_usage_by_dataset": {dataset.name: tool_usage},
            "result_file": str(result_file),
            "dry_run": True,
        }
        result_file.write_text(json.dumps({"summary": summary, "results": []}, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    results = []
    correct = 0
    total = 0
    tool_totals: Dict[str, int] = {tool: 0 for tool in configured_tools}
    tool_token_totals: Dict[str, int] = {tool: 0 for tool in configured_tools}
    agent_tool_totals: Dict[str, Dict[str, Any]] = {}
    num_batches = int(math.ceil(len(records) / args.batch_size)) if records else 0

    for record_batch in tqdm(iter_batches(records, args.batch_size), total=num_batches):
        tasks = []
        task_inputs = []
        for record in record_batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits
            input_dict = dataset.record_to_input(record)
            task_inputs.append(input_dict)
            tasks.append(asyncio.create_task(realized_graph.arun(input_dict, args.num_rounds, case=True)))

        raw_results = await asyncio.gather(*tasks)
        for raw_result, record, input_dict in zip(raw_results, record_batch, task_inputs):
            raw_answer, _, all_answers = raw_result
            answer_text = raw_answer[0] if isinstance(raw_answer, list) and raw_answer else raw_answer
            score = dataset.score(answer_text, record)
            tool_counts = dict(input_dict.get("metadata", {}).get("tool_counts", {}))
            tool_tokens = dict(input_dict.get("metadata", {}).get("tool_tokens", {}))
            tool_counts_by_agent = dict(input_dict.get("metadata", {}).get("tool_counts_by_agent", {}))
            tool_history = list(input_dict.get("metadata", {}).get("tool_history", []))
            for tool_name, count in tool_counts.items():
                tool_totals[tool_name] = int(tool_totals.get(tool_name, 0)) + int(count)
            for tool_name, token_count in tool_tokens.items():
                tool_token_totals[tool_name] = int(tool_token_totals.get(tool_name, 0)) + int(token_count)
            for agent_id, agent_counts in tool_counts_by_agent.items():
                agent_entry = agent_tool_totals.setdefault(
                    agent_id,
                    {
                        "id": agent_id,
                        "role": agent_counts.get("role", ""),
                        "agent_name": agent_counts.get("agent_name", ""),
                        "tools": {},
                        "tool_tokens": {},
                        "total": 0,
                        "total_tool_tokens": 0,
                    },
                )
                agent_entry["role"] = agent_counts.get("role", agent_entry.get("role", ""))
                agent_entry["agent_name"] = agent_counts.get("agent_name", agent_entry.get("agent_name", ""))
                for tool_name, count in dict(agent_counts.get("tools", {})).items():
                    count = int(count)
                    agent_entry["tools"][tool_name] = int(agent_entry["tools"].get(tool_name, 0)) + count
                    agent_entry["total"] = int(agent_entry.get("total", 0)) + count
                for tool_name, token_count in dict(agent_counts.get("tool_tokens", {})).items():
                    token_count = int(token_count)
                    agent_entry["tool_tokens"][tool_name] = int(agent_entry["tool_tokens"].get(tool_name, 0)) + token_count
                    agent_entry["total_tool_tokens"] = int(agent_entry.get("total_tool_tokens", 0)) + token_count
            total += 1
            correct += int(score.get("correct", False))
            results.append(
                {
                    "id": record.question_id,
                    "question": record.question,
                    "answer": record.answer,
                    "metadata": record.metadata,
                    "response": raw_answer,
                    "prediction": score.get("prediction", dataset.postprocess_answer(answer_text)),
                    "score": score,
                    "all_answers": all_answers,
                    "tool_counts": tool_counts,
                    "tool_tokens": tool_tokens,
                    "tool_counts_by_agent": tool_counts_by_agent,
                    "tool_history": tool_history,
                }
            )
            print(f"ID: {record.question_id}")
            print(f"Prediction: {results[-1]['prediction']}")
            print(f"Target: {record.answer}")
            print(f"Correct: {score.get('correct', False)}")

        summary = {
            "benchmark": dataset.name,
            "split": dataset.split,
            "loaded": len(dataset),
            "selected": len(records),
            "expected_samples": dataset.expected_samples(),
            "accuracy": correct / total if total else None,
            "correct": correct,
            "total": total,
            "default_tools": dataset.default_tools,
            "tool_usage": tool_totals,
            "tool_tokens": tool_token_totals,
            "tool_usage_by_agent": agent_tool_totals,
            "tool_usage_by_dataset": {dataset.name: tool_totals},
            "result_file": str(result_file),
        }
        result_file.write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Summary: {summary}")
        print_token_stats()

    return {
        "benchmark": dataset.name,
        "split": dataset.split,
        "accuracy": correct / total if total else None,
        "correct": correct,
        "total": total,
        "tool_usage": tool_totals,
        "tool_tokens": tool_token_totals,
        "tool_usage_by_agent": agent_tool_totals,
        "result_file": str(result_file),
    }


async def main():
    args = parse_args()
    configure_thinking(
        enable_thinking=args.llm_enable_thinking,
        thinking_budget=args.llm_thinking_budget,
    )
    dataset = AgentBenchmarkDataset(
        name="aime",
        split=args.split,
        data_dir=args.data_dir,
        dataset_path=args.dataset_path,
    )
    graph, kwargs = build_graph_with_kwargs(args, dataset)
    if not args.dry_run and (args.optimized_spatial or args.optimized_temporal):
        if not (args.train_dataset_path or args.train_split or args.train_data_dir):
            raise ValueError(
                "AIME AgentPrune training requires an independent training dataset. "
                "Please set --train_dataset_path or --train_split/--train_data_dir so training "
                "does not reuse the 60 evaluation questions."
            )
        if args.train_dataset_path and not Path(args.train_dataset_path).exists():
            raise FileNotFoundError(
                f"AIME AgentPrune training dataset not found: {args.train_dataset_path}. "
                "Create this independent jsonl file or pass --train_dataset_path to another "
                "file that does not overlap with the evaluation questions."
            )
        train_dataset = AgentBenchmarkDataset(
            name="aime",
            split=args.train_split or args.split,
            data_dir=args.train_data_dir or args.data_dir,
            dataset_path=args.train_dataset_path,
        )
        eval_ids = {record.question_id for record in selected_records(dataset, args.limit_questions)}
        overlap_ids = eval_ids.intersection(record.question_id for record in selected_records(train_dataset, args.train_limit_questions))
        if overlap_ids:
            raise ValueError(
                f"Training dataset overlaps with evaluation questions: {sorted(overlap_ids)[:10]}. "
                "Please provide a separate AIME training file."
            )
        await train_agentprune(graph, train_dataset, args, kwargs)
        print("Final spatial logits:", graph.spatial_logits)
        print("Final temporal logits:", graph.temporal_logits)
        print("Final spatial masks:", graph.spatial_masks)
        print("Final temporal masks:", graph.temporal_masks)
    summary = await evaluate(graph, dataset, args)
    print(f"Score: {summary}")


if __name__ == "__main__":
    asyncio.run(main())
