import sys
import os
import argparse
import asyncio
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from AgentPrune.graph.graph import Graph
from AgentPrune.utils.const import AgentPrune_ROOT
from AgentPrune.utils.globals import CompletionTokens, PromptTokens, Time
from dataset.longbench_dataset import (
    extract_answer_text,
    load_longbench_dataset,
    score_longbench_answer,
)


def load_result(result_file: Path) -> List[Dict[str, Any]]:
    if not result_file.exists():
        with open(result_file, "w", encoding="utf-8") as file:
            json.dump([], file)

    with open(result_file, "r", encoding="utf-8") as file:
        return json.load(file)


def dataloader(data_list: List[Dict[str, Any]], batch_size: int, i_batch: int) -> List[Dict[str, Any]]:
    return data_list[i_batch * batch_size : i_batch * batch_size + batch_size]


def get_agent_token_snapshot(graph: Graph) -> Dict[str, Dict[str, Any]]:
    return copy.deepcopy(graph.agent_token_stats)


def get_phase_agent_token_metrics(
    graph: Graph,
    start_snapshot: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    phase_metrics: List[Dict[str, Any]] = []
    for node_id, current_stats in graph.agent_token_stats.items():
        previous_stats = start_snapshot.get(node_id, {})
        prompt_tokens = current_stats["prompt_tokens"] - previous_stats.get("prompt_tokens", 0)
        completion_tokens = current_stats["completion_tokens"] - previous_stats.get("completion_tokens", 0)
        total_tokens = current_stats["total_tokens"] - previous_stats.get("total_tokens", 0)
        calls = current_stats["calls"] - previous_stats.get("calls", 0)

        if prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0 and calls == 0:
            continue

        phase_metrics.append(
            {
                "id": current_stats["id"],
                "agent_name": current_stats["agent_name"],
                "role": current_stats["role"],
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "calls": calls,
            }
        )

    return sorted(phase_metrics, key=lambda item: (-item["total_tokens"], item["role"], item["id"]))


def get_token_snapshot():
    return PromptTokens.instance().value, CompletionTokens.instance().value


def print_phase_token_summary(phase_name: str, prompt_used: float, completion_used: float):
    total_used = prompt_used + completion_used
    print(f"{phase_name} Prompt Tokens: {prompt_used}")
    print(f"{phase_name} Completion Tokens: {completion_used}")
    print(f"{phase_name} Total Tokens: {total_used}")


def print_phase_agent_token_summary(phase_name: str, phase_metrics: List[Dict[str, Any]]):
    print(f"{phase_name} Per-Agent Token Usage:")
    if len(phase_metrics) == 0:
        print("  No agent token usage recorded.")
        return

    for item in phase_metrics:
        print(
            f"  {item['role']} ({item['id']}, {item['agent_name']}): "
            f"input={item['prompt_tokens']}, output={item['completion_tokens']}, "
            f"total={item['total_tokens']}, calls={item['calls']}"
        )


def print_agent_token_summary(graph: Graph):
    print("Per-Agent Token Usage:")
    for item in graph.get_agent_token_metrics():
        print(
            f"  {item['role']} ({item['id']}, {item['agent_name']}): "
            f"input={item['prompt_tokens']}, output={item['completion_tokens']}, "
            f"total={item['total_tokens']}, calls={item['calls']}"
        )

    print("Per-Role Token Usage:")
    for item in graph.get_role_token_metrics():
        print(
            f"  {item['role']}: input={item['prompt_tokens']}, "
            f"output={item['completion_tokens']}, total={item['total_tokens']}, "
            f"calls={item['calls']}, agents={item['agents']}"
        )


def freeze_graph_for_evaluation(graph: Graph) -> None:
    graph.optimized_spatial = False
    graph.optimized_temporal = False
    graph.spatial_logits.requires_grad_(False)
    graph.temporal_logits.requires_grad_(False)


def parse_args():
    parser = argparse.ArgumentParser(description="AgentPrune Experiments on LongBench")
    parser.add_argument("--dataset_path", type=str, default="dataset/longbench")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="gpt-4-1106-preview")
    parser.add_argument(
        "--mode",
        type=str,
        default="FullConnected",
        choices=["DirectAnswer", "FullConnected", "Random", "Chain", "Debate", "Layered", "Star", "Mesh"],
    )
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--imp_per_iterations", type=int, default=5)
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--pruning_rate", type=float, default=0.25)
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--domain", type=str, default="longbench")
    parser.add_argument("--agent_names", nargs="+", type=str, default=["AnalyzeAgent"])
    parser.add_argument("--agent_nums", nargs="+", type=int, default=[4])
    parser.add_argument("--decision_method", type=str, default="FinalRefer")
    parser.add_argument("--limit_questions", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with_reasoning", action="store_true")
    parser.add_argument("--log_agent_outputs", action="store_true")
    parser.add_argument("--freeze_after_training", action="store_true", default=True)
    parser.add_argument("--optimized_spatial", action="store_true")
    parser.add_argument("--optimized_temporal", action="store_true")
    args = parser.parse_args()

    result_path = AgentPrune_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    if args.domain != "longbench":
        parser.error("LongBench runner requires --domain longbench.")
    return args


async def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = load_longbench_dataset(
        args.dataset_path,
        args.dataset_name,
        with_reasoning=args.with_reasoning,
    )
    if args.limit_questions is not None:
        dataset = dataset[: args.limit_questions]

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = AgentPrune_ROOT / "result" / "longbench"
    result_dir.mkdir(parents=True, exist_ok=True)
    safe_llm_name = args.llm_name.replace("/", "_")
    result_file = (
        Path(args.result_file)
        if args.result_file is not None
        else result_dir / f"{args.dataset_name}_{safe_llm_name}_{current_time}.json"
    )

    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(args.mode, len(agent_names))
    graph = Graph(
        domain=args.domain,
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=args.decision_method,
        log_node_outputs=args.log_agent_outputs,
        llm_temperature=args.temperature,
        optimized_spatial=args.optimized_spatial,
        optimized_temporal=args.optimized_temporal,
        **kwargs,
    )

    train_mode_enabled = args.optimized_spatial or args.optimized_temporal
    graph.set_eval_mode(not train_mode_enabled)
    optimizer = torch.optim.Adam([graph.spatial_logits, graph.temporal_logits], lr=args.lr)

    num_batches = int(math.ceil(len(dataset) / args.batch_size))
    training_batch_limit = min(num_batches, args.num_iterations) if train_mode_enabled else 0
    no_training_mode = training_batch_limit == 0
    training_phase_closed = no_training_mode
    train_prompt_start, train_completion_start = get_token_snapshot()
    train_agent_start = get_agent_token_snapshot(graph)
    inference_prompt_start = train_prompt_start if no_training_mode else None
    inference_completion_start = train_completion_start if no_training_mode else None
    inference_agent_start = train_agent_start if no_training_mode else None

    total_score = 0.0
    total_executed = 0

    progress_bar = tqdm(range(num_batches), desc=f"LongBench/{args.dataset_name}", dynamic_ncols=True)
    for i_batch in progress_bar:
        print(f"Batch {i_batch}", 80 * "-")
        start_ts = time.time()
        current_batch = dataloader(dataset, args.batch_size, i_batch)
        if not current_batch:
            break

        answer_log_probs = []
        for i_record, record in enumerate(current_batch):
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits
            realized_graph.share_cost_state(graph)
            realized_graph.current_sample_index = i_batch * args.batch_size + i_record + 1
            answer_log_probs.append(
                asyncio.create_task(realized_graph.arun({"task": record["task"]}, args.num_rounds))
            )

        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        data = load_result(result_file)

        for record, answer, log_prob in zip(current_batch, raw_answers, log_probs):
            prediction = extract_answer_text(answer[0])
            score = score_longbench_answer(
                prediction,
                record["answers"],
                record["dataset"],
                record.get("all_classes", []),
            )
            total_score += score
            total_executed += 1
            average_score = total_score / total_executed
            average_score_percent = round(100 * average_score, 2)
            utilities.append(score)
            loss_list.append(-log_prob * score)

            data.append(
                {
                    "ID": record["id"],
                    "Dataset": record["dataset"],
                    "Input": record["input"],
                    "Answers": record["answers"],
                    "All classes": record.get("all_classes", []),
                    "Length": record.get("length"),
                    "Response": answer,
                    "Prediction": prediction,
                    "Score": score,
                    "Score percent": round(100 * score, 2),
                    "Total score": total_score,
                    "Total executed": total_executed,
                    "Average score": average_score,
                    "Average score percent": average_score_percent,
                }
            )

        with open(result_file, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

        total_loss = torch.mean(torch.stack(loss_list))
        if train_mode_enabled and (args.optimized_spatial or args.optimized_temporal):
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"Batch time {time.time() - start_ts:.3f}")
        print(f"Average score: {average_score}")
        print(f"Average score percent: {average_score_percent}")
        progress_bar.set_postfix(avg_score=f"{average_score_percent:.2f}")

        if (
            (i_batch + 1) % args.imp_per_iterations == 0
            and i_batch < args.num_iterations
            and (args.optimized_spatial or args.optimized_temporal)
        ):
            graph.update_masks(args.pruning_rate)
        if train_mode_enabled and i_batch + 1 == args.num_iterations:
            args.optimized_spatial = False
            args.optimized_temporal = False
            if args.freeze_after_training:
                freeze_graph_for_evaluation(graph)
            graph.set_eval_mode(True)
        if (not training_phase_closed) and (i_batch + 1 == training_batch_limit):
            train_prompt_used = PromptTokens.instance().value - train_prompt_start
            train_completion_used = CompletionTokens.instance().value - train_completion_start
            train_agent_metrics = get_phase_agent_token_metrics(graph, train_agent_start)
            print_phase_token_summary("Training", train_prompt_used, train_completion_used)
            print_phase_agent_token_summary("Training", train_agent_metrics)
            inference_prompt_start, inference_completion_start = get_token_snapshot()
            inference_agent_start = get_agent_token_snapshot(graph)
            training_phase_closed = True

    if no_training_mode:
        print("Training skipped: both optimized_spatial and optimized_temporal are disabled.")
        print_phase_token_summary("Training", 0.0, 0.0)
        print_phase_agent_token_summary("Training", [])

    if inference_prompt_start is None or inference_completion_start is None or inference_agent_start is None:
        inference_prompt_start, inference_completion_start = get_token_snapshot()
        inference_agent_start = get_agent_token_snapshot(graph)

    inference_prompt_used = PromptTokens.instance().value - inference_prompt_start
    inference_completion_used = CompletionTokens.instance().value - inference_completion_start
    inference_agent_metrics = get_phase_agent_token_metrics(graph, inference_agent_start)
    print_phase_token_summary("Inference", inference_prompt_used, inference_completion_used)
    print_phase_agent_token_summary("Inference", inference_agent_metrics)
    final_score = total_score / total_executed if total_executed else 0.0
    print(f"Score: {final_score}")
    print(f"Score percent: {round(100 * final_score, 2)}")
    print(f"Prompt Tokens: {PromptTokens.instance().value}")
    print(f"Completion Tokens: {CompletionTokens.instance().value}")
    print(f"Total Tokens: {PromptTokens.instance().value + CompletionTokens.instance().value}")
    print_agent_token_summary(graph)
    print(f"Result file: {result_file}")


def get_kwargs(
    mode: Union[
        Literal["DirectAnswer"],
        Literal["FullConnected"],
        Literal["Random"],
        Literal["Chain"],
        Literal["Debate"],
        Literal["Layered"],
        Literal["Star"],
        Literal["Mesh"],
    ],
    N: int,
):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks: List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks: List[List[int]] = None
    node_kwargs = None

    def generate_layered_graph(n, layer_num=2):
        adj_matrix = [[0] * n for _ in range(n)]
        base_size = n // layer_num
        remainder = n % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(n):
            current_layer = layers[i]
            for j in range(n):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix

    def generate_star_graph(n):
        adj_matrix = [[0] * n for _ in range(n)]
        for i in range(1, n):
            adj_matrix[0][i] = 1
        return adj_matrix

    def generate_mesh_graph(n):
        adj_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                adj_matrix[i][j] = 1
        return adj_matrix

    if mode == "DirectAnswer":
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{"role": "Normal"}]
    elif mode == "FullConnected":
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Random":
        fixed_spatial_masks = [[random.randint(0, 1) if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode == "Chain":
        fixed_spatial_masks = [[1 if i == j + 1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i == 0 and j == N - 1 else 0 for i in range(N)] for j in range(N)]
    elif mode == "Debate":
        fixed_spatial_masks = [[0 for _ in range(N)] for _ in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Layered":
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Star":
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Mesh":
        fixed_spatial_masks = generate_mesh_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]

    if node_kwargs is None:
        node_kwargs = [{} for _ in range(N)]

    return {
        "initial_spatial_probability": initial_spatial_probability,
        "fixed_spatial_masks": fixed_spatial_masks,
        "initial_temporal_probability": initial_temporal_probability,
        "fixed_temporal_masks": fixed_temporal_masks,
        "node_kwargs": node_kwargs,
    }


if __name__ == "__main__":
    asyncio.run(main())
