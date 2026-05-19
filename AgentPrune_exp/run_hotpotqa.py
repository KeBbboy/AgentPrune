import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import asyncio
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from AgentPrune.graph.graph import Graph
from AgentPrune.utils.globals import CompletionTokens, PromptTokens
from AgentPrune_exp.train_hotpotqa import train_hotpotqa
from dataset.hotpotqa_dataset import CENTRALIZED_INPUT, FULL_BROADCAST, HotpotQADataset


class HotpotQAAnswerMetrics:
    def __init__(self) -> None:
        self._totals = {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}
        self._num_total = 0

    def update(self, predicted: str, target: str) -> None:
        scores = HotpotQADataset.answer_metrics(predicted, target)
        for key, value in scores.items():
            self._totals[key] += value
        self._num_total += 1

    def get(self) -> Dict[str, float]:
        if self._num_total == 0:
            return {key: 0.0 for key in self._totals}
        return {key: value / self._num_total for key, value in self._totals.items()}

    def print(self) -> None:
        scores = self.get()
        print(
            "HotpotQA Answer Metrics: "
            f"EM={scores['em'] * 100:.2f}, "
            f"F1={scores['f1'] * 100:.2f}, "
            f"Precision={scores['prec'] * 100:.2f}, "
            f"Recall={scores['recall'] * 100:.2f} "
            f"({self._num_total} labeled examples)"
        )


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

        phase_metrics.append({
            "id": current_stats["id"],
            "agent_name": current_stats["agent_name"],
            "role": current_stats["role"],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "calls": calls,
        })

    return sorted(
        phase_metrics,
        key=lambda item: (-item["total_tokens"], item["role"], item["id"]),
    )


def get_token_snapshot() -> tuple[float, float]:
    return PromptTokens.instance().value, CompletionTokens.instance().value


def print_phase_token_summary(phase_name: str, prompt_used: float, completion_used: float) -> None:
    total_used = prompt_used + completion_used
    print(f"{phase_name} Prompt Tokens: {prompt_used}")
    print(f"{phase_name} Completion Tokens: {completion_used}")
    print(f"{phase_name} Total Tokens: {total_used}")


def print_phase_agent_token_summary(phase_name: str, phase_metrics: List[Dict[str, Any]]) -> None:
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


def print_agent_token_summary(graph: Graph) -> None:
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


def print_retained_edge_message_summary(graph: Graph) -> None:
    summary = graph.get_retained_edge_message_summary()
    print(
        f"Retained Spatial Edges: {summary['spatial_edge_count']}, "
        f"Retained Temporal Edges: {summary['temporal_edge_count']}"
    )

    for edge_kind, edge_key in [("Spatial", "spatial_edges"), ("Temporal", "temporal_edges")]:
        print(f"{edge_kind} Edge Messages:")
        if len(summary[edge_key]) == 0:
            print("  None")
            continue
        for edge in summary[edge_key]:
            print(
                f"  {edge['src_role']} ({edge['src_id']}) -> "
                f"{edge['dst_role']} ({edge['dst_id']}): {edge['message_type']}"
            )


def get_kwargs(mode: str, node_count: int) -> Dict[str, Any]:
    initial_spatial_probability = 0.5
    initial_temporal_probability = 0.5
    fixed_spatial_masks: List[List[int]]
    fixed_temporal_masks: List[List[int]]

    def generate_layered_graph(n: int, layer_num: int = 2) -> List[List[int]]:
        adj_matrix = [[0] * n for _ in range(n)]
        base_size = n // layer_num
        remainder = n % layer_num
        layers = []
        for layer_index in range(layer_num):
            size = base_size + (1 if layer_index < remainder else 0)
            layers.extend([layer_index] * size)
        random.shuffle(layers)
        for i in range(n):
            current_layer = layers[i]
            for j in range(n):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix

    def generate_mesh_graph(n: int) -> List[List[int]]:
        adj_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                adj_matrix[i][j] = 1
        return adj_matrix

    def generate_star_graph(n: int) -> List[List[int]]:
        adj_matrix = [[0] * n for _ in range(n)]
        for i in range(1, n):
            adj_matrix[0][i] = 1
        return adj_matrix

    if mode == "DirectAnswer":
        fixed_spatial_masks = [[0 for _ in range(node_count)] for _ in range(node_count)]
        fixed_temporal_masks = [[0 for _ in range(node_count)] for _ in range(node_count)]
    elif mode == "FullConnected":
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(node_count)] for j in range(node_count)]
        fixed_temporal_masks = [[1 for _ in range(node_count)] for _ in range(node_count)]
    elif mode == "Random":
        fixed_spatial_masks = [
            [random.randint(0, 1) if i != j else 0 for i in range(node_count)]
            for j in range(node_count)
        ]
        fixed_temporal_masks = [
            [random.randint(0, 1) for _ in range(node_count)]
            for _ in range(node_count)
        ]
    elif mode == "Chain":
        fixed_spatial_masks = [[1 if i == j + 1 else 0 for i in range(node_count)] for j in range(node_count)]
        fixed_temporal_masks = [[1 if i == 0 and j == node_count - 1 else 0 for i in range(node_count)] for j in range(node_count)]
    elif mode == "Debate":
        fixed_spatial_masks = [[0 for _ in range(node_count)] for _ in range(node_count)]
        fixed_temporal_masks = [[1 for _ in range(node_count)] for _ in range(node_count)]
    elif mode == "Layered":
        fixed_spatial_masks = generate_layered_graph(node_count)
        fixed_temporal_masks = [[1 for _ in range(node_count)] for _ in range(node_count)]
    elif mode == "Mesh":
        fixed_spatial_masks = generate_mesh_graph(node_count)
        fixed_temporal_masks = [[1 for _ in range(node_count)] for _ in range(node_count)]
    elif mode == "Star":
        fixed_spatial_masks = generate_star_graph(node_count)
        fixed_temporal_masks = [[1 for _ in range(node_count)] for _ in range(node_count)]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return {
        "initial_spatial_probability": initial_spatial_probability,
        "fixed_spatial_masks": fixed_spatial_masks,
        "initial_temporal_probability": initial_temporal_probability,
        "fixed_temporal_masks": fixed_temporal_masks,
        "node_kwargs": [{} for _ in range(node_count)],
    }


def freeze_graph(graph: Graph) -> None:
    graph.optimized_spatial = False
    graph.optimized_spatial_message_type = False
    graph.optimized_temporal = False
    graph.optimized_temporal_message_type = False
    graph.spatial_logits.requires_grad_(False)
    graph.spatial_message_type_logits.requires_grad_(False)
    graph.temporal_logits.requires_grad_(False)
    graph.temporal_message_type_logits.requires_grad_(False)


async def evaluate_hotpotqa(
    graph: Graph,
    dataset: HotpotQADataset,
    num_rounds: int,
    limit_questions: Optional[int],
    eval_batch_size: int,
    prediction_file: Optional[Path],
    info_setting: str,
) -> Dict[str, float]:
    print(
        f"Evaluating AgentPrune on {dataset.__class__.__name__} split {dataset.split} "
        f"with information setting {info_setting}"
    )
    freeze_graph(graph)

    metrics = HotpotQAAnswerMetrics()
    prediction_answers: Dict[str, str] = {}
    agent_count = len(graph.nodes)

    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records: List[Any] = []
        for i_record, record in enumerate(dataset):
            if limit_questions is not None and i_record >= limit_questions:
                break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if len(records) > 0:
            yield records

    data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)
    num_batches = int(math.ceil(data_len / eval_batch_size)) if data_len else 0

    for i_batch, record_batch in tqdm(enumerate(eval_loader(eval_batch_size)), total=num_batches):
        print(80 * "-")
        start_ts = time.time()
        answer_tasks = []

        for record_offset, record in enumerate(record_batch):
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.spatial_message_type_logits = graph.spatial_message_type_logits
            realized_graph.temporal_logits = graph.temporal_logits
            realized_graph.temporal_message_type_logits = graph.temporal_message_type_logits
            realized_graph.share_cost_state(graph)
            realized_graph.current_sample_index = i_batch * eval_batch_size + record_offset + 1
            input_dict = dataset.record_to_input(
                record,
                agent_count=agent_count,
                info_setting=info_setting,
            )
            answer_tasks.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds)))

        raw_results = await asyncio.gather(*answer_tasks)
        raw_answers, _ = zip(*raw_results)
        print(f"Batch time {time.time() - start_ts:.3f}")

        for raw_answer, record in zip(raw_answers, record_batch):
            answer = dataset.postprocess_answer(raw_answer, record=record)
            prediction_answers[dataset.record_to_id(record)] = answer
            correct_answer = dataset.record_to_target_answer(record)

            print("Raw answer:", raw_answer)
            print("Postprocessed answer:", answer)
            if correct_answer is not None:
                print("Correct answer:", correct_answer)
                metrics.update(answer, correct_answer)
                metrics.print()
            else:
                print("Correct answer: <unlabeled split>")

    metrics.print()
    if prediction_file is not None:
        prediction_file.parent.mkdir(parents=True, exist_ok=True)
        with open(prediction_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "answer": prediction_answers,
                    "sp": {record_id: [] for record_id in prediction_answers},
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Wrote HotpotQA-format predictions to {prediction_file}")

    return metrics.get()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentPrune experiments on HotpotQA")
    parser.add_argument("--dataset_dir", type=str, default="dataset/hotpotqa",
                        help="Directory containing official HotpotQA json/jsonl/parquet files.")
    parser.add_argument("--train_split", type=str, default="train",
                        choices=["train", "dev", "validation", "val", "dev_distractor", "dev_fullwiki"],
                        help="HotpotQA split used for AgentPrune training/pruning.")
    parser.add_argument("--eval_split", type=str, default="dev_distractor",
                        choices=["train", "dev", "validation", "val", "dev_distractor", "dev_fullwiki", "test", "test_fullwiki"],
                        help="HotpotQA split to evaluate. Default uses the labeled distractor dev set.")
    parser.add_argument("--prediction_file", type=str, default=None,
                        help="Optional path for HotpotQA-format predictions with answer and sp fields.")
    parser.add_argument("--info_setting", type=str, default=FULL_BROADCAST,
                        choices=[CENTRALIZED_INPUT, FULL_BROADCAST],
                        help=(
                            "HotpotQA document allocation mode. "
                            "centralized_input: Agent 1 gets all 10 docs and other agents get no docs. "
                            "full_broadcast: every agent gets all 10 docs."
                        ))
    parser.add_argument("--max_context_paragraphs", type=int, default=None,
                        help="Optional cap on context paragraphs per question.")
    parser.add_argument("--max_sentences_per_paragraph", type=int, default=None,
                        help="Optional cap on sentences per context paragraph.")
    parser.add_argument("--mode", type=str, default="FullConnected",
                        choices=["DirectAnswer", "FullConnected", "Random", "Chain", "Debate", "Layered", "Star", "Mesh"],
                        help="Communication topology.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for AgentPrune edge optimization.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--agent_names", nargs="+", type=str, default=["AnalyzeAgent"])
    parser.add_argument("--agent_nums", nargs="+", type=int, default=[5])
    parser.add_argument("--num_iterations", type=int, default=10,
                        help="Number of AgentPrune optimization iterations.")
    parser.add_argument("--imp_per_iterations", type=int, default=5,
                        help="Prune every N optimization iterations.")
    parser.add_argument("--pruning_rate", type=float, default=0.25,
                        help="Fraction of currently active edges pruned at each pruning step.")
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--llm_name", type=str, default="gpt-4-1106-preview")
    parser.add_argument("--domain", type=str, default="hotpotqa")
    parser.add_argument("--decision_method", type=str, default="FinalRefer")
    parser.add_argument("--log_agent_outputs", action="store_true")
    parser.add_argument("--limit_questions", type=int, default=200,
                        help="Maximum number of evaluation questions to run. Default 200.")
    parser.add_argument("--optimized_spatial", action="store_true",
                        help="Enable AgentPrune optimization and pruning of spatial edges.")
    parser.add_argument("--optimized_temporal", action="store_true",
                        help="Enable AgentPrune optimization and pruning of temporal edges.")
    parser.add_argument("--learn_message_type", action="store_true",
                        help="Learn answer/reason/full message type policy on retained edges.")
    parser.add_argument("--freeze_after_training", action="store_true", default=True,
                        help="Freeze the learned graph before evaluation.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    return args


async def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    agent_names = [
        name
        for name, count in zip(args.agent_names, args.agent_nums)
        for _ in range(count)
    ]
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
    graph.set_message_type_learning_mode(args.learn_message_type)

    dataset_train = HotpotQADataset(
        args.train_split,
        dataset_dir=args.dataset_dir,
        max_context_paragraphs=args.max_context_paragraphs,
        max_sentences_per_paragraph=args.max_sentences_per_paragraph,
    )
    dataset_eval = HotpotQADataset(
        args.eval_split,
        dataset_dir=args.dataset_dir,
        max_context_paragraphs=args.max_context_paragraphs,
        max_sentences_per_paragraph=args.max_sentences_per_paragraph,
    )

    train_prompt_used = 0.0
    train_completion_used = 0.0
    train_agent_metrics: List[Dict[str, Any]] = []

    if args.optimized_spatial or args.optimized_temporal:
        graph.set_eval_mode(False)
        train_prompt_start, train_completion_start = get_token_snapshot()
        train_agent_start = get_agent_token_snapshot(graph)
        await train_hotpotqa(
            graph=graph,
            dataset=dataset_train,
            num_iters=args.num_iterations,
            num_rounds=args.num_rounds,
            lr=args.lr,
            batch_size=args.batch_size,
            imp_per_iters=args.imp_per_iterations,
            pruning_rate=args.pruning_rate,
            info_setting=args.info_setting,
        )
        train_prompt_used = PromptTokens.instance().value - train_prompt_start
        train_completion_used = CompletionTokens.instance().value - train_completion_start
        train_agent_metrics = get_phase_agent_token_metrics(graph, train_agent_start)
        print_phase_token_summary("Training", train_prompt_used, train_completion_used)
        print_phase_agent_token_summary("Training", train_agent_metrics)
    else:
        print("Training skipped: both optimized_spatial and optimized_temporal are disabled.")
        print_phase_token_summary("Training", train_prompt_used, train_completion_used)
        print_phase_agent_token_summary("Training", train_agent_metrics)

    print_retained_edge_message_summary(graph)

    if args.freeze_after_training:
        freeze_graph(graph)
        print("Evaluation graph frozen: kept edges use the learned masks/logits without resampling.")

    graph.set_eval_mode(True)
    eval_agent_start = get_agent_token_snapshot(graph)
    eval_prompt_start, eval_completion_start = get_token_snapshot()
    prediction_file = Path(args.prediction_file) if args.prediction_file else None
    scores = await evaluate_hotpotqa(
        graph=graph,
        dataset=dataset_eval,
        num_rounds=args.num_rounds,
        limit_questions=args.limit_questions,
        eval_batch_size=args.batch_size,
        prediction_file=prediction_file,
        info_setting=args.info_setting,
    )
    eval_prompt_used = PromptTokens.instance().value - eval_prompt_start
    eval_completion_used = CompletionTokens.instance().value - eval_completion_start

    eval_agent_metrics = get_phase_agent_token_metrics(graph, eval_agent_start)
    print_phase_token_summary("Inference", eval_prompt_used, eval_completion_used)
    print_phase_agent_token_summary("Inference", eval_agent_metrics)
    print(f"Answer EM: {scores['em']}")
    print(f"Answer F1: {scores['f1']}")
    print(f"Answer Precision: {scores['prec']}")
    print(f"Answer Recall: {scores['recall']}")
    print(f"Prompt Tokens: {PromptTokens.instance().value}")
    print(f"Completion Tokens: {CompletionTokens.instance().value}")
    print(f"Total Tokens: {PromptTokens.instance().value + CompletionTokens.instance().value}")
    print_agent_token_summary(graph)


if __name__ == "__main__":
    asyncio.run(main())
