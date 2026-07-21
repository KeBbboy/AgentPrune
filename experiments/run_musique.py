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

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import AgentDropout.agents  # noqa: F401
import AgentDropout.prompt  # noqa: F401
from AgentDropout.graph.graph import Graph
from AgentDropout.llm.gpt_chat import configure_thinking, parse_bool_arg
from AgentDropout.utils.const import AgentPrune_ROOT
from AgentDropout.utils.globals import CompletionTokens, Cost, PromptTokens, Time, print_token_stats
from AgentDropout.benchmarks.agent_benchmarks import AgentBenchmarkDataset, BenchmarkRecord, iter_batches
from experiments.run_aime import (
    add_taylor_prune_argument,
    configure_taylor_prune,
    evaluate as evaluate_with_usage,
    train_agentprune,
)


MODES = ["DirectAnswer", "FullConnected", "Random", "Chain", "Debate", "Layered", "Star", "Mesh"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run MUSIQUE with BenchmarkAgent.")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--train_split", type=str, default=None,
                        help="Split used only for AgentPrune training.")
    parser.add_argument("--train_data_dir", type=str, default=None,
                        help="Data directory used only for AgentPrune training.")
    parser.add_argument("--train_dataset_path", type=str, default=None,
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
    add_taylor_prune_argument(parser, "musique")
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--domain", type=str, default="musique")
    parser.add_argument("--decision_method", type=str, default="FinalRefer")
    parser.add_argument("--optimized_spatial", action="store_true")
    parser.add_argument("--optimized_temporal", action="store_true")
    parser.add_argument("--diff", action="store_true")
    parser.add_argument("--dec", action="store_true")
    parser.add_argument(
        "--enabled_tools",
        nargs="*",
        default=None,
        help="Defaults to MUSIQUE tools. Use --enabled_tools with no values for no tools.",
    )
    parser.add_argument("--direct_tool_call", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--web_tool_provider", type=str, default="tavily")
    parser.add_argument("--top_k_results", type=int, default=3)
    parser.add_argument("--max_search_limit", type=int, default=3)
    parser.add_argument("--max_tool_chars", type=int, default=6000)
    parser.add_argument("--max_tool_turns", type=int, default=3)
    parser.add_argument("--tool_aware_loss", action="store_true",
                        help="Enable tool-aware reward penalties during AgentPrune training.")
    parser.add_argument("--tool_loss_token_normalizer", type=float, default=10000.0,
                        help="Token normalizer used by tool-aware output/tool-token penalties.")
    parser.add_argument("--tool_loss_call_normalizer", type=float, default=10.0,
                        help="Call-count normalizer used by tool-aware call penalties.")
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
    if args.tool_loss_token_normalizer <= 0:
        parser.error("--tool_loss_token_normalizer must be > 0.")
    if args.tool_loss_call_normalizer <= 0:
        parser.error("--tool_loss_call_normalizer must be > 0.")
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
    return [{"tool_config": dict(base_tool_config)} for _ in agent_names]


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


async def evaluate(graph: Graph, dataset: AgentBenchmarkDataset, args) -> Dict[str, Any]:
    return await evaluate_with_usage(graph, dataset, args)


async def main():
    args = parse_args()
    configure_thinking(
        enable_thinking=args.llm_enable_thinking,
        thinking_budget=args.llm_thinking_budget,
    )
    dataset = AgentBenchmarkDataset(
        name="musique",
        split=args.split,
        data_dir=args.data_dir,
        dataset_path=args.dataset_path,
    )
    graph, kwargs = build_graph_with_kwargs(args, dataset)
    configure_taylor_prune(graph, args, "musique")
    if not args.dry_run and (args.optimized_spatial or args.optimized_temporal):
        if args.train_dataset_path and not Path(args.train_dataset_path).exists():
            raise FileNotFoundError(
                f"MUSIQUE AgentPrune training dataset not found: {args.train_dataset_path}. "
                "Create this jsonl file or pass --train_dataset_path to another file."
            )
        train_dataset = AgentBenchmarkDataset(
            name="musique",
            split=args.train_split or args.split,
            data_dir=args.train_data_dir or args.data_dir,
            dataset_path=args.train_dataset_path,
        )
        train_source = args.train_dataset_path or f"{train_dataset.data_dir}/{train_dataset.spec.data_subdir}/{train_dataset.split}.jsonl"
        print(f"MUSIQUE AgentPrune training dataset: {train_source}")
        await train_agentprune(graph, train_dataset, args, kwargs)
        print("Final spatial logits:", graph.spatial_logits)
        print("Final temporal logits:", graph.temporal_logits)
        print("Final spatial masks:", graph.spatial_masks)
        print("Final temporal masks:", graph.temporal_masks)
    summary = await evaluate(graph, dataset, args)
    print(f"MUSIQUE Score: {summary}")


if __name__ == "__main__":
    asyncio.run(main())
