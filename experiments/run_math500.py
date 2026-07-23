import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import AgentDropout.agents  # noqa: F401
import AgentDropout.prompt  # noqa: F401
from AgentDropout.benchmarks.agent_benchmarks import AgentBenchmarkDataset
from AgentDropout.graph.message_compression import (
    add_message_compression_arguments,
    build_message_compression_config_from_args,
    set_message_compression_config,
)
from AgentDropout.llm.gpt_chat import configure_thinking, parse_bool_arg
from experiments.run_aime import (
    MODES,
    add_taylor_prune_argument,
    build_graph_with_kwargs,
    configure_taylor_prune,
    evaluate,
    train_agentprune,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run MATH500 with BenchmarkAgent.")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--train_split", type=str, default=None, help="Split used only for AgentPrune training.")
    parser.add_argument("--train_data_dir", type=str, default=None, help="Data directory used only for AgentPrune training.")
    parser.add_argument("--train_dataset_path", type=str, default=None, help="Dataset path used only for AgentPrune training.")
    parser.add_argument("--train_limit_questions", type=int, default=20, help="Number of training questions for AgentPrune.")
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
    parser.add_argument("--num_iterations", type=int, default=10, help="Number of AgentPrune optimization iterations.")
    parser.add_argument("--imp_per_iterations", type=int, default=5, help="Prune every N optimization iterations.")
    parser.add_argument("--pruning_rate", type=float, default=0.25, help="Rate of graph edge pruning.")
    add_taylor_prune_argument(parser, "math500")
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--domain", type=str, default="math500")
    parser.add_argument("--decision_method", type=str, default="FinalRefer")
    parser.add_argument("--optimized_spatial", action="store_true")
    parser.add_argument("--optimized_temporal", action="store_true")
    parser.add_argument("--diff", action="store_true")
    parser.add_argument("--dec", action="store_true")
    parser.add_argument(
        "--enabled_tools",
        nargs="*",
        default=None,
        help="Defaults to MATH500 tools ['web_search', 'code_generator']. Use with no values for no tools.",
    )
    parser.add_argument("--direct_tool_call", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return_code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--web_tool_provider", type=str, default="tavily")
    parser.add_argument("--top_k_results", type=int, default=3)
    parser.add_argument("--max_search_limit", type=int, default=3)
    parser.add_argument("--max_tool_chars", type=int, default=6000)
    parser.add_argument("--max_tool_turns", type=int, default=1)
    parser.add_argument("--tool_aware_loss", action="store_true")
    parser.add_argument("--tool_loss_token_normalizer", type=float, default=10000.0)
    parser.add_argument("--tool_loss_call_normalizer", type=float, default=10.0)
    parser.add_argument("--llm_enable_thinking", type=parse_bool_arg, default=True)
    parser.add_argument("--llm_thinking_budget", type=int, default=8192)
    add_message_compression_arguments(parser)
    args = parser.parse_args()
    set_message_compression_config(build_message_compression_config_from_args(args))
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


async def main():
    args = parse_args()
    configure_thinking(enable_thinking=args.llm_enable_thinking, thinking_budget=args.llm_thinking_budget)
    dataset = AgentBenchmarkDataset(
        name="math500",
        split=args.split,
        data_dir=args.data_dir,
        dataset_path=args.dataset_path,
    )
    graph, kwargs = build_graph_with_kwargs(args, dataset)
    configure_taylor_prune(graph, args, "math500")
    if not args.dry_run and (args.optimized_spatial or args.optimized_temporal):
        if args.train_dataset_path and not Path(args.train_dataset_path).exists():
            raise FileNotFoundError(f"MATH500 AgentPrune training dataset not found: {args.train_dataset_path}.")
        train_dataset = AgentBenchmarkDataset(
            name="math500",
            split=args.train_split or args.split,
            data_dir=args.train_data_dir or args.data_dir,
            dataset_path=args.train_dataset_path,
        )
        train_source = args.train_dataset_path or f"{train_dataset.data_dir}/{train_dataset.spec.data_subdir}/{train_dataset.split}.jsonl"
        print(f"MATH500 AgentPrune training dataset: {train_source}")
        await train_agentprune(graph, train_dataset, args, kwargs)
    summary = await evaluate(graph, dataset, args)
    print(f"MATH500 Score: {summary}")


if __name__ == "__main__":
    asyncio.run(main())
