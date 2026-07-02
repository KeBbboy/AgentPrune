import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import asyncio
import torch

from AgentDropout.graph.graph import Graph
from AgentDropout.llm.gpt_chat import configure_thinking, parse_bool_arg
from AgentDropout.utils.const import AgentPrune_ROOT
from AgentDropout.utils.globals import Cost, CompletionTokens, PromptTokens, CachedTokens, ReasoningTokens, BilledPromptTokens, BilledCompletionTokens, print_token_stats
from dataset.mmlu_pro_dataset import MMLUProDataset
from experiments.evaluate_mmlu import evaluate
from experiments.run_mmlu import get_kwargs
from experiments.train_mmlu import train


def parse_args():
    parser = argparse.ArgumentParser(description="AgentPrune Experiments on MMLU-Pro.")
    parser.add_argument(
        '--mode',
        type=str,
        default='FullConnected',
        choices=[
            'DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered', 'Star', 'Mesh',
            'FakeFullConnected', 'FakeRandom', 'FakeChain', 'FakeStar', 'FakeMesh', 'FakeAGRandom', 'FakeAGFull',
        ],
    )
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--delta', type=float, default=0.1, help="noise level")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['AnalyzeAgent'])
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[5])
    parser.add_argument('--num_iterations', type=int, default=10)
    parser.add_argument('--imp_per_iterations', type=int, default=5)
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--pruning_rate', type=float, default=0.25)
    parser.add_argument('--llm_name', type=str, default="MiniMax-M3")
    parser.add_argument('--domain', type=str, default="mmlu_pro")
    parser.add_argument('--decision_method', type=str, default="FinalRefer")
    parser.add_argument('--optimized_spatial', action='store_true')
    parser.add_argument('--optimized_temporal', action='store_true')
    parser.add_argument('--diff', action='store_true')
    parser.add_argument('--dec', action='store_true')
    parser.add_argument('--cot', action='store_true')
    parser.add_argument('--llm_enable_thinking', type=parse_bool_arg, default=True,
                        help='Enable model thinking mode: true or false')
    parser.add_argument('--llm_thinking_budget', type=int, default=8192,
                        help='Thinking token budget (unified interface): '
                             'Qwen3-style providers use this as token budget (128-32768); '
                             'MiniMax-M3 uses >0 for adaptive, 0 for disabled')
    parser.add_argument('--sample_ratio', type=float, default=0.1, help="Ratio of dataset to sample (0.1 = 10%)")
    parser.add_argument('--sample_seed', type=int, default=42, help="Random seed for stratified sampling")
    args = parser.parse_args()

    (AgentPrune_ROOT / "result").mkdir(exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    return args


async def main():
    args = parse_args()
    configure_thinking(
        enable_thinking=args.llm_enable_thinking,
        thinking_budget=args.llm_thinking_budget,
    )
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

    dataset_train = MMLUProDataset('dev')
    dataset_val = MMLUProDataset('val')

    dataset_val.stratified_sample(ratio=args.sample_ratio, seed=args.sample_seed)

    if args.optimized_spatial or args.optimized_temporal:
        await train(
            graph=graph,
            dataset=dataset_train,
            num_iters=args.num_iterations,
            num_rounds=args.num_rounds,
            lr=args.lr,
            batch_size=20,
            imp_per_iters=args.imp_per_iterations,
            pruning_rate=args.pruning_rate,
            args=args,
            kwargs=kwargs,
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
    score = await evaluate(
        graph=graph,
        dataset=dataset_val,
        num_rounds=args.num_rounds,
        limit_questions=None,
        eval_batch_size=args.batch_size,
        dec=args.dec,
        args=args,
    )
    print(f"Score: {score}")
    print_token_stats("Final")


if __name__ == "__main__":
    asyncio.run(main())
