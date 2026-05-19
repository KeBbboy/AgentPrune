import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import asyncio
import random
from typing import List

import numpy as np
import torch

from AgentDropout.graph.graph import Graph
from AgentDropout.utils.const import AgentPrune_ROOT
from AgentDropout.utils.globals import CompletionTokens, PromptTokens
from dataset.mmlu_pro_dataset import MMLUProDataset
from experiments.evaluate_mmlu import evaluate
from experiments.train_mmlu import train
from AgentDropout_exp.run_mmlu import get_kwargs


def parse_args():
    parser = argparse.ArgumentParser(description="AgentDropout experiments on MMLU-Pro")

    parser.add_argument('--dataset_dir', type=str, default='dataset/mmlu_pro',
                        help="Directory containing MMLU-Pro split files in json/jsonl/parquet format.")
    parser.add_argument('--train_split', type=str, default='dev',
                        help="Dataset split used for training/pruning. Default 'dev'.")
    parser.add_argument('--eval_split', type=str, default='val',
                        help="Dataset split used for evaluation. Default 'val'.")
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered', 'Star', 'Mesh',
                                 'FakeFullConnected', 'FakeRandom', 'FakeChain', 'FakeStar', 'FakeMesh',
                                 'FakeAGRandom', 'FakeAGFull'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate")
    parser.add_argument('--delta', type=float, default=0.1,
                        help="noise level")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="batch size")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['AnalyzeAgent'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[5],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help="Number of optimization iterations. Default 10.")
    parser.add_argument('--imp_per_iterations', type=int, default=5,
                        help="Prune every few iterations. Default 5.")
    parser.add_argument('--num_rounds', type=int, default=2,
                        help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25,
                        help="The Rate of Pruning. Default 0.25.")
    parser.add_argument('--llm_name', type=str, default="gpt-3.5-turbo",
                        help="Model name")
    parser.add_argument('--domain', type=str, default="mmlu_pro",
                        help="Domain (the same as dataset name), default 'mmlu_pro'")
    parser.add_argument('--decision_method', type=str, default="FinalRefer",
                        help="the decision method of the final node")
    parser.add_argument('--limit_questions', type=int, default=153,
                        help="Maximum number of evaluation questions to run.")
    parser.add_argument('--optimized_spatial', action='store_true')
    parser.add_argument('--optimized_temporal', action='store_true')
    parser.add_argument('--learn_message_type', action='store_true',
                        help="Enable answer/reason message-type learning in AgentDropout second-stage training.")
    parser.add_argument('--freeze_after_training', action='store_true', default=True,
                        help="After training/pruning, freeze the learned graph for evaluation.")
    parser.add_argument('--diff', action='store_true')
    parser.add_argument('--dec', action='store_true')
    parser.add_argument('--cot', action='store_true')
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for Python, NumPy, and PyTorch. Default 42.")
    args = parser.parse_args()
    result_path = AgentPrune_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    if args.learn_message_type and not (args.optimized_spatial or args.optimized_temporal):
        parser.error("--learn_message_type requires --optimized_spatial and/or --optimized_temporal.")

    return args


def print_final_graph_state(graph: Graph):
    print("Final spatial logits: ", graph.spatial_logits)
    print("Final temporal logits: ", graph.temporal_logits)
    print("Final spatial masks: ", graph.spatial_masks)
    print("Final temporal masks: ", graph.temporal_masks)
    if not graph.diff:
        print("Final spatial sparsity:", graph.spatial_masks.sum() / graph.spatial_masks.numel())
        print("Final temporal sparsity:", graph.temporal_masks.sum() / graph.temporal_masks.numel())
    else:
        spatial_sparsity = torch.mean(torch.stack([mask.sum() / mask.numel() for mask in graph.spatial_masks]))
        print("Spatial sparsity (mean):", spatial_sparsity)
        if len(graph.temporal_masks) > 0:
            temporal_sparsity = torch.mean(torch.stack([mask.sum() / mask.numel() for mask in graph.temporal_masks]))
            print("Temporal sparsity (mean):", temporal_sparsity)


def freeze_graph_for_evaluation(graph: Graph, learn_message_type: bool):
    graph.optimized_spatial = False
    graph.optimized_temporal = False
    graph.set_message_type_learning_mode(learn_message_type)


async def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    dataset_train = MMLUProDataset(args.train_split, dataset_dir=args.dataset_dir)
    dataset_eval = MMLUProDataset(args.eval_split, dataset_dir=args.dataset_dir)

    if args.optimized_spatial or args.optimized_temporal:
        await train(
            graph=graph,
            dataset=dataset_train,
            num_iters=args.num_iterations,
            num_rounds=args.num_rounds,
            lr=args.lr,
            batch_size=args.batch_size,
            imp_per_iters=args.imp_per_iterations,
            pruning_rate=args.pruning_rate,
            args=args,
            kwargs=kwargs,
        )
    else:
        print("Training skipped: both optimized_spatial and optimized_temporal are disabled.")

    print_final_graph_state(graph)

    if args.freeze_after_training:
        freeze_graph_for_evaluation(graph, args.learn_message_type)
        print("Evaluation graph frozen: kept edges use the learned masks/logits without resampling.")

    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()
    score = await evaluate(
        graph=graph,
        dataset=dataset_eval,
        num_rounds=args.num_rounds,
        limit_questions=args.limit_questions,
        eval_batch_size=args.batch_size,
        dec=args.dec,
        args=args,
    )
    print(f"Score: {score}")
    print(f"Prompt Tokens: {PromptTokens.instance().value}")
    print(f"Completion Tokens: {CompletionTokens.instance().value}")
    print(f"Total Tokens: {PromptTokens.instance().value + CompletionTokens.instance().value}")


if __name__ == '__main__':
    asyncio.run(main())
