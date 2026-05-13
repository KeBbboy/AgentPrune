import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
from typing import Any, Dict, List
import argparse
import random
import numpy as np
import torch

from AgentPrune.graph.graph import Graph
from dataset.mmlu_pro_dataset import MMLUProDataset
from experiments.evaluate_mmlu import evaluate
from experiments.train_mmlu_edge_policy import load_edge_policy_config, train_joint_policy
from experiments.run_mmlu import (
    print_agent_token_summary,
    get_agent_token_snapshot,
    get_phase_agent_token_metrics,
    get_token_snapshot,
    print_phase_token_summary,
    print_phase_agent_token_summary,
    freeze_graph_for_evaluation,
    get_kwargs,
)
from AgentPrune.utils.globals import PromptTokens, CompletionTokens


def parse_args():
    parser = argparse.ArgumentParser(description="AgentPrune experiments on MMLU-Pro")

    parser.add_argument('--dataset_dir', type=str, default='dataset/mmlu_pro',
                        help="Directory containing MMLU-Pro split files in json/jsonl/parquet format.")
    parser.add_argument('--train_split', type=str, default='dev',
                        help="Dataset split used for training/pruning. Default 'dev' for consistency with MMLU.")
    parser.add_argument('--eval_split', type=str, default='val',
                        help="Dataset split used for evaluation. Default 'val' for consistency with MMLU.")
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered', 'Star', 'Mesh',
                                 'FakeFullConnected', 'FakeRandom', 'FakeChain', 'FakeStar', 'FakeMesh', 'FakeAGRandom', 'FakeAGFull'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate")
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
    parser.add_argument('--llm_name', type=str, default="gpt-4-1106-preview",
                        help="Model name, None runs the default ChatGPT4")
    parser.add_argument('--domain', type=str, default="mmlu_pro",
                        help="Domain (the same as dataset name), default 'mmlu_pro'")
    parser.add_argument('--decision_method', type=str, default="FinalRefer",
                        help="the decision method of the final node")
    parser.add_argument('--log_agent_outputs', action='store_true',
                        help="Print each agent's raw output and extracted answer to stdout/logs.")
    parser.add_argument('--limit_questions', type=int, default=153,
                        help="Maximum number of evaluation questions to run.")
    parser.add_argument('--optimized_spatial', action='store_true')
    parser.add_argument('--optimized_temporal', action='store_true')
    parser.add_argument('--freeze_after_training', action='store_true', default=True,
                        help="After training/pruning, freeze the learned graph for evaluation so kept edges are no longer resampled.")
    parser.add_argument('--edge_policy_config', type=str, default='configs/mmlu_edge_policy.json',
                        help="JSON config for unified graph/message training.")
    parser.add_argument('--temperature', type=float, default=0.0,
                        help="LLM generation temperature. Default 0.0 for near-deterministic output.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for Python, NumPy, and PyTorch. Default 42.")
    args = parser.parse_args()
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")

    return args


async def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(
        args.mode,
        len(agent_names),
    )

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

    dataset_train = MMLUProDataset(args.train_split, dataset_dir=args.dataset_dir)
    dataset_eval = MMLUProDataset(args.eval_split, dataset_dir=args.dataset_dir)
    edge_policy_config = load_edge_policy_config(args.edge_policy_config)

    train_prompt_used = 0.0
    train_completion_used = 0.0
    train_agent_metrics: List[Dict[str, Any]] = []

    if args.optimized_spatial or args.optimized_temporal:
        train_prompt_start, train_completion_start = get_token_snapshot()
        train_agent_start = get_agent_token_snapshot(graph)
        await train_joint_policy(
            graph=graph,
            dataset=dataset_train,
            num_rounds=args.num_rounds,
            num_iters=args.num_iterations,
            batch_size=args.batch_size,
            graph_lr=args.lr,
            imp_per_iters=args.imp_per_iterations,
            pruning_rate=args.pruning_rate,
            config=edge_policy_config,
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

    if args.freeze_after_training:
        freeze_graph_for_evaluation(graph)
        print("Evaluation graph frozen: kept edges use the learned masks/logits without resampling.")

    graph.set_eval_mode(True)
    eval_agent_start = get_agent_token_snapshot(graph)
    eval_prompt_start, eval_completion_start = get_token_snapshot()
    score = await evaluate(
        graph=graph,
        dataset=dataset_eval,
        num_rounds=args.num_rounds,
        limit_questions=args.limit_questions,
        eval_batch_size=args.batch_size,
    )
    eval_prompt_used = PromptTokens.instance().value - eval_prompt_start
    eval_completion_used = CompletionTokens.instance().value - eval_completion_start

    eval_agent_metrics = get_phase_agent_token_metrics(graph, eval_agent_start)
    print_phase_token_summary("Inference", eval_prompt_used, eval_completion_used)
    print_phase_agent_token_summary("Inference", eval_agent_metrics)
    print(f"Score: {score}")
    print(f"Prompt Tokens: {PromptTokens.instance().value}")
    print(f"Completion Tokens: {CompletionTokens.instance().value}")
    print(f"Total Tokens: {PromptTokens.instance().value + CompletionTokens.instance().value}")
    print_agent_token_summary(graph)


if __name__ == '__main__':
    asyncio.run(main())
