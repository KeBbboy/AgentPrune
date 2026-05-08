import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
from typing import Union, Literal, List, Dict, Any
import argparse
import random
import copy
import numpy as np
import torch

from AgentPrune.graph.graph import Graph
from dataset.aqua_dataset import AQuADataset
from experiments.train_mmlu import train
from experiments.evaluate_mmlu import evaluate
from experiments.run_mmlu import (
    print_agent_token_summary,
    get_agent_token_snapshot,
    get_phase_agent_token_metrics,
    get_token_snapshot,
    print_phase_token_summary,
    print_phase_agent_token_summary,
    freeze_graph_for_evaluation,
    restore_all_edges_for_evaluation,
    get_kwargs,
)
from AgentPrune.utils.globals import PromptTokens, CompletionTokens


def parse_args():
    parser = argparse.ArgumentParser(description="AgentPrune experiments on AQuA")

    parser.add_argument('--dataset_dir', type=str, default='dataset/aqua',
                        help="Directory containing AQuA split files such as dev/test json or jsonl.")
    parser.add_argument('--train_split', type=str, default='dev',
                        choices=['train', 'dev', 'val', 'test'],
                        help="Dataset split used for training/pruning.")
    parser.add_argument('--eval_split', type=str, default='test',
                        choices=['train', 'dev', 'val', 'test'],
                        help="Dataset split used for evaluation.")
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered', 'Star', 'Mesh',
                                 'FakeFullConnected', 'FakeRandom', 'FakeChain', 'FakeStar', 'FakeMesh', 'FakeAGRandom', 'FakeAGFull'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['AnalyzeAgent'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[5],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help="Number of optimization iterations.")
    parser.add_argument('--imp_per_iterations', type=int, default=5,
                        help="Prune every few iterations.")
    parser.add_argument('--num_rounds', type=int, default=2,
                        help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25,
                        help="The Rate of Pruning.")
    parser.add_argument('--llm_name', type=str, default="gpt-4-1106-preview",
                        help="Model name")
    parser.add_argument('--domain', type=str, default="aqua",
                        help="Domain name, default 'aqua'")
    parser.add_argument('--decision_method', type=str, default="FinalMajorVote",
                        help="the decision method of the final node")
    parser.add_argument('--log_agent_outputs', action='store_true',
                        help="Print each agent's raw output and extracted answer to stdout/logs.")
    parser.add_argument('--limit_questions', type=int, default=None,
                        help="Maximum number of evaluation questions to run.")
    parser.add_argument('--temperature', type=float, default=0.0,
                        help="LLM generation temperature.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for Python, NumPy, and PyTorch.")
    parser.add_argument('--optimized_spatial', action='store_true')
    parser.add_argument('--optimized_temporal', action='store_true')
    parser.add_argument('--freeze_after_training', action='store_true',
                        help="After training/pruning, freeze the learned graph for evaluation.")
    parser.add_argument('--restore_all_edges_for_eval', action='store_true',
                        help="Before evaluation, restore all masks to 1.")
    parser.add_argument('--share_special_roles_conclusion_only', action='store_true',
                        help="Only share final conclusion for selected roles in analyze prompts.")
    parser.add_argument('--share_special_roles_analysis_only', action='store_true',
                        help="Only share analysis-only content for selected roles in analyze prompts.")
    parser.add_argument(
        '--active_edge_message_mode',
        type=str,
        default='full_output',
        choices=[
            'full_output',
            'all_analysis_only',
            'all_conclusion_only',
            'all_answer_only',
            'all_answer_claim',
            'all_answer_steps',
            'all_answer_mixed_steps',
            'all_answer_top_score_steps',
            'spatial_answer_only_temporal_top_score_steps',
            'all_truncate',
            'spatial_analysis_only',
            'spatial_conclusion_only',
            'temporal_analysis_only',
            'temporal_conclusion_only',
        ],
        help="High-level active-edge message-sharing mode.",
    )
    parser.add_argument(
        '--truncate_content_ratio',
        type=float,
        default=0.75,
        choices=[0.75, 0.5, 0.25],
        help="Retained ratio of content after the answer letter in truncation mode.",
    )
    parser.add_argument(
        '--answer_steps_mixed_ratio',
        type=float,
        default=0.3,
        choices=[0.3, 0.7, 0.9],
        help="When active_edge_message_mode=all_answer_mixed_steps, ratio of active edges that share answer+steps.",
    )
    parser.add_argument(
        '--answer_steps_top_score_ratio',
        type=float,
        default=0.7,
        help="Top ratio of active edges ranked by score that share answer+steps.",
    )

    args = parser.parse_args()
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    if not 0.0 <= args.answer_steps_top_score_ratio <= 1.0:
        parser.error("--answer_steps_top_score_ratio must be between 0.0 and 1.0.")

    args.share_active_spatial_conclusion_only = False
    args.share_active_temporal_conclusion_only = False
    args.share_active_spatial_analysis_only = False
    args.share_active_temporal_analysis_only = False
    args.share_active_spatial_answer_only = False
    args.share_active_temporal_answer_only = False
    args.share_active_spatial_answer_claim = False
    args.share_active_temporal_answer_claim = False
    args.share_active_spatial_answer_steps = False
    args.share_active_temporal_answer_steps = False
    args.share_active_spatial_answer_steps_mixed = False
    args.share_active_temporal_answer_steps_mixed = False
    args.share_active_spatial_answer_steps_top_score = False
    args.share_active_temporal_answer_steps_top_score = False
    args.share_active_spatial_truncate = False
    args.share_active_temporal_truncate = False

    if args.active_edge_message_mode == 'all_analysis_only':
        args.share_active_spatial_analysis_only = True
        args.share_active_temporal_analysis_only = True
    elif args.active_edge_message_mode == 'all_conclusion_only':
        args.share_active_spatial_conclusion_only = True
        args.share_active_temporal_conclusion_only = True
    elif args.active_edge_message_mode == 'all_answer_only':
        args.share_active_spatial_answer_only = True
        args.share_active_temporal_answer_only = True
    elif args.active_edge_message_mode == 'all_answer_claim':
        args.share_active_spatial_answer_claim = True
        args.share_active_temporal_answer_claim = True
    elif args.active_edge_message_mode == 'all_answer_steps':
        args.share_active_spatial_answer_steps = True
        args.share_active_temporal_answer_steps = True
    elif args.active_edge_message_mode == 'all_answer_mixed_steps':
        args.share_active_spatial_answer_steps_mixed = True
        args.share_active_temporal_answer_steps_mixed = True
    elif args.active_edge_message_mode == 'all_answer_top_score_steps':
        args.share_active_spatial_answer_steps_top_score = True
        args.share_active_temporal_answer_steps_top_score = True
    elif args.active_edge_message_mode == 'spatial_answer_only_temporal_top_score_steps':
        args.share_active_spatial_answer_only = True
        args.share_active_temporal_answer_steps_top_score = True
    elif args.active_edge_message_mode == 'all_truncate':
        args.share_active_spatial_truncate = True
        args.share_active_temporal_truncate = True
    elif args.active_edge_message_mode == 'spatial_analysis_only':
        args.share_active_spatial_analysis_only = True
    elif args.active_edge_message_mode == 'spatial_conclusion_only':
        args.share_active_spatial_conclusion_only = True
    elif args.active_edge_message_mode == 'temporal_analysis_only':
        args.share_active_temporal_analysis_only = True
    elif args.active_edge_message_mode == 'temporal_conclusion_only':
        args.share_active_temporal_conclusion_only = True

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
        share_special_roles_conclusion_only=args.share_special_roles_conclusion_only,
        share_special_roles_analysis_only=args.share_special_roles_analysis_only,
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
        share_active_spatial_conclusion_only=args.share_active_spatial_conclusion_only,
        share_active_spatial_analysis_only=args.share_active_spatial_analysis_only,
        share_active_temporal_conclusion_only=args.share_active_temporal_conclusion_only,
        share_active_temporal_analysis_only=args.share_active_temporal_analysis_only,
        share_active_spatial_answer_only=args.share_active_spatial_answer_only,
        share_active_temporal_answer_only=args.share_active_temporal_answer_only,
        share_active_spatial_answer_claim=args.share_active_spatial_answer_claim,
        share_active_temporal_answer_claim=args.share_active_temporal_answer_claim,
        share_active_spatial_answer_steps=args.share_active_spatial_answer_steps,
        share_active_temporal_answer_steps=args.share_active_temporal_answer_steps,
        share_active_spatial_answer_steps_mixed=args.share_active_spatial_answer_steps_mixed,
        share_active_temporal_answer_steps_mixed=args.share_active_temporal_answer_steps_mixed,
        share_active_spatial_answer_steps_top_score=args.share_active_spatial_answer_steps_top_score,
        share_active_temporal_answer_steps_top_score=args.share_active_temporal_answer_steps_top_score,
        share_active_spatial_truncate=args.share_active_spatial_truncate,
        share_active_temporal_truncate=args.share_active_temporal_truncate,
        answer_steps_mixed_ratio=args.answer_steps_mixed_ratio,
        answer_steps_top_score_ratio=args.answer_steps_top_score_ratio,
        truncate_content_ratio=args.truncate_content_ratio,
        **kwargs,
    )

    dataset_train = AQuADataset(args.train_split, dataset_dir=args.dataset_dir)
    dataset_eval = AQuADataset(args.eval_split, dataset_dir=args.dataset_dir)

    train_prompt_used = 0.0
    train_completion_used = 0.0
    train_agent_metrics: List[Dict[str, Any]] = []

    if args.optimized_spatial or args.optimized_temporal:
        graph.set_eval_mode(False)
        train_prompt_start, train_completion_start = get_token_snapshot()
        train_agent_start = get_agent_token_snapshot(graph)
        await train(
            graph=graph,
            dataset=dataset_train,
            num_iters=args.num_iterations,
            num_rounds=args.num_rounds,
            lr=args.lr,
            batch_size=args.batch_size,
            imp_per_iters=args.imp_per_iterations,
            pruning_rate=args.pruning_rate,
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

    if args.restore_all_edges_for_eval:
        restore_all_edges_for_evaluation(graph)
        print("Evaluation graph restored: all spatial and temporal masks set to 1 before evaluation.")

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
