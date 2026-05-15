import sys
import os
import argparse
import yaml
import json
import time
import asyncio
from pathlib import Path
import torch
import copy
from typing import List, Union, Literal, Dict, Any
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from AgentPrune.utils.const import AgentPrune_ROOT
from AgentPrune.graph.graph import Graph
from AgentPrune.utils.globals import Time
from AgentPrune.utils.globals import PromptTokens, CompletionTokens
from dataset.multiarith_dataset import MultiArithDataset
from experiments.evaluate_mmlu import evaluate
from experiments.run_humaneval import print_retained_edge_message_summary
from experiments.run_mmlu import freeze_graph_for_evaluation
from experiments.train_mmlu import train


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


def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w', encoding='utf-8') as file:
            json.dump([], file)

    with open(result_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch * batch_size:i_batch * batch_size + batch_size]


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="AgentPrune Experiments on MultiArith")
    parser.add_argument("--dataset_dir", type=str, default="dataset/multiarith")
    parser.add_argument("--dataset_json", type=str, default="dataset/multiarith/test.json")
    parser.add_argument("--train_split", type=str, default="train",
                        choices=["train", "dev", "val", "validation", "test"],
                        help="Dataset split used for training/pruning. Default 'train'.")
    parser.add_argument("--eval_split", type=str, default="test",
                        choices=["train", "dev", "val", "validation", "test"],
                        help="Dataset split used for evaluation. Default 'test'.")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="gpt-4-1106-preview")
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered', 'Star'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size")
    parser.add_argument('--imp_per_iterations', type=int, default=5, help="Prune every few iterations. Default 5.")
    parser.add_argument('--num_rounds', type=int, default=1, help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25, help="The Rate of Pruning. Default 0.05.")
    parser.add_argument('--num_iterations', type=int, default=10, help="The num of training iterations.")
    parser.add_argument('--domain', type=str, default="multiarith", help="Domain (the same as dataset name), default 'multiarith'")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[4],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--decision_method', type=str, default='FinalRefer',
                        help='The decison method of the agentprune')
    parser.add_argument('--log_agent_outputs', action='store_true',
                        help="Print each agent's raw output and extracted answer to stdout/logs.")
    parser.add_argument('--limit_questions', type=int, default=None,
                        help="Maximum number of evaluation questions to run.")
    parser.add_argument('--temperature', type=float, default=0.0,
                        help="LLM generation temperature. Default 0.0 for near-deterministic output.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for Python and PyTorch. Default 42.")
    parser.add_argument('--freeze_after_training', action='store_true', default=True,
                        help="After training/pruning, freeze the learned graph for evaluation so kept edges are no longer resampled.")
    parser.add_argument('--optimized_spatial', action='store_true')
    parser.add_argument('--optimized_temporal', action='store_true')
    parser.add_argument('--learn_message_type', action='store_true',
                        help="Enable answer/reason message-type learning on retained edges. Without this flag, retained edges transmit full messages as in the original AgentPrune.")
    args = parser.parse_args()
    result_path = AgentPrune_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    return args


async def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{AgentPrune_ROOT}/result/multiarith")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{args.domain}_{args.llm_name}_{current_time}.json"

    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    decision_method = args.decision_method
    kwargs = get_kwargs(
        args.mode,
        len(agent_names),
    )
    graph = Graph(domain=args.domain,
                  llm_name=args.llm_name,
                  agent_names=agent_names,
                  decision_method=decision_method,
                  log_node_outputs=args.log_agent_outputs,
                  llm_temperature=args.temperature,
                  optimized_spatial=args.optimized_spatial,
                  optimized_temporal=args.optimized_temporal,
                  **kwargs)
    graph.set_message_type_learning_mode(args.learn_message_type)

    dataset_train = MultiArithDataset(args.train_split, dataset_dir=args.dataset_dir, dataset_path=args.dataset_json)
    dataset_eval = MultiArithDataset(args.eval_split, dataset_dir=args.dataset_dir, dataset_path=args.dataset_json)

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

    print_retained_edge_message_summary(graph)

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


def get_kwargs(mode: Union[Literal['DirectAnswer'], Literal['FullConnected'], Literal['Random'], Literal['Chain'], Literal['Debate'], Literal['Layered'], Literal['Star']],
               N: int,
               use_structured_output: bool = False):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks: List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks: List[List[int]] = None
    node_kwargs = None

    def generate_layered_graph(n, layer_num=2):
        adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
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
        matrix = [[0] * n for _ in range(n)]
        for i in range(0, n):
            for j in range(i + 1, n):
                matrix[i][j] = 1
        return matrix

    if mode == 'DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role': 'Programming Expert'}]
    elif mode == 'FullConnected':
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == 'Random':
        fixed_spatial_masks = [[random.randint(0, 1) if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode == 'Chain':
        fixed_spatial_masks = [[1 if i == j + 1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i == 0 and j == N - 1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for _ in range(N)] for _ in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == 'Star':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]

    if node_kwargs is None:
        node_kwargs = [{} for _ in range(N)]
    node_kwargs = [{**kwargs, 'use_structured_output': use_structured_output} for kwargs in node_kwargs]

    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs": node_kwargs}


if __name__ == '__main__':
    asyncio.run(main())
