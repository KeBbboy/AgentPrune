import sys, os
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
from dataset.mmlu_dataset import MMLUDataset
from dataset.MMLU.download import download
from experiments.train_mmlu import train
from experiments.evaluate_mmlu import evaluate
from AgentPrune.utils.const import AgentPrune_ROOT
from AgentPrune.utils.globals import PromptTokens, CompletionTokens

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


def freeze_graph_for_evaluation(graph: Graph) -> None:
    graph.optimized_spatial = False
    graph.optimized_temporal = False
    graph.spatial_logits.requires_grad_(False)
    graph.temporal_logits.requires_grad_(False)




def should_use_structured_mmlu_output(active_edge_message_mode: str) -> bool:
    return active_edge_message_mode in {
        'full_output_xml',
    }



def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered','Star', 'Mesh',
                                 'FakeFullConnected','FakeRandom','FakeChain','FakeStar','FakeMesh','FakeAGRandom','FakeAGFull'],
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
    parser.add_argument('--num_rounds',type=int,default=2,
                        help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25,
                        help="The Rate of Pruning. Default 0.05.")
    parser.add_argument('--llm_name', type=str, default="gpt-4-1106-preview",
                        help="Model name, None runs the default ChatGPT4")
    parser.add_argument('--domain', type=str, default="mmlu",
                        help="Domain (the same as dataset name), default 'MMLU'")
    parser.add_argument('--decision_method', type=str, default="FinalRefer",
                        help="the decision method of the final node")
    parser.add_argument('--log_agent_outputs', action='store_true',
                        help="Print each agent's raw output and extracted answer to stdout/logs.")
    parser.add_argument('--limit_questions', type=int, default=153,
                        help="Maximum number of validation questions to evaluate.")
    parser.add_argument('--optimized_spatial',action='store_true')
    parser.add_argument('--optimized_temporal',action='store_true')
    parser.add_argument('--freeze_after_training', action='store_true', default=True,
                        help="After training/pruning, freeze the learned graph for evaluation so kept edges are no longer resampled.")
    parser.add_argument(
        '--active_edge_message_mode',
        type=str,
        default='full_output',
        choices=[
            'full_output',
            'full_output_xml',
        ],
        help="High-level active-edge message-sharing mode for retained communication edges.",
    )
    parser.add_argument('--temperature', type=float, default=0.0,
                        help="LLM generation temperature. Default 0.0 for near-deterministic output.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for Python, NumPy, and PyTorch. Default 42.")
    args = parser.parse_args()
    result_path = AgentPrune_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
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
    
    mode = args.mode
    decision_method = args.decision_method
    agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(
        mode,
        len(agent_names),
        use_structured_output=should_use_structured_mmlu_output(args.active_edge_message_mode),
    )
    limit_questions = args.limit_questions
    
    graph = Graph(domain=args.domain,
                  llm_name=args.llm_name,
                  agent_names=agent_names,
                  decision_method=decision_method,
                  log_node_outputs=args.log_agent_outputs,
                  llm_temperature=args.temperature,
                  optimized_spatial=args.optimized_spatial,
                  optimized_temporal=args.optimized_temporal,
                  **kwargs)
    download()
    dataset_train = MMLUDataset('dev')
    dataset_val = MMLUDataset('val')
    train_prompt_used = 0.0
    train_completion_used = 0.0
    train_agent_metrics: List[Dict[str, Any]] = []
    
    if args.optimized_spatial or args.optimized_temporal:
        graph.set_eval_mode(False)
        train_prompt_start, train_completion_start = get_token_snapshot()
        train_agent_start = get_agent_token_snapshot(graph)
        await train(graph=graph,dataset=dataset_train,num_iters=args.num_iterations,num_rounds=args.num_rounds,
                    lr=args.lr,batch_size=args.batch_size,imp_per_iters=args.imp_per_iterations,pruning_rate=args.pruning_rate)
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
            dataset=dataset_val,
            num_rounds=args.num_rounds,
            limit_questions=limit_questions,
            eval_batch_size=args.batch_size,
        )
    eval_prompt_used = PromptTokens.instance().value - eval_prompt_start
    eval_completion_used = CompletionTokens.instance().value - eval_completion_start
    print(f"Baseline Score: {score}")
    print(f"Baseline Eval Prompt Tokens: {eval_prompt_used}")
    print(f"Baseline Eval Completion Tokens: {eval_completion_used}")

    eval_agent_metrics = get_phase_agent_token_metrics(graph, eval_agent_start)
    print_phase_token_summary("Inference", eval_prompt_used, eval_completion_used)
    print_phase_agent_token_summary("Inference", eval_agent_metrics)
    print(f"Score: {score}")
    print(f"Prompt Tokens: {PromptTokens.instance().value}")
    print(f"Completion Tokens: {CompletionTokens.instance().value}")
    print(f"Total Tokens: {PromptTokens.instance().value + CompletionTokens.instance().value}")
    print_agent_token_summary(graph)



def get_kwargs(mode:Union[Literal['DirectAnswer'],Literal['FullConnected'],Literal['Random'],Literal['Chain'],Literal['Debate'],Literal['Layered'],Literal['Star'],Literal['Mesh'],
                          Literal['FakeFullConnected'],Literal['FakeRandom'],Literal['FakeChain'],Literal['FakeStar'],Literal['FakeMesh'],Literal['FakeAGRandom'],Literal['FakeAGFull']],
               N:int,
               use_structured_output: bool = False):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks:List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks:List[List[int]] = None
    node_kwargs = None
    
    def generate_layered_graph(N,layer_num=2):
        adj_matrix = [[0]*N for _ in range(N)]
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(N):
            current_layer = layers[i]
            for j in range(N):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_mesh_graph(N):
        adj_matrix = [[0] * N for _ in range(N)]
        for i in range(0, N):
            for j in range(i+1,N):
                adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_star_graph(N):
        adj_matrix = [[0] * N for _ in range(N)]
        for i in range(1,N):
            adj_matrix[0][i] = 1
        return adj_matrix
    
    if mode=='DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role':'Normal'}]
    elif mode=='FullConnected' or mode == 'FakeFullConnected' or mode=='FakeAGFull':
        fixed_spatial_masks = [[1 if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode=='Random' or mode == 'FakeRandom' or mode == 'FakeAGRandom':
        fixed_spatial_masks = [[random.randint(0, 1)  if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode=='Chain' or mode == 'FakeChain':
        fixed_spatial_masks = [[1 if i==j+1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==0 and j==N-1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Mesh' or mode=='FakeMesh':
        fixed_spatial_masks = generate_mesh_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star' or mode=='FakeStar':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    
    if 'Fake' in mode and 'AG' not in mode:
        node_kwargs = [{'role':'Fake'} if i % 2 == N % 2 else {'role':'Normal'} for i in range(N)]
    elif 'Fake' in mode and 'AG' in mode:
        node_kwargs = [{'role':'Fake'} if i % 2 == N % 2 else {'role':None} for i in range(N)]

    if node_kwargs is None:
        node_kwargs = [{} for _ in range(N)]
    node_kwargs = [{**kwargs, 'use_structured_output': use_structured_output} for kwargs in node_kwargs]
        
    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs":node_kwargs}    

if __name__ == "__main__":
    asyncio.run(main())
