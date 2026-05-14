import sys
import os
import argparse
import yaml
import json
import time
import asyncio
import math
from pathlib import Path
import torch
import copy
from typing import List,Union,Literal, Dict, Any
import random
import re
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from AgentPrune.graph.graph import Graph
from AgentPrune.tools.coding.python_executor import PyExecutor
from dataset.humaneval_dataset import HumanEvalDataset, extract_humaneval_code
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
from experiments.train_mmlu_edge_policy import load_edge_policy_config, train_joint_policy
from AgentPrune.utils.globals import Time
from AgentPrune.utils.const import AgentPrune_ROOT
from AgentPrune.utils.globals import Cost, PromptTokens, CompletionTokens

def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump([], file)

    with open(result_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    return data

def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch*batch_size:i_batch*batch_size + batch_size]

def load_config(config_path):
    with open(config_path, 'r',encoding='utf-8') as file:
        return yaml.safe_load(file)

def get_training_sample_cutoff(dataset_limit: int, args, edge_policy_config: Dict[str, Any]) -> int:
    if not (args.optimized_spatial or args.optimized_temporal):
        return 0

    training_batch_size = int(edge_policy_config.get("batch_size", args.batch_size))
    training_iterations = int(edge_policy_config.get("iterations", args.num_iterations))
    training_pool_size = min(dataset_limit, int(edge_policy_config.get("num_samples", dataset_limit)))

    # Training walks the dataset from the front in batch order and may wrap if
    # iterations * batch_size exceeds the configured training pool.
    return min(dataset_limit, min(training_pool_size, training_batch_size * training_iterations))

def parse_args():
    parser = argparse.ArgumentParser(description="AgentPrune Experiments on HumanEval")
    parser.add_argument("--dataset_json", type=str, default="dataset/humaneval/humaneval-py.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="gpt-4-1106-preview")
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain','Debate','Layered','Star'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1,help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4,help="batch size")
    parser.add_argument('--imp_per_iterations', type=int, default=5,help="Prune every few iterations. Default 5.")
    parser.add_argument('--num_rounds',type=int,default=2,help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25,help="The Rate of Pruning. Default 0.05.")
    parser.add_argument('--num_iterations', type=int, default = 10,help="The num of training iterations.")
    parser.add_argument('--domain', type=str, default="humaneval",help="Domain (the same as dataset name), default 'humaneval'")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['CodeWriting'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[5],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--decision_method', type=str, default='FinalWriteCode',
                        help='The decison method of the agentprune')
    parser.add_argument('--log_agent_outputs', action='store_true',
                        help="Print each agent's raw output and extracted answer to stdout/logs.")
    parser.add_argument('--limit_questions', type=int, default=None,
                        help="Maximum number of HumanEval samples to run.")
    parser.add_argument('--temperature', type=float, default=0.0,
                        help="LLM generation temperature. Default 0.0 for near-deterministic output.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for Python, NumPy, and PyTorch. Default 42.")
    parser.add_argument('--freeze_after_training', action='store_true', default=True,
                        help="After the optimization batches finish, switch the graph to evaluation message-sharing mode for the remaining batches.")
    parser.add_argument('--optimized_spatial',action='store_true')
    parser.add_argument('--optimized_temporal',action='store_true')
    parser.add_argument('--edge_policy_config', type=str, default='configs/humaneval_edge_policy.json',
                        help="JSON config for stage-2 answer/reasoning edge content policy training.")
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
    result_file = None
    dataset = HumanEvalDataset(args.dataset_json)
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{AgentPrune_ROOT}/result/eval")
    result_dir.mkdir(parents=True, exist_ok=True)
    safe_llm_name = args.llm_name.replace("/", "_")
    result_file = result_dir / f"{safe_llm_name}_{current_time}.json"
    
    agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
    decision_method = args.decision_method
    kwargs = get_kwargs(
        args.mode,
        len(agent_names),
    )
    graph = Graph(domain="humaneval",
                  llm_name=args.llm_name,
                  agent_names=agent_names,
                  decision_method=decision_method,
                  log_node_outputs=args.log_agent_outputs,
                  llm_temperature=args.temperature,
                  optimized_spatial=args.optimized_spatial,
                  optimized_temporal=args.optimized_temporal,
                  **kwargs)
    edge_policy_config = load_edge_policy_config(args.edge_policy_config)

    dataset_limit = min(len(dataset), args.limit_questions) if args.limit_questions is not None else len(dataset)
    total_solved, total_executed = (0, 0)
    train_prompt_start, train_completion_start = get_token_snapshot()
    train_agent_start = get_agent_token_snapshot(graph)
    train_agent_metrics: List[Dict[str, Any]] = []
    train_mode_enabled = args.optimized_spatial or args.optimized_temporal

    if train_mode_enabled:
        await train_joint_policy(
            graph=graph,
            dataset=dataset,
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
        train_prompt_used = 0.0
        train_completion_used = 0.0
        print_phase_token_summary("Training", train_prompt_used, train_completion_used)
        print_phase_agent_token_summary("Training", train_agent_metrics)

    if args.freeze_after_training:
        freeze_graph_for_evaluation(graph)
        print("Evaluation graph frozen: kept edges use the learned masks/logits without resampling.")

    inference_start_index = get_training_sample_cutoff(dataset_limit, args, edge_policy_config)
    inference_records = [dataset[i] for i in range(inference_start_index, dataset_limit)]
    inference_num_batches = int(math.ceil(len(inference_records) / args.batch_size)) if inference_records else 0
    if train_mode_enabled:
        print(
            "Inference starts after training-covered samples: "
            f"skip first {inference_start_index} samples, "
            f"run remaining {len(inference_records)} samples."
        )

    graph.set_eval_mode(True)
    inference_prompt_start, inference_completion_start = get_token_snapshot()
    inference_agent_start = get_agent_token_snapshot(graph)

    if not inference_records:
        print("No remaining samples for inference after training.")

    for i_batch in range(inference_num_batches):
        start_ts = time.time()
        answer_log_probs = []
        tests = []
        
        current_batch = dataloader(inference_records, args.batch_size, i_batch)
        if not current_batch:
            print("No more data available.")
            break

        batch_start_index = inference_start_index + i_batch * args.batch_size
        batch_end_index = batch_start_index + len(current_batch) - 1
        batch_label = (
            f"Batch {batch_start_index // args.batch_size} "
            f"(samples {batch_start_index + 1}-{batch_end_index + 1})"
        )
        print(batch_label, 80 * '-')
        
        for i_record, record in enumerate(current_batch):
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits
            realized_graph.share_cost_state(graph)
            realized_graph.current_sample_index = inference_start_index + i_batch * args.batch_size + i_record + 1
            task = record["prompt"]
            test = record["test"]
            tests.append(test)
            input_dict = dataset.record_to_input(record)
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,args.num_rounds)))
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        data = load_result(result_file)
        
        for task, answer, log_prob, test in zip(current_batch, raw_answers, log_probs, tests):
            if not isinstance(answer,list):
                raise TypeError(f"Expected a list for the answer, but got {type(answer).__name__}")
            answer = dataset.postprocess_answer(answer, record=task)
            is_solved, _, _ = PyExecutor().execute(answer, [test], timeout=100)
            total_solved = total_solved + is_solved
            total_executed = total_executed + 1
            accuracy = total_solved/ total_executed
            utility = is_solved
            utilities.append(utility)
            single_loss = -log_prob * utility
            loss_list.append(single_loss)
            updated_item = {
                "Question": task,
                "Tests": test,
                "Attempt answer": answer,
                "Solved": is_solved,
                "Solution": answer,
                "Total solved": total_solved,
                "Total executed": total_executed,
                "Accuracy": accuracy
            }
            data.append(updated_item)
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        
        print(f"Batch time {time.time() - start_ts:.3f}")
        print(f"Accuracy: {accuracy}")
        print("utilities:", utilities)

    inference_prompt_used = PromptTokens.instance().value - inference_prompt_start
    inference_completion_used = CompletionTokens.instance().value - inference_completion_start
    inference_agent_metrics = get_phase_agent_token_metrics(graph, inference_agent_start)
    print_phase_token_summary("Inference", inference_prompt_used, inference_completion_used)
    print_phase_agent_token_summary("Inference", inference_agent_metrics)
    print(f"Prompt Tokens: {PromptTokens.instance().value}")
    print(f"Completion Tokens: {CompletionTokens.instance().value}")
    print(f"Total Tokens: {PromptTokens.instance().value + CompletionTokens.instance().value}")
    print_agent_token_summary(graph)



def get_kwargs(mode:Union[Literal['DirectAnswer'],Literal['FullConnected'],Literal['Random'],Literal['Chain'],Literal['Debate'],Literal['Layered'],Literal['Star']],
               N:int,
               use_structured_output: bool = False):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks:List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks:List[List[int]] = None
    node_kwargs = None
    
    def generate_layered_graph(N,layer_num=2):
        adj_matrix = [[0 for _ in range(N)] for _ in range(N)]
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
    
    def generate_star_graph(n):
        matrix = [[0] * n for _ in range(n)]
        for i in range(0, n):
            for j in range(i+1,n):
                matrix[i][j] = 1
        return matrix

    if mode=='DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role':'Programming Expert'}]
    elif mode=='FullConnected':
        fixed_spatial_masks = [[1 if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode=='Random':
        fixed_spatial_masks = [[random.randint(0, 1)  if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode=='Chain':
        fixed_spatial_masks = [[1 if i==j+1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==0 and j==N-1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    
    if node_kwargs is None:
        node_kwargs = [{} for _ in range(N)]
    node_kwargs = [{**kwargs, 'use_structured_output': use_structured_output} for kwargs in node_kwargs]

    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs":node_kwargs}    

if __name__ == '__main__':
    asyncio.run(main())
