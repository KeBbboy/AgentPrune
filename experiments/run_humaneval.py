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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from AgentPrune.graph.graph import Graph
from AgentPrune.tools.reader.readers import JSONLReader
from AgentPrune.tools.coding.python_executor import PyExecutor
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

def extract_humaneval_code(output: str) -> str:
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", output, flags=re.IGNORECASE | re.DOTALL)
    if answer_match is not None:
        return answer_match.group(1).strip()
    return output.strip()

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
    parser.add_argument(
        '--active_edge_message_mode',
        type=str,
        default='full_output',
        choices=[
            'full_output',
            'full_output_xml',
        ],
        help="Active-edge message-sharing mode applied after training batches finish.",
    )
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
    result_file = None
    dataset = JSONLReader.parse_file(args.dataset_json)
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
        use_structured_output=args.active_edge_message_mode in {
            'full_output_xml',
        },
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
    graph.set_eval_mode(False)
    optimizer = torch.optim.Adam([graph.spatial_logits,graph.temporal_logits], lr=args.lr)    
    
    dataset_limit = min(len(dataset), args.limit_questions) if args.limit_questions is not None else len(dataset)
    num_batches = int(math.ceil(dataset_limit / args.batch_size))
    total_solved, total_executed = (0, 0)
    train_mode_enabled = args.optimized_spatial or args.optimized_temporal
    training_batch_limit = min(num_batches, args.num_iterations) if train_mode_enabled else 0
    no_training_mode = training_batch_limit == 0
    train_prompt_start, train_completion_start = get_token_snapshot()
    train_agent_start = get_agent_token_snapshot(graph)
    inference_prompt_start = train_prompt_start if no_training_mode else None
    inference_completion_start = train_completion_start if no_training_mode else None
    inference_agent_start = train_agent_start if no_training_mode else None
    training_phase_closed = no_training_mode
    for i_batch in range(num_batches):
        print(f"Batch {i_batch}",80*'-')
        start_ts = time.time()
        answer_log_probs = []
        tests = []
        
        current_batch = dataloader(dataset[:dataset_limit], args.batch_size, i_batch)
        if not current_batch:
            print("No more data available.")
            break
        
        for i_record, record in enumerate(current_batch):
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits
            realized_graph.share_cost_state(graph)
            realized_graph.current_sample_index = i_batch * args.batch_size + i_record + 1
            task = record["prompt"]
            test = record["test"]
            tests.append(test)
            input_dict = {"task": task}
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,args.num_rounds)))
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        data = load_result(result_file)
        
        for task, answer, log_prob, test in zip(current_batch, raw_answers, log_probs, tests):
            if not isinstance(answer,list):
                raise TypeError(f"Expected a list for the answer, but got {type(answer).__name__}")
            answer = extract_humaneval_code(answer[0]).lstrip("```python\n").rstrip("\n```")
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
        
        total_loss = torch.mean(torch.stack(loss_list))
        if args.optimized_spatial or args.optimized_temporal:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        spatial_probs = torch.sigmoid(graph.spatial_logits)
        temporal_probs = torch.sigmoid(graph.temporal_logits)
        
        print(f"Batch time {time.time() - start_ts:.3f}")
        print(f"Accuracy: {accuracy}")
        print("utilities:", utilities)

        if (i_batch+1)%args.imp_per_iterations == 0 and i_batch < args.num_iterations and (args.optimized_spatial or args.optimized_temporal):
            spatial_masks, temporal_masks = graph.update_masks(args.pruning_rate)
        if i_batch+1 == args.num_iterations:
            args.optimized_spatial = False
            args.optimized_temporal = False
            graph.set_eval_mode(True)
        if (not training_phase_closed) and (i_batch + 1 == training_batch_limit):
            train_prompt_used = PromptTokens.instance().value - train_prompt_start
            train_completion_used = CompletionTokens.instance().value - train_completion_start
            train_agent_metrics = get_phase_agent_token_metrics(graph, train_agent_start)
            if training_batch_limit > 0:
                print_phase_token_summary("Training", train_prompt_used, train_completion_used)
                print_phase_agent_token_summary("Training", train_agent_metrics)
            else:
                print("Training skipped: both optimized_spatial and optimized_temporal are disabled.")
                print_phase_token_summary("Training", 0.0, 0.0)
                print_phase_agent_token_summary("Training", [])
            inference_prompt_start, inference_completion_start = get_token_snapshot()
            inference_agent_start = get_agent_token_snapshot(graph)
            training_phase_closed = True

    if no_training_mode:
        print("Training skipped: both optimized_spatial and optimized_temporal are disabled.")
        print_phase_token_summary("Training", 0.0, 0.0)
        print_phase_agent_token_summary("Training", [])

    if inference_prompt_start is None or inference_completion_start is None or inference_agent_start is None:
        inference_prompt_start, inference_completion_start = get_token_snapshot()
        inference_agent_start = get_agent_token_snapshot(graph)

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
