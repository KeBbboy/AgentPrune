import torch
import torch.nn.functional as F
from typing import Any, Iterator
import pandas as pd
import numpy as np
import time
import asyncio
from typing import List
import copy
import random

from AgentDropout.graph.graph import Graph
from AgentDropout.utils.globals import Cost, PromptTokens, CompletionTokens
from AgentDropout.utils.utils import nuclear_norm,frobenius_norm


def _compute_utility(dataset, prediction: str, target) -> float:
    if target is None:
        return 0.0
    answer_metrics = getattr(dataset, "answer_metrics", None)
    if callable(answer_metrics):
        metrics = answer_metrics(prediction, target)
        if isinstance(metrics, dict):
            if "f1" in metrics:
                return float(metrics["f1"])
            if "em" in metrics:
                return float(metrics["em"])
            if len(metrics) > 0:
                return float(next(iter(metrics.values())))
    return float(prediction == target)


def _message_selection_enabled(args) -> bool:
    return getattr(args, "message_selection_policy", "none") != "none"


def _message_selection_regularization(graph: Graph, args) -> torch.Tensor:
    return graph.message_selection_regularization(
        cost_weight=getattr(args, "message_cost_weight", 0.0),
        entropy_weight=getattr(args, "message_entropy_weight", 0.0),
        diversity_weight=getattr(args, "message_diversity_weight", 0.0),
        context_weight_decay=getattr(args, "message_context_weight_decay", 0.0),
    )


def _message_contextual_regularization(graph: Graph, args) -> torch.Tensor:
    return graph.contextual_message_regularization(
        entropy_weight=getattr(args, "message_context_entropy_weight", 0.0),
        balance_weight=getattr(args, "message_balance_weight", 0.0),
    )


def _message_action_penalty(graph: Graph, args) -> float:
    if getattr(args, "message_selection_policy", "none") != "aims_plus":
        return 0.0
    counts = getattr(graph, "message_action_counts", {}).get("total", {})
    penalty = (
        float(getattr(args, "message_reason_penalty_weight", 0.0)) * counts.get("reason", 0)
        + float(getattr(args, "message_full_penalty_weight", 0.0)) * counts.get("full", 0)
    )
    collapse_weight = float(getattr(args, "message_action_collapse_penalty_weight", 0.0))
    collapse_threshold = float(getattr(args, "message_action_collapse_threshold", 1.0))
    action_total = sum(counts.values())
    if collapse_weight > 0.0 and action_total > 0:
        max_action_ratio = max(counts.values()) / action_total
        penalty += collapse_weight * max(0.0, max_action_ratio - collapse_threshold)
    return penalty


def _max_resample_attempts(args) -> int:
    retry_count = max(0, int(getattr(args, "max_resample_attempts", 0)))
    return retry_count + 1


def _training_regularization_loss(graph: Graph, args, kwargs) -> torch.Tensor:
    agent_num = sum(args.agent_nums)
    if not graph.diff:
        spatial_matrix_train = graph.spatial_logits.reshape((agent_num, agent_num))
        temporal_matrix_train = graph.temporal_logits.reshape((agent_num, agent_num))
    else:
        spatial_matrix_train = [
            param.reshape((agent_num, agent_num))
            for param in graph.spatial_logits
        ]
        temporal_matrix_train = [
            param.reshape((agent_num, agent_num))
            for param in graph.temporal_logits
        ]

    spatial_matrix_fixed = torch.tensor(
        kwargs["fixed_spatial_masks"],
        dtype=torch.float32,
        device=graph.message_complexity_action_bias.device,
    ).reshape((agent_num, agent_num))
    temporal_matrix_fixed = torch.tensor(
        kwargs["fixed_temporal_masks"],
        dtype=torch.float32,
        device=graph.message_complexity_action_bias.device,
    ).reshape((agent_num, agent_num))

    if not graph.diff:
        loss_s = nuclear_norm(spatial_matrix_train)
        loss_t = nuclear_norm(temporal_matrix_train)
        frob_loss_s = frobenius_norm(spatial_matrix_fixed, spatial_matrix_train)
        frob_loss_t = frobenius_norm(temporal_matrix_fixed, temporal_matrix_train)
    else:
        loss_s = torch.mean(torch.stack([
            nuclear_norm(matrix)
            for matrix in spatial_matrix_train
        ]))
        loss_t = torch.mean(torch.stack([
            nuclear_norm(matrix)
            for matrix in temporal_matrix_train
        ]))
        frob_loss_s = torch.mean(torch.stack([
            frobenius_norm(spatial_matrix_fixed, matrix)
            for matrix in spatial_matrix_train
        ]))
        frob_loss_t = torch.mean(torch.stack([
            frobenius_norm(temporal_matrix_fixed, matrix)
            for matrix in temporal_matrix_train
        ]))

    add_loss = loss_s + loss_t + F.relu(frob_loss_s - args.delta) + F.relu(frob_loss_t - args.delta)
    return add_loss + _message_selection_regularization(graph, args)


async def _run_record_with_resampling(
        graph: Graph,
        dataset,
        record: Any,
        correct_answer: Any,
        num_rounds: int,
        args,
        ):
    max_attempts = _max_resample_attempts(args)
    input_dict = dataset.record_to_input(record)
    attempt_results = []

    for attempt_idx in range(max_attempts):
        realized_graph = copy.deepcopy(graph)
        realized_graph.share_second_stage_parameters(graph)
        raw_answer, log_prob = await realized_graph.arun(input_dict, num_rounds)
        answer = dataset.postprocess_answer(raw_answer)
        utility = _compute_utility(dataset, answer, correct_answer)
        attempt_results.append((
            raw_answer,
            log_prob,
            realized_graph,
            answer,
            utility,
            attempt_idx + 1,
        ))
        if utility > 0.0:
            break
        if attempt_idx + 1 < max_attempts:
            retry_idx = attempt_idx + 1
            print(
                "Resampling wrong training sample "
                f"(retry {retry_idx}/{max_attempts - 1}) "
                f"answer:{answer} correct:{correct_answer}"
            )

    return attempt_results


async def train(graph:Graph,
            dataset,
            num_iters:int=100,
            num_rounds:int=1,
            lr:float=0.1,
            batch_size:int = 4,
            imp_per_iters: int = 1,
            pruning_rate: float = 0.05,
            args=None,
            kwargs=None,
          ) -> None:

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_iters <= 0:
        raise ValueError("num_iters must be positive")
    if imp_per_iters <= 0:
        raise ValueError("imp_per_iters must be positive")
    
    def infinite_data_loader() -> Iterator[pd.DataFrame]:
        while True:
            for idx in range(len(dataset)):
                record = dataset[idx]
                yield record
    
    loader = infinite_data_loader()
    if args.dec:
        graph.optimized_spatial=False
        graph.optimized_temporal=False
        
        if not graph.diff:
            optimizer = torch.optim.Adam([graph.spatial_logits_1,graph.temporal_logits_1], lr=lr)
        # elif graph.diff and graph.dec_1:
        #     optimizer = torch.optim.Adam(list(graph.spatial_logits.parameters()) + list(graph.temporal_logits.parameters())+[graph.decision_logits],lr=lr)
        else:
            optimizer = torch.optim.Adam(list(graph.spatial_logits_1.parameters()) + list(graph.temporal_logits_1.parameters()),lr=lr)
        for i_iter in range(num_iters):
            print(f"Train {i_iter}", 80*'-')
            start_ts = time.time()
            correct_answers = []
            answer_log_probs = []
            add_losses = []
            for i_record, record in zip(range(batch_size), loader):
                realized_graph = copy.deepcopy(graph)
                realized_graph.spatial_logits_1 = graph.spatial_logits_1
                realized_graph.temporal_logits_1 = graph.temporal_logits_1
                # if graph.dec_1:
                #     realized_graph.decision_logits = graph.decision_logits
                
                # print(graph.spatial_logits)
                if not graph.diff:
                    spatial_matrix_train = realized_graph.spatial_logits_1.reshape((sum(args.agent_nums),sum(args.agent_nums)))
                    temporal_matrix_train = realized_graph.temporal_logits_1.reshape((sum(args.agent_nums),sum(args.agent_nums)))
                else:
                    spatial_matrix_train = [param.reshape((sum(args.agent_nums), sum(args.agent_nums))) for param in realized_graph.spatial_logits_1]
                    temporal_matrix_train = [param.reshape((sum(args.agent_nums), sum(args.agent_nums))) for param in realized_graph.temporal_logits_1]

                spatial_matrix_fixed = torch.tensor(kwargs["fixed_spatial_masks"],dtype=torch.float32).reshape((sum(args.agent_nums),sum(args.agent_nums)))
                temporal_matrix_fixed = torch.tensor(kwargs["fixed_temporal_masks"],dtype=torch.float32).reshape((sum(args.agent_nums),sum(args.agent_nums)))
                if not graph.diff:
                    loss_s = nuclear_norm(spatial_matrix_train)
                    loss_t = nuclear_norm(temporal_matrix_train)
                    frob_loss_s = frobenius_norm(spatial_matrix_fixed, spatial_matrix_train)
                    frob_loss_t = frobenius_norm(temporal_matrix_fixed, temporal_matrix_train)
                    loss_cs = connectivity_loss(spatial_matrix_train)
                    loss_ct = connectivity_loss(temporal_matrix_train)
                    # print(loss_cs)
                else:
                    # loss_s = sum(nuclear_norm(matrix) for matrix in spatial_matrix_train)
                    # loss_t = sum(nuclear_norm(matrix) for matrix in temporal_matrix_train)
                    # frob_loss_s = sum(frobenius_norm(spatial_matrix_fixed, matrix) for matrix in spatial_matrix_train)
                    # frob_loss_t = sum(frobenius_norm(temporal_matrix_fixed, matrix) for matrix in temporal_matrix_train)
                    loss_s = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in spatial_matrix_train]))
                    loss_t = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in temporal_matrix_train]))
                    loss_cs = torch.mean(torch.stack([connectivity_loss(matrix) for matrix in spatial_matrix_train]))
                    loss_ct = torch.mean(torch.stack([connectivity_loss(matrix) for matrix in temporal_matrix_train]))
                    frob_loss_s = torch.mean(torch.stack([frobenius_norm(spatial_matrix_fixed, matrix) for matrix in spatial_matrix_train]))
                    frob_loss_t = torch.mean(torch.stack([frobenius_norm(temporal_matrix_fixed, matrix) for matrix in temporal_matrix_train]))
                add_loss = loss_s + loss_t + F.relu(frob_loss_s - args.delta) + F.relu(frob_loss_t - args.delta)
                add_loss = 0
                input_dict = dataset.record_to_input(record)
                if args.dec:
                    answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,num_rounds,skip=True)))
                else:
                    answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,num_rounds)))
                correct_answer = dataset.record_to_target_answer(record)
                correct_answers.append(correct_answer)
                add_losses.append(add_loss)
                
            raw_results = await asyncio.gather(*answer_log_probs)
            raw_answers, log_probs = zip(*raw_results)
            loss_list: List[torch.Tensor] = []
            utilities: List[float] = []
            answers: List[str] = []
            
            for raw_answer, log_prob, add_loss, correct_answer in zip(raw_answers, log_probs, add_losses, correct_answers):
                answer = dataset.postprocess_answer(raw_answer)
                answers.append(answer)
                utility = _compute_utility(dataset, answer, correct_answer)
                utilities.append(utility)
                single_loss = - log_prob * utility
                loss_list.append(single_loss+add_loss)
                print(f"correct answer:{correct_answer}")
        
            total_loss = torch.mean(torch.stack(loss_list))
            optimizer.zero_grad() 
            total_loss.backward()
            optimizer.step()
            if not graph.diff:
                spatial_probs = torch.sigmoid(graph.spatial_logits_1)
                temporal_probs = torch.sigmoid(graph.temporal_logits_1)
            else:
                spatial_probs = [torch.sigmoid(logit) for logit in graph.spatial_logits_1]
                temporal_probs = [torch.sigmoid(logit) for logit in graph.temporal_logits_1]
            
            print("raw_answers:",raw_answers)
            print("answers:",answers)
            print(f"Batch time {time.time() - start_ts:.3f}")
            print("utilities:", utilities)
            print("loss:", total_loss.item())
            print("Spatial logits Grad:", graph.spatial_logits_1[0].grad)
            # print("Temporal logits Grad:", graph.spatial_logits.grad)
            print("Spatial logits:", graph.spatial_logits_1)
            print("Temporal logits:", graph.temporal_logits_1)
            # if graph.dec_1:
            #     print("Decision logits:", graph.decision_logits)
            print("Spatial probs:", spatial_probs)
            print("Temporal probs:", temporal_probs)
            print(f"Cost {Cost.instance().value}")
            print(f"PromptTokens {PromptTokens.instance().value}")
            print(f"CompletionTokens {CompletionTokens.instance().value}")
            if (i_iter + 1) % imp_per_iters == 0:
                graph.update_masks_dec()

    loader = infinite_data_loader()

    graph.optimized_spatial=True
    graph.optimized_temporal=True
    graph.set_message_type_learning_mode(_message_selection_enabled(args))
    optimizer = torch.optim.Adam(graph.second_stage_optimizer_parameters(), lr=lr)

    for i_iter in range(num_iters):
        print(f"Train {i_iter}", 80*'-')
        start_ts = time.time()
        correct_answers = []
        answer_log_probs = []
        for i_record, record in zip(range(batch_size), loader):
            correct_answer = dataset.record_to_target_answer(record)
            correct_answers.append(correct_answer)
            answer_log_probs.append(asyncio.create_task(
                _run_record_with_resampling(
                    graph=graph,
                    dataset=dataset,
                    record=record,
                    correct_answer=correct_answer,
                    num_rounds=num_rounds,
                    args=args,
                )
            ))
            
        raw_attempt_batches = await asyncio.gather(*answer_log_probs)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        answers: List[str] = []
        raw_answers = []
        attempt_counts = []
        policy_entries = []
        
        for attempt_results, correct_answer in zip(raw_attempt_batches, correct_answers):
            attempt_counts.append(len(attempt_results))
            print(f"correct answer:{correct_answer}")
            for raw_answer, log_prob, realized_graph, answer, utility, attempt_idx in attempt_results:
                action_penalty = _message_action_penalty(realized_graph, args)
                effective_utility = utility - action_penalty
                utilities.append(utility)
                answers.append(answer)
                raw_answers.append(raw_answer)
                policy_entries.append({
                    "log_prob": log_prob,
                    "effective_utility": effective_utility,
                    "regularization": _message_contextual_regularization(realized_graph, args),
                })
                if attempt_idx > 1:
                    print(f"resample attempt:{attempt_idx} answer:{answer} utility:{utility:.4f}")
                if action_penalty > 0.0:
                    print(f"AIMS+ action penalty:{action_penalty:.4f} effective utility:{effective_utility:.4f}")

        mean_effective_utility = (
            sum(entry["effective_utility"] for entry in policy_entries)
            / max(1, len(policy_entries))
        )
        effective_utilities = []
        for entry in policy_entries:
            advantage = entry["effective_utility"] - mean_effective_utility
            effective_utilities.append(entry["effective_utility"])
            single_loss = - entry["log_prob"] * advantage
            loss_list.append(single_loss + entry["regularization"])
    
        total_loss = torch.mean(torch.stack(loss_list)) + _training_regularization_loss(graph, args, kwargs)
        optimizer.zero_grad() 
        total_loss.backward()
        optimizer.step()
        if not graph.diff:
            spatial_probs = torch.sigmoid(graph.spatial_logits)
            temporal_probs = torch.sigmoid(graph.temporal_logits)
        else:
            spatial_probs = [torch.sigmoid(logit) for logit in graph.spatial_logits]
            temporal_probs = [torch.sigmoid(logit) for logit in graph.temporal_logits]
        
        print("raw_answers:",raw_answers)
        print("answers:",answers)
        print("resample attempts:", attempt_counts)
        print(f"Batch time {time.time() - start_ts:.3f}")
        print("utilities:", utilities)
        print("effective utilities:", effective_utilities)
        print("loss:", total_loss.item()) 
        # print("Spatial logits Grad:", graph.spatial_logits.grad)
        # print("Temporal logits Grad:", graph.spatial_logits.grad)
        print("Spatial logits:", graph.spatial_logits)
        print("Temporal logits:", graph.temporal_logits)
        # if graph.dec_1:
        #     print("Decision logits:", graph.decision_logits)
        print("Spatial probs:", spatial_probs)
        print("Temporal probs:", temporal_probs)
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")
        
        if (i_iter + 1) % imp_per_iters == 0:
            if not graph.diff:
                spatial_masks, temporal_masks = graph.update_masks(pruning_rate)
            else:
                spatial_masks, temporal_masks = graph.update_masks_diff(pruning_rate)
            print("spatial masks:",spatial_masks)
            print("temporal masks:",temporal_masks)
            if not graph.diff:
                print("spatial sparsity:",spatial_masks.sum()/spatial_masks.numel())
                print("temporal sparsity:",temporal_masks.sum()/temporal_masks.numel())
            else:
                print("spatial sparsity:",spatial_masks[0].sum()/spatial_masks[0].numel())
                print("temporal sparsity:",temporal_masks[0].sum()/temporal_masks[0].numel())

    # graph.optimized_spatial=False
    # graph.optimized_temporal=False        


def connectivity_loss(A: torch.Tensor) -> torch.Tensor:

    expA = torch.matrix_exp(A)

    penalty = torch.trace(expA) - A.shape[0]
    return penalty

def set_seed(seed):
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def remove_zero_rows_and_columns(matrix,row_to_delete,col_to_delete):
    new_matrix = torch.cat([matrix[:row_to_delete], matrix[row_to_delete+1:]])
    new_matrix = torch.cat([new_matrix[:, :col_to_delete], new_matrix[:, col_to_delete+1:]], dim=1)

    return new_matrix
