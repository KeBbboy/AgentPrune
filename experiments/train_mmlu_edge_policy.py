import asyncio
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from AgentPrune.graph.edge_policy import EdgeContentPolicy
from AgentPrune.graph.graph import Graph
from AgentPrune.utils.globals import CompletionTokens, PromptTokens


DEFAULT_EDGE_POLICY_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "num_samples": 40,
    "batch_size": 4,
    "iterations": 10,
    "lr": 0.001,
    "hidden_dim": 32,
    "initial_answer_bias": 0.0,
    "initial_step_bias": 0.0,
    "min_spatial_step_edges_eval": 2,
    "min_temporal_step_edges_eval": 4,
    "token_budget_mode": "fixed_total_average",
    "token_budget": 917000,
    "token_budget_num_samples": 153,
    "lambda_max": 0.5,
    "step_token_lambda_max": 0.3,
    "step_probability_penalty_max": 0.05,
    "lambda_warmup_ratio": 0.2,
    "entropy_beta_start": 0.02,
    "entropy_beta_end": 0.005,
    "baseline_momentum": 0.9,
    "save_path": None,
}


def load_edge_policy_config(config_path: str | None) -> Dict[str, Any]:
    config = dict(DEFAULT_EDGE_POLICY_CONFIG)
    if config_path is None:
        return config

    path = Path(config_path)
    if not path.exists():
        print(f"[EdgePolicy] Config not found: {path}. Using defaults with enabled=False.")
        return config

    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    config.update(loaded)
    return config


def _lambda_at(iter_idx: int, total_iters: int, warmup_ratio: float, lambda_max: float) -> float:
    warmup_iters = int(total_iters * warmup_ratio)
    if iter_idx < warmup_iters:
        return 0.0
    denom = max(1, total_iters - warmup_iters)
    progress = min(max((iter_idx - warmup_iters + 1) / denom, 0.0), 1.0)
    return lambda_max * progress


def _beta_at(iter_idx: int, total_iters: int, beta_start: float, beta_end: float) -> float:
    denom = max(1, total_iters - 1)
    progress = min(max(iter_idx / denom, 0.0), 1.0)
    return beta_start + (beta_end - beta_start) * progress


def _active_spatial_edge_count(graph: Graph) -> int:
    count = 0
    for (src_id, dst_id), edge_mask in zip(graph.potential_spatial_edges, graph.spatial_masks):
        if src_id != dst_id and edge_mask > 0:
            count += 1
    return count


def _active_temporal_edge_count(graph: Graph) -> int:
    count = 0
    for edge_mask in graph.temporal_masks:
        if edge_mask > 0:
            count += 1
    return count


def _slice_dataset(dataset, num_samples: int) -> List[Any]:
    return [dataset[i] for i in range(min(num_samples, len(dataset)))]


def _node_label(graph: Graph, node_id: str) -> str:
    node = graph.find_node(node_id)
    label = node.role or node.agent_name or node.node_name
    return f"{label}:{node_id}"


def _print_final_edge_decisions(graph: Graph, policy: EdgeContentPolicy) -> None:
    spatial_answer = 0
    spatial_step = 0
    temporal_answer = 0
    temporal_step = 0
    step_rows: List[Dict[str, Any]] = []
    spatial_reasoning_edge_indices = graph._get_eval_reasoning_edge_indices("spatial")
    temporal_reasoning_edge_indices = graph._get_eval_reasoning_edge_indices("temporal")

    for edge_idx, ((src_id, dst_id), edge_logit, edge_mask) in enumerate(
        zip(graph.potential_spatial_edges, graph.spatial_logits, graph.spatial_masks)
    ):
        if src_id == dst_id or edge_mask <= 0:
            continue
        edge_weight = float(torch.sigmoid(edge_logit.detach()).item())
        action = "step" if edge_idx in spatial_reasoning_edge_indices else "answer"
        if action == "answer":
            spatial_answer += 1
        else:
            spatial_step += 1
            step_rows.append({
                "kind": "spatial",
                "edge_idx": edge_idx,
                "src": _node_label(graph, src_id),
                "dst": _node_label(graph, dst_id),
                "weight": edge_weight,
            })

    for edge_idx, ((src_id, dst_id), edge_logit, edge_mask) in enumerate(
        zip(graph.potential_temporal_edges, graph.temporal_logits, graph.temporal_masks)
    ):
        if edge_mask <= 0:
            continue
        edge_weight = float(torch.sigmoid(edge_logit.detach()).item())
        action = "step" if edge_idx in temporal_reasoning_edge_indices else "answer"
        if action == "answer":
            temporal_answer += 1
        else:
            temporal_step += 1
            step_rows.append({
                "kind": "temporal",
                "edge_idx": edge_idx,
                "src": _node_label(graph, src_id),
                "dst": _node_label(graph, dst_id),
                "weight": edge_weight,
            })

    print(
            "[JointTrain][FinalDecisionSummary] "
        f"spatial_answer={spatial_answer} spatial_step={spatial_step} "
        f"temporal_answer={temporal_answer} temporal_step={temporal_step}"
    )
    for row in step_rows:
        print(
            "[JointTrain][FinalStepEdge] "
            f"kind={row['kind']} edge_idx={row['edge_idx']} "
            f"src={row['src']} dst={row['dst']} "
            f"weight={row['weight']:.6f}"
        )


def _active_edge_policy_features(graph: Graph, policy: EdgeContentPolicy) -> torch.Tensor:
    device = next(policy.parameters()).device
    edge_features: List[torch.Tensor] = []

    for edge_idx, ((src_id, dst_id), edge_mask) in enumerate(
        zip(graph.potential_spatial_edges, graph.spatial_masks)
    ):
        if src_id == dst_id or edge_mask <= 0:
            continue
        edge_features.append(graph.get_edge_policy_feature(edge_idx, "spatial", device=device).squeeze(0))

    for edge_idx, edge_mask in enumerate(graph.temporal_masks):
        if edge_mask <= 0:
            continue
        edge_features.append(graph.get_edge_policy_feature(edge_idx, "temporal", device=device).squeeze(0))

    if not edge_features:
        return torch.empty((0, graph.get_edge_policy_feature_dim()), dtype=torch.float32, device=device)
    return torch.stack(edge_features)


async def _collect_sample_baseline_budgets(
    graph: Graph,
    dataset,
    records: List[Any],
    num_rounds: int,
    batch_size: int,
) -> List[int]:
    budgets: List[int] = []
    print("[JointTrain] Collecting per-sample full-message token budgets.")

    for batch_start in range(0, len(records), batch_size):
        batch_records = records[batch_start:batch_start + batch_size]
        realized_graphs: List[Graph] = []
        tasks = []

        for offset, record in enumerate(batch_records):
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits
            realized_graph.spatial_masks = graph.spatial_masks
            realized_graph.temporal_masks = graph.temporal_masks
            realized_graph.edge_content_policy = None
            realized_graph.edge_message_mode = "full"
            realized_graph.edge_policy_training = False
            realized_graph.optimized_spatial = False
            realized_graph.optimized_temporal = False
            realized_graph.set_eval_mode(True)
            realized_graph.current_sample_index = batch_start + offset + 1
            realized_graphs.append(realized_graph)
            tasks.append(asyncio.create_task(realized_graph.arun(dataset.record_to_input(record), num_rounds)))

        await asyncio.gather(*tasks)

        for offset, realized_graph in enumerate(realized_graphs):
            budget = max(1, int(realized_graph.current_run_total_tokens))
            budgets.append(budget)
            print(
                "[JointTrain][Budget] "
                f"sample={batch_start + offset + 1}/{len(records)} "
                f"full_message_tokens={budget}"
            )

    return budgets


async def train_joint_policy(
    graph: Graph,
    dataset,
    num_rounds: int,
    num_iters: int,
    batch_size: int,
    graph_lr: float,
    imp_per_iters: int,
    pruning_rate: float,
    config: Dict[str, Any],
) -> EdgeContentPolicy | None:
    use_content_policy = bool(config.get("enabled", False))
    active_spatial_edges = _active_spatial_edge_count(graph)
    active_temporal_edges = _active_temporal_edge_count(graph)
    active_edges = active_spatial_edges + active_temporal_edges
    if active_edges == 0 and not use_content_policy:
        print("[JointTrain] Training skipped: no active spatial or temporal edges.")
        return None

    num_samples = int(config["num_samples"])
    batch_size = int(config.get("batch_size", batch_size))
    iterations = int(config.get("iterations", num_iters))
    policy_lr = float(config["lr"])
    token_budget = float(config["token_budget"])
    token_budget_num_samples = int(config.get("token_budget_num_samples", 1))
    token_budget_mode = str(config.get("token_budget_mode", "fixed"))
    lambda_max = float(config["lambda_max"])
    step_token_lambda_max = float(config.get("step_token_lambda_max", 1.0))
    lambda_warmup_ratio = float(config["lambda_warmup_ratio"])
    beta_start = float(config["entropy_beta_start"])
    beta_end = float(config["entropy_beta_end"])
    hidden_dim = int(config["hidden_dim"])
    initial_answer_bias = float(config.get("initial_answer_bias", 0.0))
    initial_step_bias = float(config.get("initial_step_bias", 0.0))
    min_spatial_step_edges_eval = int(config.get("min_spatial_step_edges_eval", 2))
    min_temporal_step_edges_eval = int(config.get("min_temporal_step_edges_eval", 4))
    step_probability_penalty_max = float(config.get("step_probability_penalty_max", 0.05))
    baseline_momentum = float(config.get("baseline_momentum", 0.9))

    records = _slice_dataset(dataset, num_samples)
    if len(records) == 0:
        print("[JointTrain] Training skipped: no training records.")
        return None

    policy = None
    optimizer_groups: List[Dict[str, Any]] = []

    graph_trainable_params: List[torch.nn.Parameter] = []
    if graph.optimized_spatial:
        graph.spatial_logits.requires_grad_(True)
        graph_trainable_params.append(graph.spatial_logits)
    else:
        graph.spatial_logits.requires_grad_(False)
    if graph.optimized_temporal:
        graph.temporal_logits.requires_grad_(True)
        graph_trainable_params.append(graph.temporal_logits)
    else:
        graph.temporal_logits.requires_grad_(False)

    if graph_trainable_params:
        optimizer_groups.append({"params": graph_trainable_params, "lr": graph_lr})

    if use_content_policy:
        policy = EdgeContentPolicy(
            input_dim=graph.get_edge_policy_feature_dim(),
            hidden_dim=hidden_dim,
            initial_answer_bias=initial_answer_bias,
            initial_step_bias=initial_step_bias,
        )
        optimizer_groups.append({"params": policy.parameters(), "lr": policy_lr})

    if len(optimizer_groups) == 0:
        print("[JointTrain] Training skipped: no trainable graph parameters and content policy disabled.")
        return None

    optimizer = torch.optim.Adam(optimizer_groups)

    graph.edge_content_policy = policy
    graph.edge_message_mode = "rl_policy" if use_content_policy else "full"
    graph.edge_policy_training = use_content_policy
    graph.edge_policy_min_spatial_step_edges = min_spatial_step_edges_eval if use_content_policy else 0
    graph.edge_policy_min_temporal_step_edges = min_temporal_step_edges_eval if use_content_policy else 0
    graph.set_eval_mode(use_content_policy)

    print("[JointTrain] Unified graph/content training starts.")
    print(
        "[JointTrain] Training data: "
        f"split={getattr(dataset, 'split', 'unknown')}, "
        f"selected_dataset_indices=0..{len(records) - 1}, "
        "order=dataset order after MMLUDataset loading"
    )
    print(
        "[JointTrain] Config: "
        f"samples={len(records)}, batch_size={batch_size}, iterations={iterations}, "
        f"graph_lr={graph_lr}, policy_lr={policy_lr if use_content_policy else 0.0}, "
        f"active_spatial_edges={active_spatial_edges}, "
        f"active_temporal_edges={active_temporal_edges}, "
        f"policy_input_dim={graph.get_edge_policy_feature_dim() if use_content_policy else 0}, "
        f"token_budget_mode={token_budget_mode}, token_budget={token_budget}, "
        f"token_budget_num_samples={token_budget_num_samples}, "
        f"lambda_max={lambda_max}, step_token_lambda_max={step_token_lambda_max}, "
        f"step_probability_penalty_max={step_probability_penalty_max}, "
        f"baseline_momentum={baseline_momentum}, "
        f"eval_min_step_edges=[spatial:{min_spatial_step_edges_eval}, temporal:{min_temporal_step_edges_eval}], "
        f"initial_bias=[answer:{initial_answer_bias}, step:{initial_step_bias}], "
        f"use_content_policy={use_content_policy}, "
        f"beta={beta_start}->{beta_end}"
    )

    global_prompt_start = PromptTokens.instance().value
    global_completion_start = CompletionTokens.instance().value

    sample_budgets: List[int] | None = None
    fixed_total_average_budget = token_budget / max(1, token_budget_num_samples)
    if token_budget_mode == "fixed_total_average":
        print(
            "[JointTrain] Using fixed total-average token budget: "
            f"{token_budget} / {token_budget_num_samples} = "
            f"{fixed_total_average_budget:.2f} tokens/sample"
        )

    if token_budget_mode == "sample_baseline":
        sample_budgets = await _collect_sample_baseline_budgets(
            graph=graph,
            dataset=dataset,
            records=records,
            num_rounds=num_rounds,
            batch_size=batch_size,
        )
    elif token_budget_mode not in {"fixed", "fixed_total_average", "batch_mean"}:
        raise ValueError(
            f"Unsupported token_budget_mode: {token_budget_mode}. "
            "Expected one of: sample_baseline, fixed_total_average, batch_mean, fixed."
        )

    reward_baseline_ema: float | None = None
    content_baseline_ema: float | None = None

    for iter_idx in range(iterations):
        start_ts = time.time()
        batch_records = [
            records[(iter_idx * batch_size + offset) % len(records)]
            for offset in range(batch_size)
        ]
        lambda_t = _lambda_at(iter_idx, iterations, lambda_warmup_ratio, lambda_max)
        step_lambda_t = _lambda_at(
            iter_idx,
            iterations,
            lambda_warmup_ratio,
            step_token_lambda_max,
        )
        step_probability_lambda_t = _lambda_at(
            iter_idx,
            iterations,
            lambda_warmup_ratio,
            step_probability_penalty_max,
        )
        beta_t = _beta_at(iter_idx, iterations, beta_start, beta_end)

        realized_graphs: List[Graph] = []
        tasks = []
        correct_answers: List[str] = []

        for offset, record in enumerate(batch_records):
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits
            realized_graph.spatial_masks = graph.spatial_masks
            realized_graph.temporal_masks = graph.temporal_masks
            realized_graph.share_cost_state(graph)
            realized_graph.edge_content_policy = policy
            realized_graph.edge_message_mode = "rl_policy" if use_content_policy else "full"
            realized_graph.edge_policy_training = use_content_policy
            realized_graph.edge_policy_min_spatial_step_edges = 0
            realized_graph.edge_policy_min_temporal_step_edges = 0
            realized_graph.optimized_spatial = graph.optimized_spatial
            realized_graph.optimized_temporal = graph.optimized_temporal
            realized_graph.set_eval_mode(use_content_policy)
            realized_graph.current_sample_index = iter_idx * batch_size + offset + 1
            realized_graphs.append(realized_graph)

            input_dict = dataset.record_to_input(record)
            correct_answers.append(dataset.record_to_target_answer(record))
            tasks.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds)))

        raw_results = await asyncio.gather(*tasks)
        raw_answers, structure_log_probs = zip(*raw_results)

        sample_infos: List[Dict[str, Any]] = []
        correct_flags: List[float] = []
        token_totals: List[int] = []
        structure_log_prob_sums: List[torch.Tensor] = []
        entropy_sums: List[torch.Tensor] = []
        answer_actions = 0
        reasoning_actions = 0

        for sample_idx, (realized_graph, raw_answer, correct_answer, structure_log_prob) in enumerate(
            zip(realized_graphs, raw_answers, correct_answers, structure_log_probs)
        ):
            record_slot = (iter_idx * batch_size + sample_idx) % len(records)
            predicted_answer = dataset.postprocess_answer(raw_answer, record=batch_records[sample_idx])
            if hasattr(dataset, "compute_reward"):
                correct = float(dataset.compute_reward(raw_answer, batch_records[sample_idx]))
            else:
                correct = float(predicted_answer == correct_answer)
            tokens_used = int(realized_graph.current_run_total_tokens)
            step_tokens = int(realized_graph.current_run_step_tokens)
            action_counts = realized_graph.get_edge_policy_action_counts()
            answer_actions += action_counts.get("answer", 0)
            reasoning_actions += action_counts.get("reasoning", 0)

            correct_flags.append(correct)
            token_totals.append(tokens_used)
            sample_infos.append({
                "record_slot": record_slot,
                "predicted_answer": predicted_answer,
                "correct_answer": correct_answer,
                "correct": correct,
                "tokens_used": tokens_used,
                "step_tokens": step_tokens,
                "spatial_step_tokens": int(realized_graph.current_run_spatial_step_tokens),
                "temporal_step_tokens": int(realized_graph.current_run_temporal_step_tokens),
                "step_tokens_by_edge": realized_graph.get_step_tokens_by_edge(),
                "edge_action_records": realized_graph.get_edge_policy_action_records(),
                "action_counts": action_counts,
            })
            structure_log_prob_sums.append(structure_log_prob)
            entropy_sums.append(realized_graph.get_edge_policy_entropy_sum())

        batch_mean_budget = max(1.0, sum(token_totals) / len(token_totals))
        base_rewards: List[float] = []
        rewards: List[float] = []
        for sample_idx, sample_info in enumerate(sample_infos):
            if token_budget_mode == "sample_baseline" and sample_budgets is not None:
                budget = float(sample_budgets[sample_info["record_slot"]])
            elif token_budget_mode == "fixed_total_average":
                budget = fixed_total_average_budget
            elif token_budget_mode == "batch_mean":
                budget = batch_mean_budget
            else:
                budget = token_budget

            tokens_used = sample_info["tokens_used"]
            step_tokens = sample_info["step_tokens"]
            correct = sample_info["correct"]
            token_penalty = lambda_t * (tokens_used / budget)
            step_token_penalty = step_lambda_t * (step_tokens / budget)
            base_reward = correct - token_penalty
            reward = base_reward - step_token_penalty
            sample_info["budget"] = budget
            sample_info["base_reward"] = base_reward
            sample_info["reward"] = reward
            sample_info["step_token_penalty"] = step_token_penalty
            base_rewards.append(base_reward)
            rewards.append(reward)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        batch_reward_mean = float(rewards_tensor.mean())
        if reward_baseline_ema is None:
            reward_baseline_ema = batch_reward_mean
        else:
            reward_baseline_ema = (
                baseline_momentum * reward_baseline_ema
                + (1.0 - baseline_momentum) * batch_reward_mean
            )
        reward_baseline = reward_baseline_ema
        structure_log_prob_tensor = torch.stack(structure_log_prob_sums)
        structure_advantages = rewards_tensor - float(reward_baseline)
        structure_rl_loss = -torch.mean(structure_advantages.detach() * structure_log_prob_tensor)

        content_rl_loss = torch.tensor(0.0, device=structure_log_prob_tensor.device)
        entropy_loss = torch.tensor(0.0, device=structure_log_prob_tensor.device)
        step_probability_loss = torch.tensor(0.0, device=structure_log_prob_tensor.device)
        step_probability_mean = 0.0

        if use_content_policy and policy is not None:
            base_rewards_tensor = torch.tensor(base_rewards, dtype=torch.float32)
            batch_content_mean = float(base_rewards_tensor.mean())
            if content_baseline_ema is None:
                content_baseline_ema = batch_content_mean
            else:
                content_baseline_ema = (
                    baseline_momentum * content_baseline_ema
                    + (1.0 - baseline_momentum) * batch_content_mean
                )
            content_baseline = content_baseline_ema
            entropy_tensor = torch.stack(entropy_sums)

            sample_policy_terms: List[torch.Tensor] = []
            for sample_info in sample_infos:
                edge_terms: List[torch.Tensor] = []
                base_advantage = sample_info["base_reward"] - float(content_baseline)
                for action_record in sample_info["edge_action_records"]:
                    edge_key = (action_record["edge_kind"], action_record["edge_idx"])
                    edge_step_tokens = sample_info["step_tokens_by_edge"].get(edge_key, 0)
                    edge_step_penalty = step_lambda_t * (edge_step_tokens / sample_info["budget"])
                    edge_advantage = base_advantage - edge_step_penalty
                    edge_terms.append(action_record["log_prob"] * edge_advantage)

                if edge_terms:
                    sample_policy_terms.append(torch.sum(torch.stack(edge_terms)))

            if sample_policy_terms:
                content_rl_loss = -torch.mean(torch.stack(sample_policy_terms))
            entropy_loss = -beta_t * torch.mean(entropy_tensor)
            active_edge_features = _active_edge_policy_features(graph, policy)
            if active_edge_features.numel() > 0:
                step_probabilities = torch.softmax(policy(active_edge_features), dim=-1)[:, 1]
                step_probability_loss = step_probability_lambda_t * torch.mean(step_probabilities)
                step_probability_mean = float(torch.mean(step_probabilities.detach()))

        total_loss = structure_rl_loss + content_rl_loss + entropy_loss + step_probability_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (iter_idx + 1) % imp_per_iters == 0 and (graph.optimized_spatial or graph.optimized_temporal):
            graph.update_masks(pruning_rate)
            active_spatial_edges = _active_spatial_edge_count(graph)
            active_temporal_edges = _active_temporal_edge_count(graph)
            print(
                "[JointTrain][Prune] "
                f"iter={iter_idx + 1}/{iterations} "
                f"active_spatial_edges={active_spatial_edges} "
                f"active_temporal_edges={active_temporal_edges}"
            )

        total_actions = answer_actions + reasoning_actions
        answer_ratio = answer_actions / total_actions if total_actions else 0.0
        spatial_actions = sum(
            graph_counts
            for sample_info in sample_infos
            for key, graph_counts in sample_info["action_counts"].items()
            if key in {"spatial_answer", "spatial_reasoning"}
        )
        temporal_actions = sum(
            graph_counts
            for sample_info in sample_infos
            for key, graph_counts in sample_info["action_counts"].items()
            if key in {"temporal_answer", "temporal_reasoning"}
        )
        print(
            "[JointTrain][Iter] "
            f"{iter_idx + 1}/{iterations} time={time.time() - start_ts:.3f}s "
            f"lambda={lambda_t:.4f} step_lambda={step_lambda_t:.4f} "
            f"step_prob_lambda={step_probability_lambda_t:.4f} beta={beta_t:.4f} "
            f"acc={sum(correct_flags) / len(correct_flags):.3f} "
            f"reward_mean={sum(rewards) / len(rewards):.4f} "
            f"reward_baseline={float(reward_baseline):.4f} "
            f"content_baseline={float(content_baseline_ema) if content_baseline_ema is not None else 0.0:.4f} "
            f"tokens_mean={sum(token_totals) / len(token_totals):.1f} "
            f"step_tokens_mean={sum(info['step_tokens'] for info in sample_infos) / len(sample_infos):.1f} "
            f"answer_ratio={answer_ratio:.3f} "
            f"step_probability_mean={step_probability_mean:.3f} "
            f"spatial_actions={spatial_actions} temporal_actions={temporal_actions} "
            f"loss={float(total_loss.detach()):.4f} "
            f"structure_rl_loss={float(structure_rl_loss.detach()):.4f} "
            f"content_rl_loss={float(content_rl_loss.detach()):.4f} "
            f"entropy_loss={float(entropy_loss.detach()):.4f} "
            f"step_probability_loss={float(step_probability_loss.detach()):.4f}"
        )

    graph.edge_content_policy = policy
    graph.edge_message_mode = "rl_policy" if use_content_policy else "full"
    graph.edge_policy_training = False
    graph.set_eval_mode(use_content_policy)
    if policy is not None:
        policy.eval()

    print(
        "[JointTrain][Result] "
        f"active_spatial_edges={_active_spatial_edge_count(graph)} "
        f"active_temporal_edges={_active_temporal_edge_count(graph)}"
    )
    if policy is not None:
        _print_final_edge_decisions(graph, policy)

    if config.get("save_path"):
        print("[JointTrain] save_path is ignored: final edge decisions are printed, no .pt is written.")
    else:
        print("[JointTrain] Policy checkpoint save skipped: final edge decisions are printed above.")

    global_prompt_used = PromptTokens.instance().value - global_prompt_start
    global_completion_used = CompletionTokens.instance().value - global_completion_start
    print(
        "[JointTrain] Training complete. "
        f"Prompt Tokens: {global_prompt_used}, "
        f"Completion Tokens: {global_completion_used}, "
        f"Total Tokens: {global_prompt_used + global_completion_used}"
    )
    return policy


train_edge_content_policy = train_joint_policy
