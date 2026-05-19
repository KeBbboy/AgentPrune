import asyncio
import copy
import time
from typing import Any, Dict, Iterator, List

import numpy as np
import torch

from AgentPrune.graph.graph import Graph
from dataset.hotpotqa_dataset import HotpotQADataset


async def train_hotpotqa(
    graph: Graph,
    dataset: HotpotQADataset,
    num_iters: int,
    num_rounds: int,
    lr: float,
    batch_size: int,
    imp_per_iters: int,
    pruning_rate: float,
    info_setting: str,
) -> None:
    def infinite_data_loader() -> Iterator[Dict[str, Any]]:
        perm = np.random.permutation(len(dataset))
        while True:
            for idx in perm:
                yield dataset[idx.item()]

    loader = infinite_data_loader()
    optimizer_params = [graph.spatial_logits, graph.temporal_logits]
    if graph.enable_spatial_message_type_action or graph.enable_temporal_message_type_action:
        optimizer_params.extend([graph.spatial_message_type_logits, graph.temporal_message_type_logits])
    optimizer = torch.optim.Adam(optimizer_params, lr=lr)
    agent_count = len(graph.nodes)

    for i_iter in range(num_iters):
        print(f"Iter {i_iter}", 80 * "-")
        start_ts = time.time()
        answer_tasks = []
        correct_answers = []
        records = []

        for _, record in zip(range(batch_size), loader):
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.spatial_message_type_logits = graph.spatial_message_type_logits
            realized_graph.temporal_logits = graph.temporal_logits
            realized_graph.temporal_message_type_logits = graph.temporal_message_type_logits
            realized_graph.share_cost_state(graph)
            input_dict = dataset.record_to_input(
                record,
                agent_count=agent_count,
                info_setting=info_setting,
            )
            answer_tasks.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds)))
            correct_answers.append(dataset.record_to_target_answer(record))
            records.append(record)

        raw_results = await asyncio.gather(*answer_tasks)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        answers: List[str] = []

        for raw_answer, log_prob, correct_answer, record in zip(raw_answers, log_probs, correct_answers, records):
            answer = dataset.postprocess_answer(raw_answer, record=record)
            answers.append(answer)
            if correct_answer is None:
                utility = 0.0
            else:
                utility = HotpotQADataset.answer_metrics(answer, correct_answer)["f1"]
            utilities.append(utility)
            loss_list.append(-log_prob * utility)
            print(f"correct answer:{correct_answer}")

        total_loss = torch.mean(torch.stack(loss_list))
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print("raw_answers:", raw_answers)
        print("answers:", answers)
        print(f"Batch time {time.time() - start_ts:.3f}")
        print("utilities:", utilities)

        if (i_iter + 1) % imp_per_iters == 0:
            graph.update_masks(pruning_rate)
