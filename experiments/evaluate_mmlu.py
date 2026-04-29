import os
import json
import math
import time
import asyncio
from collections import Counter, defaultdict
from typing import Union,Literal,Optional,Iterator,List,Any,Dict
from tqdm import tqdm
import copy

from AgentPrune.graph.graph import Graph
from experiments.accuracy import Accuracy
from AgentPrune.utils.globals import Cost, PromptTokens, CompletionTokens


def summarize_pruned_delivery_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    def summarize_subset(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        delivered = [event for event in subset if event["delivered"]]
        missing = [event for event in subset if not event["delivered"]]
        missing_by_reason = Counter(event["failure_reason"] or "unknown" for event in missing)
        delivered_char_sum = sum(event.get("delivered_char_len", 0) for event in delivered)
        delivered_word_sum = sum(event.get("delivered_word_len", 0) for event in delivered)
        return {
            "total_candidates": len(subset),
            "delivered_count": len(delivered),
            "missing_count": len(missing),
            "delivered_rate": (len(delivered) / len(subset)) if subset else 0.0,
            "avg_delivered_char_len": (delivered_char_sum / len(delivered)) if delivered else 0.0,
            "avg_delivered_word_len": (delivered_word_sum / len(delivered)) if delivered else 0.0,
            "missing_by_reason": dict(sorted(missing_by_reason.items())),
        }

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    missing_edge_counter: Counter = Counter()
    delivered_edge_counter: Counter = Counter()
    missing_target_counter: Counter = Counter()
    missing_round_counter: Counter = Counter()

    for event in events:
        by_type[event["edge_type"]].append(event)
        edge_key = (
            event["edge_type"],
            event["source_role"],
            event["source_id"],
            event["target_role"],
            event["target_id"],
        )
        if event["delivered"]:
            delivered_edge_counter[edge_key] += 1
        else:
            missing_edge_counter[edge_key] += 1
            missing_target_counter[(event["target_role"], event["target_id"])] += 1
            round_label = event.get("round_label")
            if round_label is None and event.get("round_index") is not None:
                round_label = f"round_{event['round_index'] + 1}"
            missing_round_counter[round_label or "unknown"] += 1

    return {
        "total_events": len(events),
        "by_edge_type": {
            edge_type: summarize_subset(subset)
            for edge_type, subset in sorted(by_type.items())
        },
        "overall": summarize_subset(events),
        "top_missing_edges": [
            {
                "edge_type": edge_type,
                "source_role": source_role,
                "source_id": source_id,
                "target_role": target_role,
                "target_id": target_id,
                "missing_count": count,
            }
            for (edge_type, source_role, source_id, target_role, target_id), count in missing_edge_counter.most_common(20)
        ],
        "top_delivered_edges": [
            {
                "edge_type": edge_type,
                "source_role": source_role,
                "source_id": source_id,
                "target_role": target_role,
                "target_id": target_id,
                "delivered_count": count,
            }
            for (edge_type, source_role, source_id, target_role, target_id), count in delivered_edge_counter.most_common(20)
        ],
        "top_missing_targets": [
            {
                "target_role": target_role,
                "target_id": target_id,
                "missing_count": count,
            }
            for (target_role, target_id), count in missing_target_counter.most_common(20)
        ],
        "missing_by_round": {
            str(round_label): count
            for round_label, count in sorted(missing_round_counter.items(), key=lambda item: (-item[1], item[0]))
        },
    }


def print_pruned_delivery_summary(summary: Dict[str, Any]) -> None:
    print("Pruned Delivery Audit Summary:")
    overall = summary["overall"]
    print(
        f"  Overall: candidates={overall['total_candidates']}, delivered={overall['delivered_count']}, "
        f"missing={overall['missing_count']}, delivered_rate={overall['delivered_rate']:.4f}"
    )
    for edge_type, edge_summary in summary["by_edge_type"].items():
        print(
            f"  {edge_type}: candidates={edge_summary['total_candidates']}, "
            f"delivered={edge_summary['delivered_count']}, missing={edge_summary['missing_count']}, "
            f"delivered_rate={edge_summary['delivered_rate']:.4f}, "
            f"avg_delivered_word_len={edge_summary['avg_delivered_word_len']:.2f}"
        )
        if edge_summary["missing_by_reason"]:
            print(f"    missing_by_reason={edge_summary['missing_by_reason']}")
    if summary["top_missing_edges"]:
        print("  Top Missing Edges:")
        for item in summary["top_missing_edges"][:10]:
            print(
                f"    {item['edge_type']}: {item['source_role']}({item['source_id']}) -> "
                f"{item['target_role']}({item['target_id']}), missing={item['missing_count']}"
            )

async def evaluate(
        graph:Graph,
        dataset,
        num_rounds:int = 1,
        limit_questions: Optional[int] = None,
        eval_batch_size: int = 4,
        audit_pruned_delivery: bool = False,
        audit_output_path: Optional[str] = None,
        ) -> float:

    print(f"Evaluating AgentPrune on {dataset.__class__.__name__} split {dataset.split}")
    
    graph.spatial_logits.requires_grad_ = False
    graph.temporal_logits.requires_grad_ = False
    
    accuracy = Accuracy()
    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for i_record, record in enumerate(dataset):
            if limit_questions is not None:
                if i_record >= limit_questions:
                    break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if len(records) > 0:
            yield records
        return
    data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)
    num_batches = int(math.ceil(data_len / eval_batch_size))
    all_audit_events: List[Dict[str, Any]] = []

    for i_batch, record_batch in tqdm(enumerate(eval_loader(batch_size=eval_batch_size)), total=num_batches):
        print(80*'-')

        start_ts = time.time()
        answer_log_probs = []
        realized_graphs = []
        
        for record_offset, record in enumerate(record_batch):
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits
            realized_graph.share_cost_state(graph)
            realized_graph.current_sample_index = i_batch * eval_batch_size + record_offset + 1
            input_dict = dataset.record_to_input(record)
            # print(input_dict)
            realized_graphs.append(realized_graph)
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,num_rounds)))
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        if audit_pruned_delivery:
            for realized_graph in realized_graphs:
                all_audit_events.extend(realized_graph.get_pruned_delivery_audit_events())
        print(f"Batch time {time.time() - start_ts:.3f}")
        for raw_answer, record in zip(raw_answers, record_batch):
            print("Raw answer:", raw_answer)
            answer = dataset.postprocess_answer(raw_answer)
            print("Postprocessed answer:", answer)
            correct_answer = dataset.record_to_target_answer(record)
            print("Correct answer:", correct_answer)
            accuracy.update(answer, correct_answer)
            accuracy.print()
    accuracy.print()
    print("Done!")

    if audit_pruned_delivery:
        summary = summarize_pruned_delivery_events(all_audit_events)
        graph.last_pruned_delivery_report = {
            "summary": summary,
            "events": all_audit_events,
        }
        print_pruned_delivery_summary(summary)
        if audit_output_path is not None:
            output_dir = os.path.dirname(audit_output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(audit_output_path, "w", encoding="utf-8") as f:
                json.dump(graph.last_pruned_delivery_report, f, ensure_ascii=False, indent=2)
            print(f"Pruned delivery audit written to {audit_output_path}")

    return accuracy.get()


def dump_eval_results(self, dct: Dict[str, Any]) -> None:
    if self._art_dir_name is not None:
        eval_json_name = os.path.join(self._art_dir_name, "evaluation.json")
        with open(eval_json_name, "w") as f:
            json.dump(dct, f)
