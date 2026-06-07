from __future__ import annotations

import math
import re
import threading
from typing import Any

import numpy as np
import torch


DEFAULT_ROLE_AWARE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ROLE_AWARE_RATE = 0.5
DEFAULT_DEDUP_THRESHOLD = 0.88
DEFAULT_NODE_AWARE_TOTAL_BUDGET = 10
DEFAULT_NODE_AWARE_RATE = 1.0
DEFAULT_NODE_AWARE_SOURCE_WEIGHT = 0.3
_EMBEDDER_CACHE = {}
_EMBEDDER_LOCK = threading.Lock()


class RoleAwareEdgeCompressor:
    """Receiver-aware sentence selector for edge-level message compression."""

    def __init__(self):
        self.model_name = DEFAULT_ROLE_AWARE_MODEL
        self.rate = DEFAULT_ROLE_AWARE_RATE
        self.sentence_budget = 0
        self.dedup_threshold = DEFAULT_DEDUP_THRESHOLD
        self.min_sentences = 2
        self.preserve_answer_line = True
        self.compress_code = False
        self.num_seen = 0
        self.num_compressed = 0
        self.original_sentences = 0
        self.compressed_sentences = 0
        self.last_sentence_scores: list[tuple[str, float]] = []
        self._embedder = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_embedder"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def compress(
        self,
        text: Any,
        domain: str = "",
        role: str = "",
        receiver_role: str = "",
        receiver_description: str = "",
        receiver_agent: str = "",
        message_type: str = "full",
        source: str = "spatial",
    ) -> Any:
        if not isinstance(text, str):
            return text

        stripped_text = text.strip()
        if not stripped_text:
            return text
        if self._is_code_payload(stripped_text, domain):
            return text
        if source == "temporal":
            return text

        self.num_seen += 1
        answer_line = None
        compressible_text = stripped_text
        if self.preserve_answer_line:
            answer_line, compressible_text = self._split_leading_answer_line(stripped_text)

        sentences = self._split_sentences(compressible_text)
        if len(sentences) < self.min_sentences:
            self.last_sentence_scores = [(sentence, 1.0) for sentence in sentences]
            return text

        role_query = self._receiver_query(receiver_role, receiver_description, receiver_agent)
        if not role_query:
            return text

        sentence_scores, sentence_embeddings = self._score_sentences(sentences, role_query)
        self.last_sentence_scores = list(zip(sentences, sentence_scores))
        budget = self._sentence_budget(len(sentences))
        selected_indices = self._select_sentences(
            sentence_scores=sentence_scores,
            sentence_embeddings=sentence_embeddings,
            budget=budget,
        )
        if not selected_indices:
            return text

        selected_indices = sorted(selected_indices)
        selected_sentences = [sentences[idx] for idx in selected_indices]
        compressed = "\n".join(selected_sentences).strip()
        compressed = self._restore_leading_answer_line(compressed, answer_line)
        if not compressed or compressed == stripped_text:
            return text
        if len(selected_sentences) >= len(sentences):
            return text

        self.num_compressed += 1
        self.original_sentences += len(sentences)
        self.compressed_sentences += max(1, len(self._split_sentences(compressed)))
        return compressed

    def summary(self) -> str:
        if self.num_compressed == 0:
            summary = "Role-aware edge compressor compressed 0 messages."
        else:
            ratio = self.compressed_sentences / max(self.original_sentences, 1)
            summary = (
                f"Role-aware edge compressor compressed {self.num_compressed}/{self.num_seen} messages "
                f"from {self.original_sentences} to {self.compressed_sentences} sentences "
                f"(keep ratio {ratio:.3f})."
            )
        return summary

    def _load_embedder(self):
        if self._embedder is not None:
            return self._embedder
        if self.model_name in _EMBEDDER_CACHE:
            self._embedder = _EMBEDDER_CACHE[self.model_name]
            return self._embedder

        with _EMBEDDER_LOCK:
            if self.model_name in _EMBEDDER_CACHE:
                self._embedder = _EMBEDDER_CACHE[self.model_name]
                return self._embedder
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Role-aware compression is enabled, but `transformers` "
                    "is not installed. Install project requirements first."
                ) from exc

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
            self._embedder = (tokenizer, model, device)
            _EMBEDDER_CACHE[self.model_name] = self._embedder
            return self._embedder

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        tokenizer, model, device = self._load_embedder()
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).to(last_hidden_state.dtype)
        masked_hidden = last_hidden_state * attention_mask
        summed = masked_hidden.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        embeddings = torch.nn.functional.normalize(summed / counts, p=2, dim=1)
        return embeddings.cpu().numpy().astype(np.float32)

    def _score_sentences(self, sentences: list[str], role_query: str):
        embeddings = self._embed_texts([role_query] + sentences)
        role_embedding = embeddings[0]
        sentence_embeddings = embeddings[1:]
        scores = np.matmul(sentence_embeddings, role_embedding)
        return scores.astype(float).tolist(), sentence_embeddings

    def _select_sentences(
        self,
        sentence_scores: list[float],
        sentence_embeddings: np.ndarray,
        budget: int,
    ) -> list[int]:
        ranked_indices = sorted(
            range(len(sentence_scores)),
            key=lambda idx: sentence_scores[idx],
            reverse=True,
        )
        selected_indices: list[int] = []
        for idx in ranked_indices:
            if len(selected_indices) >= budget:
                break
            if self._is_duplicate(idx, selected_indices, sentence_embeddings):
                continue
            selected_indices.append(idx)
        return selected_indices

    def _is_duplicate(
        self,
        candidate_idx: int,
        selected_indices: list[int],
        sentence_embeddings: np.ndarray,
    ) -> bool:
        if not selected_indices:
            return False
        candidate = sentence_embeddings[candidate_idx]
        selected = sentence_embeddings[selected_indices]
        similarities = np.matmul(selected, candidate)
        return bool(np.max(similarities) > self.dedup_threshold)

    def _sentence_budget(self, sentence_count: int) -> int:
        if self.sentence_budget > 0:
            return min(self.sentence_budget, sentence_count)
        return min(max(1, math.ceil(sentence_count * self.rate)), sentence_count)

    def _receiver_query(
        self,
        receiver_role: str,
        receiver_description: str,
        receiver_agent: str,
    ) -> str:
        parts = []
        for value in (receiver_role, receiver_description, receiver_agent):
            if value and value not in parts:
                parts.append(str(value))
        return ". ".join(parts).strip()

    def _split_sentences(self, text: str) -> list[str]:
        sentences = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            line_sentences = re.findall(r"[^.!?\n]+(?:[.!?]+|$)", line)
            if line_sentences:
                sentences.extend(sentence.strip() for sentence in line_sentences if sentence.strip())
            else:
                sentences.append(line)
        return sentences

    def _split_leading_answer_line(self, text: str) -> tuple[str | None, str]:
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            match = re.match(r"^([A-D])(?:$|[\s).:：](.*)$)", stripped)
            if not match:
                break

            answer_line = match.group(1)
            first_line_remainder = (match.group(2) or "").lstrip(").:： ").strip()
            remaining_lines = lines[idx + 1:]
            compressible_lines = []
            if first_line_remainder:
                compressible_lines.append(first_line_remainder)
            compressible_lines.extend(remaining_lines)
            return answer_line, "\n".join(compressible_lines)
        return None, text

    def _restore_leading_answer_line(self, text: str, answer_line: str | None) -> str:
        if answer_line is None:
            return text
        return f"{answer_line}\n{text}" if text else answer_line

    def _approx_word_count(self, text: str) -> int:
        return len(re.findall(r"\S+", text))

    def _is_code_payload(self, text: str, domain: str) -> bool:
        if self.compress_code:
            return False
        if domain != "humaneval":
            return False
        return (
            "```" in text
            or "<answer>" in text.lower()
            or bool(re.search(r"^\s*(def|class|import|from)\s+", text, flags=re.MULTILINE))
        )

class RoleAwareNodeAggregator(RoleAwareEdgeCompressor):
    """Receiver-side spatial message aggregator with cross-sender deduplication."""

    def __init__(self):
        super().__init__()
        self.rate = DEFAULT_NODE_AWARE_RATE
        self.min_sentences = 1
        self.total_budget = DEFAULT_NODE_AWARE_TOTAL_BUDGET
        self.source_weight = DEFAULT_NODE_AWARE_SOURCE_WEIGHT
        self.num_aggregated = 0

    def aggregate(
        self,
        spatial_info: dict[str, dict],
        domain: str = "",
        receiver_role: str = "",
        receiver_description: str = "",
        receiver_agent: str = "",
    ) -> dict[str, dict]:
        if len(spatial_info) <= 1:
            return spatial_info

        role_query = self._receiver_query(receiver_role, receiver_description, receiver_agent)
        if not role_query:
            return spatial_info

        message_records = []
        sentence_records = []
        all_sentences = []
        for sender_id, info in spatial_info.items():
            output = info.get("output")
            if not isinstance(output, str) or not output.strip():
                continue
            if self._is_code_payload(output.strip(), domain):
                continue

            answer_line, compressible_text = self._split_leading_answer_line(output.strip())
            sentences = self._split_sentences(compressible_text)
            if not sentences:
                continue

            record_idx = len(message_records)
            message_records.append(
                {
                    "sender_id": sender_id,
                    "answer_line": answer_line,
                    "sentences": sentences,
                    "source_score": 0.0,
                }
            )
            for local_idx, sentence in enumerate(sentences):
                sentence_records.append(
                    {
                        "sender_id": sender_id,
                        "record_idx": record_idx,
                        "local_idx": local_idx,
                        "sentence": sentence,
                    }
                )
                all_sentences.append(sentence)

        if len(sentence_records) <= 1:
            return spatial_info

        sentence_scores, sentence_embeddings = self._score_sentences(all_sentences, role_query)
        for idx, record in enumerate(sentence_records):
            record["sentence_score"] = sentence_scores[idx]

        for record_idx, message_record in enumerate(message_records):
            scores = [
                record["sentence_score"]
                for record in sentence_records
                if record["record_idx"] == record_idx
            ]
            message_record["source_score"] = float(np.mean(scores)) if scores else 0.0

        for record in sentence_records:
            source_score = message_records[record["record_idx"]]["source_score"]
            sentence_score = record["sentence_score"]
            record["global_score"] = (
                (1.0 - self.source_weight) * sentence_score
                + self.source_weight * source_score
            )

        budget = self._aggregation_budget(len(sentence_records))
        ranked_indices = sorted(
            range(len(sentence_records)),
            key=lambda idx: sentence_records[idx]["global_score"],
            reverse=True,
        )
        selected_indices = []
        for idx in ranked_indices:
            if len(selected_indices) >= budget:
                break
            if self._is_duplicate(idx, selected_indices, sentence_embeddings):
                continue
            selected_indices.append(idx)

        if not selected_indices:
            return spatial_info

        selected_by_sender: dict[str, list[int]] = {}
        for idx in selected_indices:
            sender_id = sentence_records[idx]["sender_id"]
            selected_by_sender.setdefault(sender_id, []).append(idx)

        aggregated_info = {}
        changed = False
        for sender_id, info in spatial_info.items():
            selected = selected_by_sender.get(sender_id)
            if selected is None:
                output = info.get("output")
                if isinstance(output, str) and output.strip():
                    changed = True
                    continue
                aggregated_info[sender_id] = info
                continue

            selected = sorted(selected, key=lambda idx: sentence_records[idx]["local_idx"])
            record_idx = sentence_records[selected[0]]["record_idx"]
            answer_line = message_records[record_idx]["answer_line"]
            selected_sentences = [sentence_records[idx]["sentence"] for idx in selected]
            new_output = "\n".join(selected_sentences).strip()
            new_output = self._restore_leading_answer_line(new_output, answer_line)
            new_info = dict(info)
            new_info["output"] = new_output
            new_info["aggregation_score"] = message_records[record_idx]["source_score"]
            aggregated_info[sender_id] = new_info
            changed |= new_output != info.get("output")

        if not changed:
            return spatial_info

        self.num_aggregated += 1
        self.original_sentences += len(sentence_records)
        self.compressed_sentences += len(selected_indices)
        return aggregated_info

    def summary(self) -> str:
        if self.num_aggregated == 0:
            return "Role-aware node aggregator aggregated 0 spatial contexts."
        ratio = self.compressed_sentences / max(self.original_sentences, 1)
        return (
            f"Role-aware node aggregator aggregated {self.num_aggregated} spatial contexts "
            f"from {self.original_sentences} to {self.compressed_sentences} sentences "
            f"(keep ratio {ratio:.3f})."
        )

    def _aggregation_budget(self, sentence_count: int) -> int:
        if self.total_budget > 0:
            return min(self.total_budget, sentence_count)
        return min(max(1, math.ceil(sentence_count * self.rate)), sentence_count)

def add_role_aware_args(parser):
    parser.add_argument(
        "--role_aware_compression",
        action="store_true",
        help="Compress inter-agent messages with receiver-role-aware sentence selection.",
    )
    parser.add_argument(
        "--node_aware_aggregation",
        action="store_true",
        help="Aggregate and deduplicate spatial messages at each receiver node.",
    )


def role_aware_graph_kwargs(args) -> dict[str, Any]:
    return {
        "role_aware_compression": getattr(args, "role_aware_compression", False),
        "node_aware_aggregation": getattr(args, "node_aware_aggregation", False),
    }
