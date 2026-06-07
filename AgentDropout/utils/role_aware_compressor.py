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
DEFAULT_QWEN_REWRITE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_QWEN_REWRITE_MAX_NEW_TOKENS = 128
TEMPORAL_FIELD_PROTOTYPES = {
    "key_reason": (
        "important reason",
        "key clue",
        "main rationale",
        "decisive evidence",
        "core reasoning",
    ),
    "evidence": (
        "supporting fact",
        "computed result",
        "observation",
        "given condition",
        "calculation detail",
    ),
    "disagreement": (
        "wrong alternative",
        "contradiction",
        "rejected option",
        "incorrect answer",
        "opposing conclusion",
    ),
    "uncertainty": (
        "uncertain conclusion",
        "not sure",
        "ambiguous evidence",
        "cannot determine",
        "possible limitation",
    ),
}
TEMPORAL_FIELD_BUDGETS = {
    "key_reason": 1,
    "evidence": 1,
    "disagreement": 1,
    "uncertainty": 1,
}
_EMBEDDER_CACHE = {}
_EMBEDDER_LOCK = threading.Lock()
_REWRITER_CACHE = {}
_REWRITER_LOCK = threading.Lock()


class RoleAwareEdgeCompressor:
    """Receiver-aware sentence selector for edge-level message compression."""

    def __init__(
        self,
        model_name: str = DEFAULT_ROLE_AWARE_MODEL,
        rate: float = DEFAULT_ROLE_AWARE_RATE,
        sentence_budget: int = 0,
        dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
        min_sentences: int = 2,
        preserve_answer_line: bool = True,
        compress_code: bool = False,
        rewrite_with_qwen: bool = False,
        rewrite_model_name: str = DEFAULT_QWEN_REWRITE_MODEL,
        rewrite_max_new_tokens: int = DEFAULT_QWEN_REWRITE_MAX_NEW_TOKENS,
        debug: bool = False,
    ):
        if not 0 < float(rate) <= 1:
            raise ValueError("Role-aware compression rate must be in (0, 1].")
        if not 0 <= float(dedup_threshold) <= 1:
            raise ValueError("Role-aware dedup threshold must be in [0, 1].")

        self.model_name = model_name
        self.rate = float(rate)
        self.sentence_budget = max(0, int(sentence_budget))
        self.dedup_threshold = float(dedup_threshold)
        self.min_sentences = max(1, int(min_sentences))
        self.preserve_answer_line = preserve_answer_line
        self.compress_code = compress_code
        self.rewrite_with_qwen = bool(rewrite_with_qwen)
        self.rewrite_model_name = rewrite_model_name or DEFAULT_QWEN_REWRITE_MODEL
        self.rewrite_max_new_tokens = max(1, int(rewrite_max_new_tokens))
        self.debug = debug
        self.num_seen = 0
        self.num_compressed = 0
        self.num_rewritten = 0
        self.original_sentences = 0
        self.compressed_sentences = 0
        self.last_sentence_scores: list[tuple[str, float]] = []
        self._embedder = None
        self._rewriter = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_embedder"] = None
        state["_rewriter"] = None
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

        compressed = self._maybe_rewrite(
            compressed,
            original_text=stripped_text,
            domain=domain,
            sender_role=role,
            receiver_role=receiver_role,
            receiver_description=receiver_description,
            receiver_agent=receiver_agent,
            message_type=message_type,
            source=source,
            answer_line=answer_line,
        )

        self.num_compressed += 1
        self.original_sentences += len(sentences)
        self.compressed_sentences += max(1, len(self._split_sentences(compressed)))
        if self.debug:
            self._print_debug(
                sender_role=role,
                receiver=role_query,
                source=source,
                message_type=message_type,
                sentence_scores=sentence_scores,
                sentences=sentences,
                selected_indices=selected_indices,
            )
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
        if self.rewrite_with_qwen:
            summary += (
                f" Qwen rewrite accepted {self.num_rewritten} messages "
                f"with {self.rewrite_model_name}."
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

    def _load_rewriter(self):
        if self._rewriter is not None:
            return self._rewriter
        if self.rewrite_model_name in _REWRITER_CACHE:
            self._rewriter = _REWRITER_CACHE[self.rewrite_model_name]
            return self._rewriter

        with _REWRITER_LOCK:
            if self.rewrite_model_name in _REWRITER_CACHE:
                self._rewriter = _REWRITER_CACHE[self.rewrite_model_name]
                return self._rewriter
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Qwen rewrite is enabled, but `transformers` is not installed. "
                    "Install project requirements first."
                ) from exc

            tokenizer = AutoTokenizer.from_pretrained(
                self.rewrite_model_name,
                trust_remote_code=True,
            )
            load_kwargs = {"trust_remote_code": True}
            if torch.cuda.is_available():
                load_kwargs["torch_dtype"] = torch.float16
            model = AutoModelForCausalLM.from_pretrained(
                self.rewrite_model_name,
                **load_kwargs,
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
            self._rewriter = (tokenizer, model, device)
            _REWRITER_CACHE[self.rewrite_model_name] = self._rewriter
            return self._rewriter

    def _maybe_rewrite(
        self,
        compressed: str,
        original_text: str,
        domain: str,
        sender_role: str,
        receiver_role: str,
        receiver_description: str,
        receiver_agent: str,
        message_type: str,
        source: str,
        answer_line: str | None = None,
    ) -> str:
        if not self.rewrite_with_qwen:
            return compressed
        if not compressed.strip() or self._is_code_payload(compressed, domain):
            return compressed

        try:
            rewritten = self._rewrite_with_qwen(
                compressed=compressed,
                original_text=original_text,
                sender_role=sender_role,
                receiver_role=receiver_role,
                receiver_description=receiver_description,
                receiver_agent=receiver_agent,
                message_type=message_type,
                source=source,
            )
        except Exception as exc:
            if self.debug:
                print(f"[RoleAwareRewrite] skipped after generation error: {exc}")
            return compressed

        rewritten = self._validate_rewrite(
            rewritten=rewritten,
            compressed=compressed,
            answer_line=answer_line,
        )
        if rewritten is None:
            return compressed

        self.num_rewritten += 1
        return rewritten

    def _rewrite_with_qwen(
        self,
        compressed: str,
        original_text: str,
        sender_role: str,
        receiver_role: str,
        receiver_description: str,
        receiver_agent: str,
        message_type: str,
        source: str,
    ) -> str:
        tokenizer, model, device = self._load_rewriter()
        prompt = self._rewrite_prompt(
            compressed=compressed,
            original_text=original_text,
            sender_role=sender_role,
            receiver_role=receiver_role,
            receiver_description=receiver_description,
            receiver_agent=receiver_agent,
            message_type=message_type,
            source=source,
        )
        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = "\n".join(message["content"] for message in prompt)

        encoded = tokenizer(prompt_text, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_length = encoded["input_ids"].shape[-1]
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=self.rewrite_max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        generated_tokens = generated[0][input_length:]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def _rewrite_prompt(
        self,
        compressed: str,
        original_text: str,
        sender_role: str,
        receiver_role: str,
        receiver_description: str,
        receiver_agent: str,
        message_type: str,
        source: str,
    ) -> list[dict[str, str]]:
        receiver = self._receiver_query(receiver_role, receiver_description, receiver_agent)
        user_parts = [
            f"Sender role: {sender_role or 'unknown'}",
            f"Receiver role: {receiver or 'unknown'}",
            f"Source: {source}",
            f"Message type: {message_type}",
            "Compressed message to rewrite:",
            compressed,
        ]
        if original_text != compressed:
            user_parts.extend(["Original message for reference:", original_text])
        return [
            {
                "role": "system",
                "content": (
                    "Rewrite inter-agent reasoning messages into a shorter, clear note. "
                    "Preserve final answers, option letters, numbers, equations, code, "
                    "and structured labels exactly. Drop filler. Return only the rewritten note."
                ),
            },
            {"role": "user", "content": "\n".join(user_parts)},
        ]

    def _validate_rewrite(
        self,
        rewritten: str,
        compressed: str,
        answer_line: str | None,
    ) -> str | None:
        rewritten = self._strip_rewrite_output(rewritten)
        if not rewritten:
            return None

        if answer_line is not None:
            _, rewritten_body = self._split_leading_answer_line(rewritten)
            rewritten = self._restore_leading_answer_line(rewritten_body.strip(), answer_line)

        rewritten = self._restore_memory_card_constraints(
            rewritten=rewritten,
            compressed=compressed,
        )
        if rewritten == compressed:
            return None
        if self._approx_word_count(rewritten) >= self._approx_word_count(compressed):
            return None
        return rewritten

    def _strip_rewrite_output(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:text)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        for prefix in ("rewritten note:", "rewritten message:", "rewrite:"):
            if text.lower().startswith(prefix):
                return text[len(prefix):].strip()
        return text

    def _restore_memory_card_constraints(self, rewritten: str, compressed: str) -> str:
        if not compressed.lstrip().startswith("[MemoryCard]"):
            return rewritten

        answer_match = re.search(r"^answer:\s*(.+)$", compressed, flags=re.MULTILINE)
        if not rewritten.lstrip().startswith("[MemoryCard]"):
            rewritten = "[MemoryCard]\n" + rewritten.lstrip()
        if answer_match is None:
            return rewritten

        answer_line = f"answer: {answer_match.group(1).strip()}"
        if re.search(rf"^{re.escape(answer_line)}$", rewritten, flags=re.MULTILINE):
            return rewritten

        lines = rewritten.splitlines()
        if lines:
            lines.insert(1, answer_line)
            return "\n".join(lines).strip()
        return f"[MemoryCard]\n{answer_line}"

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

    def _compress_temporal_memory_card(
        self,
        text: str,
        sender_role: str,
        receiver_role: str,
        receiver_description: str,
        receiver_agent: str,
        domain: str,
        message_type: str,
    ) -> str | None:
        answer_line, compressible_text = self._split_leading_answer_line(text)
        answer = self._extract_temporal_answer(text, answer_line, domain)
        sentences = self._split_sentences(compressible_text)
        if not answer and not sentences:
            return None

        role_query = self._receiver_query(receiver_role, receiver_description, receiver_agent)
        fields = self._extract_temporal_fields(sentences, role_query)
        card = self._format_temporal_memory_card(
            answer=answer,
            fields=fields,
            sender_role=sender_role,
        )
        if not card:
            return None
        if self._approx_word_count(card) >= self._approx_word_count(text):
            return None

        card = self._maybe_rewrite(
            card,
            original_text=text,
            domain=domain,
            sender_role=sender_role,
            receiver_role=receiver_role,
            receiver_description=receiver_description,
            receiver_agent=receiver_agent,
            message_type=message_type,
            source="temporal",
        )

        self.num_compressed += 1
        self.original_sentences += max(1, len(sentences))
        self.compressed_sentences += max(1, len(card.splitlines()))
        return card

    def _extract_temporal_answer(
        self,
        text: str,
        answer_line: str | None,
        domain: str,
    ) -> str:
        if answer_line:
            return answer_line

        if domain in {"mmlu", "mmlu_pro", ""}:
            option_match = re.search(
                r"\b(?:final\s+answer|answer|answer\s+is)\s*(?:is|:)?\s*([A-D])\b",
                text,
                flags=re.IGNORECASE,
            )
            if option_match is not None:
                return option_match.group(1).upper()

        numeric_match = re.search(
            r"\b(?:final\s+answer|answer|answer\s+is)\s*(?:is|:)?\s*(-?\d+(?:\.\d+)?)\b",
            text,
            flags=re.IGNORECASE,
        )
        if numeric_match is not None:
            return numeric_match.group(1)
        return ""

    def _extract_temporal_fields(
        self,
        sentences: list[str],
        role_query: str,
    ) -> dict[str, list[str]]:
        fields: dict[str, list[tuple[str, float]]] = {
            field_name: [] for field_name in TEMPORAL_FIELD_BUDGETS
        }
        if not sentences:
            return {field_name: [] for field_name in TEMPORAL_FIELD_BUDGETS}

        role_scores = [0.0] * len(sentences)
        field_scores = self._score_temporal_fields(sentences)
        if role_query:
            role_scores, _ = self._score_sentences(sentences, role_query)

        for idx, sentence in enumerate(sentences):
            forced_field = self._forced_temporal_field(sentence)
            if forced_field is not None:
                field_name = forced_field
                score = 1.0 + role_scores[idx]
            else:
                field_name, score = max(
                    field_scores[idx].items(),
                    key=lambda item: item[1],
                )
                score += 0.25 * role_scores[idx]
            fields[field_name].append((sentence, float(score)))

        selected_fields: dict[str, list[str]] = {}
        for field_name, candidates in fields.items():
            budget = TEMPORAL_FIELD_BUDGETS[field_name]
            ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
            selected_fields[field_name] = [sentence for sentence, _ in ranked[:budget]]
        return selected_fields

    def _score_temporal_fields(self, sentences: list[str]) -> list[dict[str, float]]:
        prototype_texts = []
        prototype_fields = []
        for field_name, prototypes in TEMPORAL_FIELD_PROTOTYPES.items():
            for prototype in prototypes:
                prototype_texts.append(prototype)
                prototype_fields.append(field_name)

        embeddings = self._embed_texts(sentences + prototype_texts)
        sentence_embeddings = embeddings[:len(sentences)]
        prototype_embeddings = embeddings[len(sentences):]
        similarities = np.matmul(sentence_embeddings, prototype_embeddings.T)

        field_scores = []
        for row in similarities:
            scores = {field_name: 0.0 for field_name in TEMPORAL_FIELD_PROTOTYPES}
            for score, field_name in zip(row, prototype_fields):
                scores[field_name] = max(scores[field_name], float(score))
            field_scores.append(scores)
        return field_scores

    def _forced_temporal_field(self, sentence: str) -> str | None:
        lower = sentence.lower()
        if re.search(r"\b(maybe|possibly|unclear|uncertain|not sure|cannot determine)\b", lower):
            return "uncertainty"
        disagreement_pattern = (
            r"\b(wrong|incorrect|reject|rejected|contradict|fails)\b"
            r"|\b(?:option\s+)?[A-D]\s+(?:is\s+)?not\b"
            r"|\bis\s+not\s+(?:correct|valid|true|the\s+answer)\b"
        )
        if re.search(disagreement_pattern, lower):
            return "disagreement"
        return None

    def _format_temporal_memory_card(
        self,
        answer: str,
        fields: dict[str, list[str]],
        sender_role: str,
    ) -> str:
        lines = ["[MemoryCard]"]
        if answer:
            lines.append(f"answer: {answer}")
        for field_name in ("key_reason", "evidence", "disagreement", "uncertainty"):
            values = fields.get(field_name, [])
            if not values:
                continue
            lines.append(f"{field_name}: {' '.join(values)}")
        if sender_role:
            lines.append(f"role: {sender_role}")
        return "\n".join(lines).strip() if len(lines) > 1 else ""

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

    def _print_debug(
        self,
        sender_role: str,
        receiver: str,
        source: str,
        message_type: str,
        sentence_scores: list[float],
        sentences: list[str],
        selected_indices: list[int],
    ) -> None:
        selected = set(selected_indices)
        print(
            f"[RoleAwareEdge] sender={sender_role} receiver={receiver} "
            f"source={source} message_type={message_type}"
        )
        for idx, (sentence, score) in enumerate(zip(sentences, sentence_scores)):
            mark = "*" if idx in selected else " "
            print(f"[RoleAwareEdge]{mark} score={score:.4f} sentence={sentence}")


class RoleAwareNodeAggregator(RoleAwareEdgeCompressor):
    """Receiver-side spatial message aggregator with cross-sender deduplication."""

    def __init__(
        self,
        model_name: str = DEFAULT_ROLE_AWARE_MODEL,
        rate: float = DEFAULT_NODE_AWARE_RATE,
        total_budget: int = DEFAULT_NODE_AWARE_TOTAL_BUDGET,
        dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
        source_weight: float = DEFAULT_NODE_AWARE_SOURCE_WEIGHT,
        debug: bool = False,
    ):
        super().__init__(
            model_name=model_name,
            rate=rate,
            sentence_budget=0,
            dedup_threshold=dedup_threshold,
            min_sentences=1,
            preserve_answer_line=True,
            compress_code=False,
            debug=debug,
        )
        if not 0 <= float(source_weight) <= 1:
            raise ValueError("Node aggregation source weight must be in [0, 1].")
        self.total_budget = max(0, int(total_budget))
        self.source_weight = float(source_weight)
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
        if self.debug:
            self._print_aggregation_debug(
                receiver=role_query,
                sentence_records=sentence_records,
                selected_indices=selected_indices,
            )
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

    def _print_aggregation_debug(
        self,
        receiver: str,
        sentence_records: list[dict],
        selected_indices: list[int],
    ) -> None:
        selected = set(selected_indices)
        print(f"[RoleAwareNode] receiver={receiver}")
        for idx, record in enumerate(sentence_records):
            mark = "*" if idx in selected else " "
            print(
                f"[RoleAwareNode]{mark} sender={record['sender_id']} "
                f"global={record['global_score']:.4f} sentence={record['sentence']}"
            )


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
    parser.add_argument(
        "--role_aware_qwen_rewrite",
        action="store_true",
        help="Rewrite role-aware compressed messages with a Qwen 0.5B model when the rewrite is shorter.",
    )
    parser.add_argument(
        "--role_aware_qwen_model_name",
        type=str,
        default=DEFAULT_QWEN_REWRITE_MODEL,
        help="HF model used for optional Qwen rewrite.",
    )
    parser.add_argument(
        "--role_aware_qwen_max_new_tokens",
        type=int,
        default=DEFAULT_QWEN_REWRITE_MAX_NEW_TOKENS,
        help="Maximum new tokens for optional Qwen rewrite.",
    )


def role_aware_graph_kwargs(args) -> dict[str, Any]:
    return {
        "role_aware_compression": getattr(args, "role_aware_compression", False),
        "role_aware_model_name": getattr(args, "role_aware_model_name", DEFAULT_ROLE_AWARE_MODEL),
        "role_aware_rate": getattr(args, "role_aware_rate", DEFAULT_ROLE_AWARE_RATE),
        "role_aware_sentence_budget": getattr(args, "role_aware_sentence_budget", 0),
        "role_aware_dedup_threshold": getattr(args, "role_aware_dedup_threshold", DEFAULT_DEDUP_THRESHOLD),
        "role_aware_min_sentences": getattr(args, "role_aware_min_sentences", 2),
        "role_aware_compress_code": getattr(args, "role_aware_compress_code", False),
        "role_aware_qwen_rewrite": getattr(args, "role_aware_qwen_rewrite", False),
        "role_aware_qwen_model_name": getattr(args, "role_aware_qwen_model_name", DEFAULT_QWEN_REWRITE_MODEL),
        "role_aware_qwen_max_new_tokens": getattr(
            args,
            "role_aware_qwen_max_new_tokens",
            DEFAULT_QWEN_REWRITE_MAX_NEW_TOKENS,
        ),
        "role_aware_debug": getattr(args, "role_aware_debug", False),
        "node_aware_aggregation": getattr(args, "node_aware_aggregation", False),
        "node_aware_model_name": getattr(args, "node_aware_model_name", DEFAULT_ROLE_AWARE_MODEL),
        "node_aware_rate": getattr(args, "node_aware_rate", DEFAULT_NODE_AWARE_RATE),
        "node_aware_total_budget": getattr(args, "node_aware_total_budget", DEFAULT_NODE_AWARE_TOTAL_BUDGET),
        "node_aware_dedup_threshold": getattr(args, "node_aware_dedup_threshold", DEFAULT_DEDUP_THRESHOLD),
        "node_aware_source_weight": getattr(args, "node_aware_source_weight", DEFAULT_NODE_AWARE_SOURCE_WEIGHT),
        "node_aware_debug": getattr(args, "node_aware_debug", False),
    }
