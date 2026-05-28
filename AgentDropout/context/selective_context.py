from __future__ import annotations

import os
import re
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterator, Protocol

from dotenv import load_dotenv


load_dotenv()


_TRUTHY = {"1", "true", "yes", "y", "on"}
_FALSEY = {"0", "false", "no", "n", "off"}
_RUNTIME_ENABLED: ContextVar[bool] = ContextVar("selective_context_runtime_enabled", default=True)


@contextmanager
def selective_context_disabled() -> Iterator[None]:
    token = _RUNTIME_ENABLED.set(False)
    try:
        yield
    finally:
        _RUNTIME_ENABLED.reset(token)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSEY:
        return False
    return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class SelectiveContextConfig:
    enabled: bool = True
    model_name: str = "Qwen/Qwen2.5-0.5B"
    reduce_ratio: float = 0.5
    granularity: str = "phrase"
    min_chars: int = 0
    min_units: int = 1
    max_length: int = 1024
    device: str = "auto"
    preserve_first_line: bool = True
    preserve_first_units: int = 0
    preserve_last_units: int = 0
    fail_open: bool = True
    local_files_only: bool = False
    debug: bool = False

    @classmethod
    def from_env(cls) -> "SelectiveContextConfig":
        return cls(
            enabled=_env_bool("SELECTIVE_CONTEXT_ENABLED", True),
            model_name=os.getenv("SELECTIVE_CONTEXT_MODEL", "Qwen/Qwen2.5-0.5B").strip() or "Qwen/Qwen2.5-0.5B",
            reduce_ratio=min(max(_env_float("SELECTIVE_CONTEXT_REDUCE_RATIO", 0.5), 0.0), 0.95),
            granularity=os.getenv("SELECTIVE_CONTEXT_GRANULARITY", "phrase").strip().lower() or "phrase",
            min_chars=max(0, _env_int("SELECTIVE_CONTEXT_MIN_CHARS", 0)),
            min_units=max(1, _env_int("SELECTIVE_CONTEXT_MIN_UNITS", 1)),
            max_length=max(2, _env_int("SELECTIVE_CONTEXT_MAX_LENGTH", 1024)),
            device=os.getenv("SELECTIVE_CONTEXT_DEVICE", "auto").strip().lower() or "auto",
            preserve_first_line=_env_bool("SELECTIVE_CONTEXT_PRESERVE_FIRST_LINE", True),
            preserve_first_units=max(0, _env_int("SELECTIVE_CONTEXT_PRESERVE_FIRST_UNITS", 0)),
            preserve_last_units=max(0, _env_int("SELECTIVE_CONTEXT_PRESERVE_LAST_UNITS", 0)),
            fail_open=_env_bool("SELECTIVE_CONTEXT_FAIL_OPEN", True),
            local_files_only=_env_bool("SELECTIVE_CONTEXT_LOCAL_FILES_ONLY", False),
            debug=_env_bool("SELECTIVE_CONTEXT_DEBUG", False),
        )


@dataclass(frozen=True)
class Unit:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class TokenScore:
    start: int
    end: int
    score: float


class SurprisalScorer(Protocol):
    def score(self, text: str) -> list[TokenScore]:
        ...


class TransformersSurprisalScorer:
    """Scores tokens by self-information with a local causal language model."""

    def __init__(self, config: SelectiveContextConfig):
        self.config = config
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = "cpu"

    def _load(self) -> tuple[Any, Any, Any]:
        if self._tokenizer is not None and self._model is not None and self._torch is not None:
            return self._tokenizer, self._model, self._torch

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True,
            local_files_only=self.config.local_files_only,
        )
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError(
                "Selective Context requires a fast tokenizer for character offsets. "
                f"Model {self.config.model_name!r} did not provide one."
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            local_files_only=self.config.local_files_only,
        )
        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device
        model.to(device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        self._device = device
        return tokenizer, model, torch

    def score(self, text: str) -> list[TokenScore]:
        tokenizer, model, torch = self._load()
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            truncation=False,
        )
        input_ids = encoded.get("input_ids", [])
        offsets = encoded.get("offset_mapping", [])
        if len(input_ids) <= 1:
            return []

        losses_by_index: list[float | None] = [None] * len(input_ids)
        chunk_start = 0
        while chunk_start < len(input_ids) - 1:
            chunk_end = min(chunk_start + self.config.max_length, len(input_ids))
            chunk_ids = input_ids[chunk_start:chunk_end]
            if len(chunk_ids) <= 1:
                break

            input_tensor = torch.tensor([chunk_ids], dtype=torch.long, device=self._device)
            with torch.no_grad():
                logits = model(input_tensor).logits[0, :-1, :]
                targets = input_tensor[0, 1:]
                losses = torch.nn.functional.cross_entropy(logits, targets, reduction="none")

            for local_index, loss in enumerate(losses.detach().cpu().tolist(), start=1):
                absolute_index = chunk_start + local_index
                if absolute_index < len(losses_by_index):
                    losses_by_index[absolute_index] = float(loss)

            if chunk_end >= len(input_ids):
                break
            chunk_start = max(chunk_end - 1, chunk_start + 1)

        token_scores: list[TokenScore] = []
        for index, loss in enumerate(losses_by_index):
            if loss is None or index >= len(offsets):
                continue
            start, end = offsets[index]
            if end <= start:
                continue
            token_scores.append(TokenScore(start=int(start), end=int(end), score=loss))
        return token_scores


class SelectiveContextCompressor:
    """Implements Selective Context for inter-agent messages.

    The compressor keeps lexical units with higher self-information and removes
    low-information units until the configured reduce ratio is reached.
    """

    _TOKEN_RE = re.compile(r"\s*\S+")
    _PHRASE_RE = re.compile(r"[^,;:.!?\n，；：。！？]+[,;:.!?，；：。！？]?\s*|\n+\s*")
    _SENTENCE_RE = re.compile(r"[^.!?\n。！？]+[.!?。！？]?\s*|\n+\s*")

    def __init__(
        self,
        config: SelectiveContextConfig | None = None,
        scorer: SurprisalScorer | None = None,
    ):
        self.config = config if config is not None else SelectiveContextConfig()
        self._scorer = scorer
        self._warned_failure = False

    @property
    def scorer(self) -> SurprisalScorer:
        if self._scorer is None:
            self._scorer = TransformersSurprisalScorer(self.config)
        return self._scorer

    def compress_payload(self, payload: Any) -> Any:
        if isinstance(payload, str):
            return self.compress(payload)
        if isinstance(payload, list):
            return [self.compress_payload(item) for item in payload]
        return payload

    def compress(self, text: str) -> str:
        if not _RUNTIME_ENABLED.get():
            return text
        if not self.config.enabled:
            return text
        if self.config.reduce_ratio <= 0:
            return text
        prefix, body = self._split_preserved_first_line(text)
        if not body:
            return text
        compressed_body = self._compress_body(
            body,
            score_text=text if prefix else body,
            unit_offset=len(prefix),
        )
        if compressed_body == body:
            return text
        compressed = f"{prefix}{compressed_body}"
        if self.config.debug:
            print(
                "[SelectiveContext] compressed "
                f"chars={len(text)}->{len(compressed)} "
                f"preserve_first_line={self.config.preserve_first_line}",
                flush=True,
            )
        return compressed

    def _compress_body(self, text: str, score_text: str | None = None, unit_offset: int = 0) -> str:
        if len(text.strip()) < self.config.min_chars:
            return text

        units = self._split_units(text)
        content_indices = [idx for idx, unit in enumerate(units) if unit.text.strip()]
        if len(content_indices) <= self.config.min_units:
            return text

        try:
            scores = self._score_units(score_text if score_text is not None else text, units, unit_offset)
        except Exception as exc:
            if not self.config.fail_open:
                raise
            if not self._warned_failure:
                warnings.warn(
                    "Selective Context scoring failed; using uncompressed context. "
                    f"Reason: {exc}",
                    RuntimeWarning,
                )
                self._warned_failure = True
            return text

        keep_count = self._target_keep_count(len(content_indices))
        selected_indices = self._select_units(content_indices, scores, keep_count)
        compressed = "".join(
            unit.text for idx, unit in enumerate(units) if idx in selected_indices
        ).strip()
        return compressed if compressed else text

    def _split_preserved_first_line(self, text: str) -> tuple[str, str]:
        if not self.config.preserve_first_line:
            return "", text

        first_newline_index = text.find("\n")
        if first_newline_index < 0:
            return text, ""
        split_index = first_newline_index + 1
        return text[:split_index], text[split_index:]

    def _target_keep_count(self, unit_count: int) -> int:
        drop_count = int(round(unit_count * self.config.reduce_ratio))
        return max(1, unit_count - drop_count)

    def _select_units(
        self,
        content_indices: list[int],
        scores: dict[int, float],
        keep_count: int,
    ) -> set[int]:
        protected: set[int] = set()
        if self.config.preserve_first_units:
            protected.update(content_indices[: self.config.preserve_first_units])
        if self.config.preserve_last_units:
            protected.update(content_indices[-self.config.preserve_last_units :])

        needed = max(0, keep_count - len(protected))
        candidates = [idx for idx in content_indices if idx not in protected]
        candidates.sort(key=lambda idx: scores.get(idx, 0.0), reverse=True)
        return protected.union(candidates[:needed])

    def _score_units(self, text: str, units: list[Unit], unit_offset: int = 0) -> dict[int, float]:
        token_scores = self.scorer.score(text)
        if not token_scores:
            return {idx: 0.0 for idx in range(len(units))}

        scores: dict[int, float] = {}
        for idx, unit in enumerate(units):
            unit_start = unit.start + unit_offset
            unit_end = unit.end + unit_offset
            weighted_score = 0.0
            overlap_size = 0
            for token in token_scores:
                overlap = max(0, min(unit_end, token.end) - max(unit_start, token.start))
                if overlap <= 0:
                    continue
                weighted_score += token.score * overlap
                overlap_size += overlap
            scores[idx] = weighted_score / overlap_size if overlap_size else 0.0
        return scores

    def _split_units(self, text: str) -> list[Unit]:
        granularity = self.config.granularity
        if granularity == "token":
            pattern = self._TOKEN_RE
        elif granularity == "sentence":
            pattern = self._SENTENCE_RE
        else:
            pattern = self._PHRASE_RE

        units = [
            Unit(match.group(0), match.start(), match.end())
            for match in pattern.finditer(text)
            if match.group(0)
        ]
        return units if units else [Unit(text, 0, len(text))]


@lru_cache(maxsize=1)
def get_selective_context_compressor() -> SelectiveContextCompressor:
    return SelectiveContextCompressor(SelectiveContextConfig.from_env())


def configure_selective_context(enabled: bool | None = None) -> None:
    if enabled is not None:
        os.environ["SELECTIVE_CONTEXT_ENABLED"] = "1" if enabled else "0"
    get_selective_context_compressor.cache_clear()
