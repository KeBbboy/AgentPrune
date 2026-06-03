from __future__ import annotations

import os
import re
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Iterator, Protocol

import numpy as np
from dotenv import load_dotenv


load_dotenv()


_DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
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
    model_name: str = _DEFAULT_MODEL_NAME
    lang: str = "en"
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
            model_name=os.getenv("SELECTIVE_CONTEXT_MODEL", _DEFAULT_MODEL_NAME).strip()
            or _DEFAULT_MODEL_NAME,
            lang=os.getenv("SELECTIVE_CONTEXT_LANG", "en").strip().lower() or "en",
            reduce_ratio=min(max(_env_float("SELECTIVE_CONTEXT_REDUCE_RATIO", 0.5), 0.0), 0.95),
            granularity=os.getenv("SELECTIVE_CONTEXT_GRANULARITY", "phrase").strip().lower()
            or "phrase",
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


@dataclass(frozen=True)
class LexicalUnits:
    text: list[str]
    self_info: list[float]
    unit_type: str


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
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        text_input_ids = [int(token_id) for token_id in encoded.get("input_ids", [])]
        offsets = encoded.get("offset_mapping", [])
        if not text_input_ids:
            return []

        prefix_token_id = getattr(tokenizer, "bos_token_id", None)
        if prefix_token_id is None:
            prefix_token_id = getattr(tokenizer, "eos_token_id", None)

        if prefix_token_id is not None:
            model_input_ids = [int(prefix_token_id)] + text_input_ids
            text_index_offset = 1
        else:
            model_input_ids = text_input_ids
            text_index_offset = 0

        if len(model_input_ids) <= 1:
            return []

        losses_by_text_index: list[float | None] = [None] * len(text_input_ids)
        chunk_start = 0
        while chunk_start < len(model_input_ids) - 1:
            chunk_end = min(chunk_start + self.config.max_length, len(model_input_ids))
            chunk_ids = model_input_ids[chunk_start:chunk_end]
            if len(chunk_ids) <= 1:
                break

            input_tensor = torch.tensor([chunk_ids], dtype=torch.long, device=self._device)
            with torch.no_grad():
                logits = model(input_tensor).logits[0, :-1, :]
                targets = input_tensor[0, 1:]
                losses = torch.nn.functional.cross_entropy(logits, targets, reduction="none")

            for local_index, loss in enumerate(losses.detach().cpu().tolist(), start=1):
                absolute_model_index = chunk_start + local_index
                text_index = absolute_model_index - text_index_offset
                if 0 <= text_index < len(losses_by_text_index):
                    losses_by_text_index[text_index] = float(loss)

            if chunk_end >= len(model_input_ids):
                break
            chunk_start = max(chunk_end - 1, chunk_start + 1)

        token_scores: list[TokenScore] = []
        for index, loss in enumerate(losses_by_text_index):
            if loss is None or index >= len(offsets):
                continue
            start, end = offsets[index]
            if end <= start:
                continue
            token_scores.append(TokenScore(start=int(start), end=int(end), score=loss))
        return token_scores


class SelectiveContext:
    """Official-style Selective Context API.

    Usage:
        sc = SelectiveContext(model_type="Qwen/Qwen2.5-0.5B", lang="en")
        context, reduced_content = sc(text, reduce_ratio=0.5)
    """

    _SENTENCE_SPLIT_PATTERN = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s")

    def __init__(
        self,
        model_type: str | None = None,
        lang: str | None = None,
        scorer: SurprisalScorer | None = None,
        config: SelectiveContextConfig | None = None,
    ):
        model_name = model_type or (config.model_name if config is not None else _DEFAULT_MODEL_NAME)
        resolved_lang = lang or (config.lang if config is not None else "en")
        if config is None:
            self.config = SelectiveContextConfig(model_name=model_name, lang=resolved_lang)
        else:
            self.config = replace(config, model_name=model_name, lang=resolved_lang)
        self.model_type = self.config.model_name
        self.lang = self.config.lang
        self._scorer = scorer
        self._nlp = None

    @property
    def scorer(self) -> SurprisalScorer:
        if self._scorer is None:
            self._scorer = TransformersSurprisalScorer(self.config)
        return self._scorer

    def __call__(
        self,
        text: str,
        reduce_ratio: float = 0.35,
        reduce_level: str = "phrase",
        reserved_context_ratio: float = 0,
        preserve_first_units: int = 0,
        preserve_last_units: int = 0,
    ) -> tuple[str, list[str]]:
        context = self._beautify_context(text)
        if not context:
            return context, []

        reserved_context = ""
        if reserved_context_ratio > 0:
            reserved_context_len = int(len(context) * min(float(reserved_context_ratio), 1.0))
            context, reserved_context = context[:reserved_context_len], context[reserved_context_len:]
            context = self._beautify_context(context)
            reserved_context = self._beautify_context(reserved_context)
            if not context:
                return reserved_context, []

        level = self._normalize_reduce_level(reduce_level)
        ratio = min(max(float(reduce_ratio), 0.0), 0.95)
        lexical_units = self._calculate_lexical_units(context, level)
        masked_context, reduced_content = self._self_info_mask(
            lexical_units,
            ratio,
            preserve_first_units=preserve_first_units,
            preserve_last_units=preserve_last_units,
        )
        if reserved_context:
            masked_context = self._beautify_context(masked_context + reserved_context)
        return masked_context, reduced_content

    def _calculate_lexical_units(self, context: str, reduce_level: str) -> LexicalUnits:
        sentence_units = [unit for unit in self._split_sentence_units(context) if unit.text.strip()]
        if reduce_level == "sent":
            return LexicalUnits(
                text=[unit.text for unit in sentence_units],
                self_info=[self._sentence_self_info(unit.text) for unit in sentence_units],
                unit_type="sent",
            )

        lexical_text: list[str] = []
        lexical_self_info: list[float] = []
        for sentence_unit in sentence_units:
            token_scores = self.scorer.score(sentence_unit.text)
            if reduce_level == "token":
                for token_score in token_scores:
                    token_text = sentence_unit.text[token_score.start : token_score.end]
                    if not token_text.strip():
                        continue
                    lexical_text.append(token_text)
                    lexical_self_info.append(token_score.score)
                continue

            units = self._split_units(sentence_unit.text, reduce_level)
            scores = self._score_units(units, token_scores)
            if lexical_text and units:
                units[0] = Unit(f" {units[0].text.lstrip()}", units[0].start, units[0].end)
            for index, unit in enumerate(units):
                if unit.text:
                    lexical_text.append(unit.text)
                    lexical_self_info.append(scores.get(index, 0.0))
        return LexicalUnits(
            text=lexical_text,
            self_info=lexical_self_info,
            unit_type=reduce_level,
        )

    def _self_info_mask(
        self,
        lexical_units: LexicalUnits,
        reduce_ratio: float,
        preserve_first_units: int = 0,
        preserve_last_units: int = 0,
    ) -> tuple[str, list[str]]:
        if not lexical_units.text:
            return "", []
        if reduce_ratio <= 0:
            return self._join_units(lexical_units.text, lexical_units.unit_type), []

        infos = np.array(lexical_units.self_info, dtype=float)
        if infos.size == 0 or np.all(np.isnan(infos)):
            return self._join_units(lexical_units.text, lexical_units.unit_type), []

        threshold = float(np.nanpercentile(infos, reduce_ratio * 100))
        protected_indices: set[int] = set()
        if preserve_first_units:
            protected_indices.update(range(min(preserve_first_units, len(lexical_units.text))))
        if preserve_last_units:
            start = max(0, len(lexical_units.text) - preserve_last_units)
            protected_indices.update(range(start, len(lexical_units.text)))

        kept_units: list[str] = []
        reduced_content: list[str] = []
        for index, (unit_text, self_info) in enumerate(
            zip(lexical_units.text, lexical_units.self_info)
        ):
            if index not in protected_indices and self_info < threshold:
                reduced_content.append(self._beautify_context(unit_text))
            else:
                kept_units.append(unit_text)

        compressed_context = self._join_units(kept_units, lexical_units.unit_type)
        reduced_content = [content for content in reduced_content if content]
        return compressed_context, reduced_content

    def _sentence_self_info(self, text: str) -> float:
        token_scores = self.scorer.score(text)
        if not token_scores:
            return 0.0
        return float(np.mean([token.score for token in token_scores]))

    def _score_units(self, units: list[Unit], token_scores: list[TokenScore]) -> dict[int, float]:
        if not token_scores:
            return {idx: 0.0 for idx in range(len(units))}

        scores_by_unit: dict[int, list[float]] = {idx: [] for idx in range(len(units))}
        for token in token_scores:
            matched_indices = [
                idx
                for idx, unit in enumerate(units)
                if max(0, min(unit.end, token.end) - max(unit.start, token.start)) > 0
            ]
            if not matched_indices:
                continue
            partial_info = token.score / len(matched_indices)
            for idx in matched_indices:
                scores_by_unit[idx].append(partial_info)
        return {
            idx: float(np.mean(unit_scores)) if unit_scores else 0.0
            for idx, unit_scores in scores_by_unit.items()
        }

    def _split_sentence_units(self, text: str) -> list[Unit]:
        units: list[Unit] = []
        start = 0
        for match in self._SENTENCE_SPLIT_PATTERN.finditer(text):
            raw_sentence = text[start : match.start()]
            leading_trimmed = len(raw_sentence) - len(raw_sentence.lstrip())
            trailing_trimmed = len(raw_sentence.rstrip())
            sentence = raw_sentence.strip()
            if sentence:
                units.append(
                    Unit(
                        sentence,
                        start + leading_trimmed,
                        start + trailing_trimmed,
                    )
                )
            start = match.end()

        raw_sentence = text[start:]
        leading_trimmed = len(raw_sentence) - len(raw_sentence.lstrip())
        trailing_trimmed = len(raw_sentence.rstrip())
        sentence = raw_sentence.strip()
        if sentence:
            units.append(Unit(sentence, start + leading_trimmed, start + trailing_trimmed))
        return units if units else [Unit(text, 0, len(text))]

    def _split_units(self, text: str, reduce_level: str) -> list[Unit]:
        level = self._normalize_reduce_level(reduce_level)
        if level == "sent":
            return self._split_sentence_units(text)
        if level == "token":
            return [
                Unit(text[token.start : token.end], token.start, token.end)
                for token in self.scorer.score(text)
                if text[token.start : token.end].strip()
            ]
        return self._split_phrase_units(text)

    def _split_phrase_units(self, text: str) -> list[Unit]:
        doc = self._get_phrase_tokenizer()(text)
        units: list[Unit] = []
        for index, token in enumerate(doc):
            start = token.idx
            phrase_text = token.text
            if index > 0:
                previous = doc[index - 1]
                whitespace = previous.whitespace_
                start = token.idx - len(whitespace)
                phrase_text = f"{whitespace}{token.text}"
            if phrase_text:
                units.append(Unit(phrase_text, start, token.idx + len(token.text)))
        return units if units else [Unit(text, 0, len(text))]

    def _get_phrase_tokenizer(self):
        if self._nlp is not None:
            return self._nlp

        import spacy

        if self.lang == "en":
            nlp = spacy.load("en_core_web_sm", disable=["ner"])
            if "merge_noun_chunks" not in nlp.pipe_names:
                nlp.add_pipe("merge_noun_chunks")
        elif self.lang == "zh":
            nlp = spacy.load("zh_core_web_sm", disable=["ner"])
        else:
            raise NotImplementedError("Selective Context currently supports lang='en' and lang='zh'.")

        self._nlp = nlp
        return self._nlp

    @staticmethod
    def _normalize_reduce_level(reduce_level: str) -> str:
        normalized = (reduce_level or "phrase").strip().lower()
        if normalized == "sent":
            return "sent"
        if normalized == "token":
            return "token"
        if normalized == "phrase":
            return "phrase"
        raise ValueError("reduce_level must be one of: sent, phrase, token")

    @staticmethod
    def _beautify_context(context: str) -> str:
        return re.sub(r"\s+", " ", context).strip()

    def _join_units(self, units: list[str], unit_type: str) -> str:
        if unit_type == "sent":
            return self._beautify_context(" ".join(units))
        return self._beautify_context("".join(units))


class SelectiveContextCompressor:
    """Implements Selective Context for inter-agent messages.

    The compressor keeps the existing graph integration while delegating the
    actual compression to the official-style SelectiveContext API.
    """

    def __init__(
        self,
        config: SelectiveContextConfig | None = None,
        scorer: SurprisalScorer | None = None,
    ):
        self.config = config if config is not None else SelectiveContextConfig()
        self._selective_context = SelectiveContext(
            model_type=self.config.model_name,
            lang=self.config.lang,
            scorer=scorer,
            config=self.config,
        )
        self._warned_failure = False

    @property
    def scorer(self) -> SurprisalScorer:
        return self._selective_context.scorer

    def compress_payload(self, payload: Any) -> Any:
        if isinstance(payload, str):
            return self.compress(payload)
        if isinstance(payload, list):
            return [self.compress_payload(item) for item in payload]
        return payload

    def compress(self, text: str) -> str:
        compressed, _ = self.compress_with_reduced_content(text)
        return compressed

    def compress_with_reduced_content(self, text: str) -> tuple[str, list[str]]:
        if not _RUNTIME_ENABLED.get():
            return text, []
        if not self.config.enabled:
            return text, []
        if self.config.reduce_ratio <= 0:
            return text, []

        prefix, body = self._split_preserved_first_line(text)
        if not body:
            return text, []
        if len(body.strip()) < self.config.min_chars:
            return text, []
        if self._eligible_unit_count(body) <= self.config.min_units:
            return text, []

        try:
            compressed_body, reduced_content = self._selective_context(
                body,
                reduce_ratio=self.config.reduce_ratio,
                reduce_level=self.config.granularity,
                preserve_first_units=self.config.preserve_first_units,
                preserve_last_units=self.config.preserve_last_units,
            )
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
            return text, []

        if not compressed_body:
            return text, []
        compressed = f"{prefix}{compressed_body}"
        if self.config.debug:
            print(
                "[SelectiveContext] compressed "
                f"chars={len(text)}->{len(compressed)} "
                f"reduced_units={len(reduced_content)} "
                f"preserve_first_line={self.config.preserve_first_line}",
                flush=True,
            )
        return compressed, reduced_content

    def _split_preserved_first_line(self, text: str) -> tuple[str, str]:
        if not self.config.preserve_first_line:
            return "", text

        first_newline_index = text.find("\n")
        if first_newline_index < 0:
            return text, ""
        split_index = first_newline_index + 1
        return text[:split_index], text[split_index:]

    def _eligible_unit_count(self, text: str) -> int:
        context = self._selective_context._beautify_context(text)
        level = self._selective_context._normalize_reduce_level(self.config.granularity)
        if level == "sent":
            return len(
                [
                    unit
                    for unit in self._selective_context._split_sentence_units(context)
                    if unit.text.strip()
                ]
            )

        count = 0
        for sentence_unit in self._selective_context._split_sentence_units(context):
            count += len(
                [
                    unit
                    for unit in self._selective_context._split_units(sentence_unit.text, level)
                    if unit.text.strip()
                ]
            )
        return count


@lru_cache(maxsize=1)
def get_selective_context_compressor() -> SelectiveContextCompressor:
    return SelectiveContextCompressor(SelectiveContextConfig.from_env())


def configure_selective_context(enabled: bool | None = None) -> None:
    if enabled is not None:
        os.environ["SELECTIVE_CONTEXT_ENABLED"] = "1" if enabled else "0"
    get_selective_context_compressor.cache_clear()
