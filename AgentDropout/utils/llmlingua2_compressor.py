from __future__ import annotations

import re
import threading
from typing import Any


DEFAULT_LLMLINGUA2_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
DEFAULT_FORCE_TOKENS = ("\n", ".", "?", "!", ",")
_PROMPT_COMPRESSOR_CACHE = {}
_PROMPT_COMPRESSOR_LOCK = threading.Lock()


class LLMLingua2MessageCompressor:
    """LLMLingua-2 官方 PromptCompressor 的懒加载包装器。"""

    def __init__(
        self,
        model_name: str = DEFAULT_LLMLINGUA2_MODEL,
        rate: float = 0.5,
        min_tokens: int = 20,
        force_tokens: tuple[str, ...] | list[str] | None = None,
        compress_code: bool = False,
    ):
        if not 0 < float(rate) <= 1:
            raise ValueError("LLMLingua-2 compression rate must be in (0, 1].")

        self.model_name = model_name
        self.rate = float(rate)
        self.min_tokens = max(0, int(min_tokens))
        self.force_tokens = tuple(force_tokens or DEFAULT_FORCE_TOKENS)
        self.compress_code = compress_code
        self.num_seen = 0
        self.num_compressed = 0
        self.original_tokens = 0
        self.compressed_tokens = 0
        self._compressor = None
        self._lock = threading.Lock()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_compressor"] = None
        state["_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def compress(
        self,
        text: Any,
        domain: str = "",
        role: str = "",
        message_type: str = "full",
        source: str = "spatial",
    ) -> Any:
        if not isinstance(text, str):
            return text

        stripped_text = text.strip()
        if not stripped_text:
            return text

        self.num_seen += 1
        token_count = self._approx_token_count(stripped_text)
        if token_count < self.min_tokens:
            return text
        if self._is_code_payload(stripped_text, domain):
            return text

        answer_line, compressible_text = self._split_leading_answer_line(stripped_text)
        compressible_text = compressible_text.strip()
        if not compressible_text:
            return text

        compressible_token_count = self._approx_token_count(compressible_text)
        if compressible_token_count < self.min_tokens:
            return text

        compressed = self._compress_with_llmlingua2(compressible_text)
        if not isinstance(compressed, str) or not compressed.strip():
            return text

        compressed = self._restore_leading_answer_line(compressed.strip(), answer_line)
        compressed_token_count = self._approx_token_count(compressed)
        if compressed_token_count >= token_count:
            return text

        self.num_compressed += 1
        self.original_tokens += token_count
        self.compressed_tokens += compressed_token_count
        return compressed

    def summary(self) -> str:
        if self.num_compressed == 0:
            return "LLMLingua-2 compressed 0 messages."
        ratio = self.compressed_tokens / max(self.original_tokens, 1)
        return (
            f"LLMLingua-2 compressed {self.num_compressed}/{self.num_seen} messages "
            f"from {self.original_tokens} to {self.compressed_tokens} approx tokens "
            f"(keep ratio {ratio:.3f})."
        )

    def _load_compressor(self):
        if self._compressor is not None:
            return self._compressor
        if self.model_name in _PROMPT_COMPRESSOR_CACHE:
            self._compressor = _PROMPT_COMPRESSOR_CACHE[self.model_name]
            return self._compressor

        with _PROMPT_COMPRESSOR_LOCK:
            if self._compressor is not None:
                return self._compressor
            if self.model_name in _PROMPT_COMPRESSOR_CACHE:
                self._compressor = _PROMPT_COMPRESSOR_CACHE[self.model_name]
                return self._compressor
            try:
                from llmlingua import PromptCompressor
            except ImportError as exc:
                raise RuntimeError(
                    "LLMLingua-2 compression is enabled, but the `llmlingua` "
                    "package is not installed. Install project requirements or run "
                    "`pip install llmlingua`."
                ) from exc

            self._compressor = PromptCompressor(
                model_name=self.model_name,
                use_llmlingua2=True,
            )
            _PROMPT_COMPRESSOR_CACHE[self.model_name] = self._compressor
            return self._compressor

    def _compress_with_llmlingua2(self, text: str) -> str:
        compressor = self._load_compressor()
        kwargs = {
            "rate": self.rate,
            "force_tokens": list(self.force_tokens),
            "drop_consecutive": True,
        }
        try:
            result = compressor.compress_prompt(text, **kwargs)
        except TypeError:
            result = compressor.compress_prompt(text, rate=self.rate)

        if isinstance(result, dict):
            for key in ("compressed_prompt", "compressed_text", "prompt"):
                value = result.get(key)
                if isinstance(value, str):
                    return value
        if isinstance(result, str):
            return result
        return text

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

    def _approx_token_count(self, text: str) -> int:
        return len(re.findall(r"\S+", text))

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
