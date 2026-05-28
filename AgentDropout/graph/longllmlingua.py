import os
from functools import lru_cache
from typing import Any, Dict


DEFAULT_LONGLINGUA_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_LONGLINGUA_DEVICE_MAP = "cuda"
DEFAULT_LONGLINGUA_RATE = 0.5


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw_value!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc


def _stringify_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "\n".join(_stringify_output(item) for item in output)
    return str(output)


def _split_first_line(text: str) -> tuple[str, str]:
    newline_index = text.find("\n")
    if newline_index == -1:
        return text, ""
    return text[:newline_index], text[newline_index + 1:]


@lru_cache(maxsize=1)
def _get_prompt_compressor():
    try:
        from llmlingua import PromptCompressor
    except ImportError as exc:
        raise RuntimeError(
            "LongLLMLingua compression requires the `llmlingua` package. "
            "Install project requirements or run `pip install llmlingua`."
        ) from exc

    model_name = os.getenv("AGENTPRUNE_LONGLINGUA_MODEL", DEFAULT_LONGLINGUA_MODEL)
    device_map = os.getenv("AGENTPRUNE_LONGLINGUA_DEVICE_MAP", DEFAULT_LONGLINGUA_DEVICE_MAP)
    kwargs = {}
    if model_name:
        kwargs["model_name"] = model_name
    if device_map:
        kwargs["device_map"] = device_map
    return PromptCompressor(**kwargs)


def compress_with_longllmlingua(text: Any, question: str, instruction: str = "") -> str:
    text = _stringify_output(text).strip()
    if not text:
        return text

    condition = "\n\n".join(part for part in [question or "", instruction] if part)
    compressor = _get_prompt_compressor()
    result = compressor.compress_prompt(
        [text],
        instruction="",
        question=condition,
        rate=_env_float("AGENTPRUNE_LONGLINGUA_RATE", DEFAULT_LONGLINGUA_RATE),
        condition_in_question=os.getenv(
            "AGENTPRUNE_LONGLINGUA_CONDITION_IN_QUESTION",
            "after_condition",
        ),
        reorder_context=os.getenv("AGENTPRUNE_LONGLINGUA_REORDER_CONTEXT", "sort"),
        dynamic_context_compression_ratio=_env_float(
            "AGENTPRUNE_LONGLINGUA_DYNAMIC_CONTEXT_COMPRESSION_RATIO",
            0.3,
        ),
        condition_compare=True,
        context_budget=os.getenv("AGENTPRUNE_LONGLINGUA_CONTEXT_BUDGET", "+100"),
        rank_method="longllmlingua",
        concate_question=False,
    )
    if isinstance(result, dict):
        return result.get("compressed_prompt", text)
    return str(result)


def compress_info_messages_with_longllmlingua(
    info_messages: Dict[str, Dict[str, Any]],
    question: str,
    instruction: str = "",
) -> Dict[str, Dict[str, Any]]:
    max_chars = _env_int("AGENTPRUNE_LONGLINGUA_MAX_CHARS", 0)
    compressed_messages = {}
    for node_id, info in info_messages.items():
        output = _stringify_output(info.get("output", ""))
        stripped_output = output.strip()
        if (
            not stripped_output
            or stripped_output == "None."
            or (stripped_output.startswith("```") and stripped_output.endswith("```"))
            or (max_chars > 0 and len(output) <= max_chars)
        ):
            compressed_output = output
        else:
            answer_line, body = _split_first_line(output)
            if not body.strip() or (max_chars > 0 and len(body) <= max_chars):
                compressed_output = output
            else:
                compressed_body = compress_with_longllmlingua(
                    body,
                    question=question,
                    instruction=instruction,
                )
                compressed_output = f"{answer_line}\n{compressed_body}"
        compressed_messages[node_id] = {
            **info,
            "output": compressed_output,
            "message_type": "longllmlingua",
        }
    return compressed_messages
