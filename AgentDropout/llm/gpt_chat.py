import asyncio
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import anthropic
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass
class LLMResponse:
    reasoning: Optional[str]
    answer: str
    full_text: str

    def __str__(self):
        return self.full_text


def split_thinking(text: str) -> LLMResponse:
    if not text:
        return LLMResponse(reasoning=None, answer="", full_text="")
    m = THINK_RE.search(text)
    if not m:
        return LLMResponse(reasoning=None, answer=text.strip(), full_text=text)
    reasoning = m.group(1).strip()
    answer = (text[: m.start()] + text[m.end():]).strip()
    return LLMResponse(reasoning=reasoning, answer=answer, full_text=text)


def create_llm_response_from_parts(reasoning: Optional[str], answer: str) -> LLMResponse:
    if reasoning:
        full_text = f"<think>{reasoning}</think>{answer}"
    else:
        full_text = answer
    return LLMResponse(reasoning=reasoning, answer=answer.strip(), full_text=full_text)

try:
    import async_timeout
except ImportError:
    async_timeout = None

from AgentDropout.llm.format import Message
from AgentDropout.llm.llm import LLM
from AgentDropout.llm.llm_registry import LLMRegistry
from AgentDropout.llm.price import cost_count, cost_count_llama3
from AgentDropout.utils.globals import TokenUsageRecords


load_dotenv()


def _norm_url(url: str | None) -> str | None:
    """Strip whitespace, trailing slash, and optional /chat/completions suffix."""
    if not url:
        return None
    url = url.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    return url


# --- Provider credentials (resolved from .env) ---------------------------------
# qwen (OpenAI-compatible) uses BASE_URL/API_KEY.
# minimax (Anthropic-compatible) uses MINIMAX_BASE_URL/MINIMAX_API_KEY.
QWEN_URL = _norm_url(os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL"))
QWEN_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")

MODEL = "MiniMax-M3"  # exposed as MINIMAX_MODEL_NAME for backwards compatibility
MINIMAX_MODEL_NAME = MODEL
MM_KEY = os.getenv("MINIMAX_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or QWEN_KEY
MINIMAX_API_KEY = MM_KEY
MM_URL = os.getenv("MINIMAX_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL")
MINIMAX_BASE_URL = MM_URL
MM_SYSTEM = "You are a helpful assistant."
MINIMAX_SYSTEM_PROMPT = MM_SYSTEM

LOCAL_URL = _norm_url(os.getenv("LOCAL_BASE_URL") or QWEN_URL) or "http://localhost:6789/v1"
LOCAL_BASE_URL = LOCAL_URL
LOCAL_KEY = os.getenv("LOCAL_API_KEY") or QWEN_KEY or "API-KEY"
LOCAL_API_KEY = LOCAL_KEY

# Legacy aliases kept for external imports (the project now uses QWEN_* instead).
MINE_BASE_URL = QWEN_URL
MINE_API_KEYS = QWEN_KEY

# --- Concurrency / retry tunables ---------------------------------------------
MAX_CONCURRENCY = max(1, int(os.getenv("LLM_MAX_CONCURRENT_REQUESTS", "10")))
LLM_MAX_CONCURRENT_REQUESTS = MAX_CONCURRENCY
MIN_INTERVAL = max(0.0, float(os.getenv("LLM_REQUEST_INTERVAL_SECONDS", "0")))
LLM_REQUEST_INTERVAL_SECONDS = MIN_INTERVAL
RETRIES = max(1, int(os.getenv("LLM_RETRY_ATTEMPTS", "8")))
LLM_RETRY_ATTEMPTS = RETRIES
LOG_USAGE = os.getenv("LOG_MINIMAX_TOKEN_USAGE", "0").lower() in {"1", "true", "yes"}
LOG_MINIMAX_TOKEN_USAGE = LOG_USAGE
MAX_TOKENS = max(1, int(os.getenv("MINIMAX_MAX_TOKENS", "32768")))
MINIMAX_DEFAULT_MAX_TOKENS = MAX_TOKENS
# Official MMLU-Pro inference params: temperature=0.0, top_p=0.95.
TEMPERATURE = float(os.getenv("MINIMAX_TEMPERATURE", "0.0"))
MINIMAX_DEFAULT_TEMPERATURE = TEMPERATURE
TOP_P = float(os.getenv("MINIMAX_TOP_P", "0.95"))
MINIMAX_DEFAULT_TOP_P = TOP_P

# --- Thinking mode toggle (configured by experiment args; default: ON) --------
# All providers routed here can accept a provider-specific thinking parameter.
# Keep this as an in-process setting instead of reading it from environment
# variables so experiments can control it via argparse.
LLM_ENABLE_THINKING = True
LLM_THINKING_BUDGET = 8192


def configure_thinking(
    enable_thinking: bool = True,
    thinking_budget: int = 8192,
) -> None:
    """Configure provider-specific thinking parameters for this process.
    
    Unified interface for thinking budget across all providers:
    - Qwen3: thinking_budget directly sets the token budget
    - MiniMax-M3: thinking_budget > 0 means "adaptive", 0 means "disabled"
    """
    global LLM_ENABLE_THINKING, LLM_THINKING_BUDGET
    LLM_ENABLE_THINKING = bool(enable_thinking)
    LLM_THINKING_BUDGET = int(thinking_budget)


def parse_bool_arg(value: Any) -> bool:
    """Parse true/false CLI values without argparse's bool('false') pitfall."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError("expected true or false")


def _thinking_extra_body(model: str) -> Dict[str, Any]:
    """Build the per-provider extra_body that controls thinking mode.

    Returns an empty dict when thinking is disabled or when the provider
    doesn't expose a switch. Callers should pass it via ``extra_body=`` to
    the OpenAI-compatible client so unknown keys are forwarded verbatim.
    
    Unified interface:
    - LLM_ENABLE_THINKING controls whether to enable thinking for both models
    - LLM_THINKING_BUDGET controls the budget for both models:
        * Qwen3: directly sets the token budget (only when 128 <= budget <= 32768)
        * MiniMax-M3: >0 for adaptive, 0 for disabled
    """
    if not LLM_ENABLE_THINKING:
        return {}
    model_lower = (model or "").lower()
    # Qwen3 / SiliconFlow path: enable_thinking + thinking_budget.
    if "qwen" in model_lower:
        body: Dict[str, Any] = {"enable_thinking": True}
        if 128 <= LLM_THINKING_BUDGET <= 32768:
            body["thinking_budget"] = LLM_THINKING_BUDGET
        return body
    # MiniMax-M3 path: thinking.type ∈ {adaptive, disabled}.
    # Use thinking_budget > 0 for adaptive, 0 for disabled.
    if "minimax" in model_lower or "MiniMax" in model:
        thinking_type = "adaptive" if LLM_THINKING_BUDGET > 0 else "disabled"
        return {"thinking": {"type": thinking_type}}
    return {}

_SEMA = asyncio.Semaphore(MAX_CONCURRENCY)
_LLM_REQUEST_SEMAPHORE = _SEMA
_LOCK = asyncio.Lock()
_LLM_REQUEST_LOCK = _LOCK
_LAST_TS = 0.0
_LAST_LLM_REQUEST_TS = _LAST_TS


@asynccontextmanager
async def _noop_timeout():
    yield


def _timeout(seconds: float):
    if async_timeout is not None:
        return async_timeout.timeout(seconds)
    if hasattr(asyncio, "timeout"):
        return asyncio.timeout(seconds)
    return _noop_timeout()


def _run_async_from_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: Dict[str, Any] = {}

    def runner():
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


async def _wait_slot() -> None:
    """Throttle calls so consecutive requests are spaced by MIN_INTERVAL seconds."""
    global _LAST_TS
    if MIN_INTERVAL <= 0:
        return
    async with _LOCK:
        wait = MIN_INTERVAL - (time.monotonic() - _LAST_TS)
        if wait > 0:
            await asyncio.sleep(wait)
        _LAST_TS = time.monotonic()


# --- Message helpers ----------------------------------------------------------
def _msg_get(message: Any, key: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
                elif "content" in block:
                    parts.append(_to_text(block["content"]))
                else:
                    parts.append(str(block))
            else:
                text = getattr(block, "text", None)
                parts.append(str(text) if text is not None else str(block))
        return "\n".join(p for p in parts if p)
    return str(content)


def _content_blocks(content: Any) -> List[Dict[str, str]]:
    text = _to_text(content)
    return [{"type": "text", "text": text if text else " "}]


def _append_msg(messages: List[Dict[str, Any]], role: str, content: Any) -> None:
    if role not in {"user", "assistant"}:
        role = "user"
    blocks = _content_blocks(content)
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"].extend(blocks)
    else:
        messages.append({"role": role, "content": blocks})


def _client_kwargs() -> Dict[str, str]:
    kwargs: Dict[str, str] = {}
    if MM_KEY:
        kwargs["api_key"] = MM_KEY
    if MM_URL:
        kwargs["base_url"] = MM_URL.strip().rstrip("/")
    return kwargs


def _anthropic_base_url() -> Optional[str]:
    """Resolve the anthropic-compatible base_url for MiniMax-M3.

    The MiniMax platform exposes two parallel APIs:

    * OpenAI-compatible: ``https://api.minimaxi.com/v1/chat/completions``
    * Anthropic-compatible: ``https://api.minimaxi.com/anthropic/v1/messages``

    The OpenAI SDK keeps working against ``.../v1``; the anthropic SDK needs the
    URL rewritten to ``.../anthropic`` (the SDK auto-appends ``/v1/messages``).
    If the operator already configured an anthropic-style URL via
    ``ANTHROPIC_BASE_URL`` we trust that as-is.
    """
    explicit_anthropic = os.getenv("ANTHROPIC_BASE_URL")
    if explicit_anthropic:
        return explicit_anthropic.strip().rstrip("/")
    if not MM_URL:
        return None
    base = MM_URL.strip().rstrip("/")
    # Drop any trailing /v1 (or /v1/) before appending /anthropic.
    if base.endswith("/v1"):
        base = base[: -3].rstrip("/")
    return f"{base}/anthropic"


def _prepare_msgs(msg: Union[str, List[Dict]]) -> tuple[str, List[Dict[str, Any]]]:
    if isinstance(msg, str):
        msg = [{"role": "user", "content": msg}]

    system_parts: List[str] = []
    messages: List[Dict[str, Any]] = []
    for item in msg:
        role = str(_msg_get(item, "role", "user"))
        content = _msg_get(item, "content", "")
        if role == "system":
            text = _to_text(content)
            if text:
                system_parts.append(text)
            continue
        _append_msg(messages, role, content)

    if not messages:
        messages.append({"role": "user", "content": [{"type": "text", "text": " "}]})

    system = "\n\n".join(system_parts).strip() or MM_SYSTEM
    return system, messages


def _prompt_text(system: str, messages: List[Dict[str, Any]]) -> str:
    parts = [system] if system else []
    for m in messages:
        parts.append(f"{m['role']}: {_to_text(m.get('content', ''))}")
    return "\n\n".join(parts)


def _extract_text(message: Any) -> str:
    parts: List[str] = []
    for block in getattr(message, "content", []) or []:
        block_type = getattr(block, "type", None)
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        elif block_type == "text":
            parts.append(str(getattr(block, "text", "")))
    response = "\n".join(p for p in parts if p).strip()
    return response or _to_text(getattr(message, "content", ""))


# Backwards-compatible aliases for the old anthropic-prefixed helpers.
_anthropic_content_blocks = _content_blocks
_append_anthropic_message = _append_msg
_anthropic_client_kwargs = _client_kwargs
_prepare_anthropic_messages = _prepare_msgs
_anthropic_prompt_text = _prompt_text
_extract_anthropic_text = _extract_text


def _request_kwargs(
    msg: Union[str, List[Dict]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> tuple[Dict[str, Any], str]:
    system, messages = _prepare_msgs(msg)
    kwargs: Dict[str, Any] = {
        "model": MODEL,
        "max_tokens": max_tokens or 1000,
        "system": system,
        "messages": messages,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    return kwargs, _prompt_text(system, messages)


# Backwards-compatible alias.
_minimax_request_kwargs = _request_kwargs


def _record_usage(prompt: str, response: str, usage: Any = None, model: str = MODEL) -> None:
    model_lower = model.lower()
    if "qwen" in model_lower:
        _, prompt_tokens, completion_tokens = cost_count_llama3(prompt, response, model, usage=usage)
    elif "minimax" in model_lower or "MiniMax" in model:
        _, prompt_tokens, completion_tokens = cost_count(prompt, response, model, usage=usage)
    else:
        _, prompt_tokens, completion_tokens = cost_count(prompt, response, model, usage=usage)
    
    cached_tokens = None
    reasoning_tokens = None
    if usage is not None:
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        if prompt_details:
            cached_tokens = getattr(prompt_details, "cached_tokens", None)
        # Anthropic-style fallback (e.g. MiniMax-M3 returns cache_read_input_tokens).
        if cached_tokens is None:
            cached_tokens = getattr(usage, "cache_read_input_tokens", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        if completion_details:
            reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)
    
    TokenUsageRecords.instance().value.append({
        "model": model,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
    })
    if LOG_USAGE:
        cached_str = f" cached={cached_tokens}" if cached_tokens else ""
        reasoning_str = f" reasoning={reasoning_tokens}" if reasoning_tokens else ""
        print(f"{model} input_tokens={prompt_tokens} output_tokens={completion_tokens}{cached_str}{reasoning_str}")


# Backwards-compatible alias.
_record_minimax_usage = _record_usage


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(model: str, msg: List[Dict], max_tokens: Optional[int] = None, temperature: Optional[float] = None):
    kwargs, prompt = _request_kwargs(msg, max_tokens=max_tokens, temperature=temperature)
    client = anthropic.AsyncAnthropic(**_client_kwargs())
    try:
        async with _timeout(1000):
            completion = await client.messages.create(**kwargs)
        text = _extract_text(completion)
        if isinstance(text, str):
            _record_usage(prompt, text, usage=completion.usage, model=MODEL)
            return text
    except Exception as e:
        raise RuntimeError(f"Failed to complete the async chat request: {e}")


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
def chat(model: str, msg: List[Dict], max_tokens: Optional[int] = None, temperature: Optional[float] = None):
    kwargs, prompt = _request_kwargs(msg, max_tokens=max_tokens, temperature=temperature)
    client = anthropic.Anthropic(**_client_kwargs())
    try:
        completion = client.messages.create(**kwargs)
        text = _extract_text(completion)
        if isinstance(text, str):
            _record_usage(prompt, text, usage=completion.usage, model=MODEL)
            return text
    except Exception as e:
        raise RuntimeError(f"Failed to complete the chat request: {e}")


async def achat_deepseek(model: str, msg: List[Dict]):
    # NOTE: deepseek_api / deepseek_url are intentionally unresolved legacy
    # references; this path is disabled and only retained for the registry.
    api_kwargs = dict(api_key=deepseek_api, base_url=deepseek_url)
    client = AsyncOpenAI(**api_kwargs)
    try:
        async with _timeout(1000):
            completion = await client.chat.completions.create(model="", messages=msg)
        return completion.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Failed to complete the async chat request: {e}")


@retry(wait=wait_random_exponential(min=5, max=90), stop=stop_after_attempt(RETRIES), reraise=True)
async def achat_qwen(model: str, msg: List[Dict], max_tokens: Optional[int] = None, temperature: Optional[float] = None):
    # Normalise the message list to the OpenAI chat-completions dict schema so
    # callers can pass ``List[Message]``, ``List[dict]`` or a raw ``str`` without
    # the SDK choking on dataclass objects.
    messages = _normalize_openai_messages(msg)
    client = AsyncOpenAI(api_key=QWEN_KEY, base_url=QWEN_URL)
    try:
        async with _SEMA:
            await _wait_slot()
            async with _timeout(1000):
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "extra_body": _thinking_extra_body(model),
                }
                if max_tokens is not None:
                    kwargs["max_tokens"] = int(max_tokens)
                if temperature is not None:
                    kwargs["temperature"] = float(temperature)
                completion = await client.chat.completions.create(**kwargs)
        message = completion.choices[0].message
        text = message.content
        reasoning_content = getattr(message, "reasoning_content", None)
        if isinstance(text, str):
            prompt = "".join(item["content"] for item in messages)
            cost_count_llama3(prompt, text, model, usage=completion.usage)
            if reasoning_content:
                return create_llm_response_from_parts(reasoning_content, text)
            return split_thinking(text)
    except Exception as e:
        print(f"Error in achat_qwen: {e}")
        raise


# Backwards-compatible alias (the function dispatches via Qwen / OpenAI-compatible APIs).
achat_llama = achat_qwen


def _defaults(self, max_tokens, temperature, num_comps):
    if max_tokens is None:
        max_tokens = self.DEFAULT_MAX_TOKENS
    if temperature is None:
        temperature = self.DEFAULT_TEMPERATURE
    if num_comps is None:
        num_comps = self.DEFUALT_NUM_COMPLETIONS
    return max_tokens, temperature, num_comps


def _to_msg_list(messages):
    if isinstance(messages, str):
        messages = [Message(role="user", content=messages)]
    return messages


def _normalize_openai_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """Coerce Message / dict / str into the OpenAI chat-completions schema.

    The MiniMax-M3 endpoint at ``https://api.minimaxi.com/v1`` is OpenAI-
    compatible and requires ``{"role": str, "content": str | list}`` items.
    ``achat_mm`` previously forwarded the raw ``Message`` dataclass list and
    relied on duck-typed attribute access; this normalises once and early so
    downstream SDKs can't choke on a malformed payload.
    """

    out: List[Dict[str, Any]] = []
    for raw in _to_msg_list(messages):
        if isinstance(raw, str):
            out.append({"role": "user", "content": raw})
            continue
        if isinstance(raw, dict):
            role = str(raw.get("role", "user"))
            content = raw.get("content", "")
        else:
            role = str(getattr(raw, "role", "user"))
            content = getattr(raw, "content", "")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = _to_text(content)
        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"
        out.append({"role": role, "content": content})
    if not out:
        out.append({"role": "user", "content": " "})
    return out


@LLMRegistry.register('GPTChat')
class GPTChat(LLM):
    """Anthropic-compatible client, kept for legacy callers only.

    MiniMax-M3 routing goes through :class:`MiniMaxChat` (the ``minimax``
    registry key) which calls the OpenAI-compatible MiniMax-M3 endpoint.
    This class still talks to the Anthropic SDK with the constants from the
    top of this file (``MODEL``/``MM_KEY``/``MM_URL``) so external imports
    keep working, but new code should not use it.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(self, messages: List[Message], max_tokens=None, temperature=None, num_comps=None):
        max_tokens, temperature, num_comps = _defaults(self, max_tokens, temperature, num_comps)
        return await achat(self.model_name, _to_msg_list(messages), max_tokens=max_tokens, temperature=temperature)

    def gen(self, messages: List[Message], max_tokens=None, temperature=None, num_comps=None):
        max_tokens, temperature, num_comps = _defaults(self, max_tokens, temperature, num_comps)
        return chat(self.model_name, _to_msg_list(messages), max_tokens=max_tokens, temperature=temperature)


@LLMRegistry.register('deepseek')
class DeepseekChat(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(self, messages: List[Message], max_tokens=None, temperature=None, num_comps=None):
        max_tokens, temperature, num_comps = _defaults(self, max_tokens, temperature, num_comps)
        return await achat_deepseek(self.model_name, _to_msg_list(messages))

    def gen(self, messages: List[Message], max_tokens=None, temperature=None, num_comps=None):
        pass


@LLMRegistry.register('llama')
class LlamaChat(LLM):
    # Match the MMLU-Pro recipe's per-request budget for Qwen3-14B. The default
    # 1000 tokens returned by the base ``LLM`` class is far too small for a
    # thinking-budget call against Qwen3, which routinely emits multi-KB
    # reasoning plus a final ``Answer: X`` line.
    DEFAULT_MAX_TOKENS = MAX_TOKENS
    DEFAULT_TEMPERATURE = TEMPERATURE

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(self, messages: List[Message], max_tokens=None, temperature=None, num_comps=None):
        max_tokens, temperature, num_comps = _defaults(self, max_tokens, temperature, num_comps)
        return await achat_qwen(self.model_name, _to_msg_list(messages), max_tokens=max_tokens, temperature=temperature)

    def gen(self, messages: List[Message], max_tokens=None, temperature=None, num_comps=None):
        max_tokens, temperature, num_comps = _defaults(self, max_tokens, temperature, num_comps)
        return _run_async_from_sync(
            achat_qwen(self.model_name, _to_msg_list(messages), max_tokens=max_tokens, temperature=temperature)
        )


@retry(wait=wait_random_exponential(min=5, max=90), stop=stop_after_attempt(RETRIES), reraise=True)
async def achat_mm(model: str, msg: List[Dict], max_tokens=None, temperature=None, top_p=None):
    """Streaming MiniMax-M3 caller using the official ``anthropic`` SDK.

    Anthropic differs from the OpenAI-compatible SDK in three ways:
    * request takes ``system`` + ``messages[].content = [{"type":"text","text":...}]``
      instead of flat ``messages[].content`` strings;
    * the response body is ``message.content = [block, ...]`` where each block has
      ``type ∈ {"text","thinking"}``. Reasoning lives in ``block.thinking`` and the
      final answer text in ``block.text``;
    * thinking is enabled via the top-level ``thinking={"type":"enabled","budget_tokens":...}``
      param instead of ``extra_body``.

    The MiniMax platform requires streaming once ``thinking`` is enabled or
    ``max_tokens`` is large (it returns ``Streaming is required for operations
    that may take longer than 10 minutes`` otherwise), so we always issue the
    request through ``client.messages.stream`` and assemble the answer from the
    emitted content blocks.
    """
    try:
        system, messages = _prepare_msgs(msg)
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max(1024, int(max_tokens or MAX_TOKENS)),
            "system": system,
            "messages": messages,
        }
        if LLM_ENABLE_THINKING and LLM_THINKING_BUDGET > 0:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": int(LLM_THINKING_BUDGET)}
        if temperature is not None:
            kwargs["temperature"] = float(temperature)

        client = anthropic.AsyncAnthropic(api_key=MM_KEY, base_url=_anthropic_base_url())
        thinking_text = ""
        answer_text = ""
        usage_obj = None
        async with _SEMA:
            await _wait_slot()
            async with client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    etype = getattr(event, "type", None)
                    if etype == "content_block_start":
                        block = getattr(event, "content_block", None)
                        btype = getattr(block, "type", None)
                        if btype == "thinking":
                            thinking_text = ""
                        elif btype == "text":
                            answer_text = ""
                    elif etype == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        btype = getattr(getattr(event, "content_block", None), "type", None)
                        if btype == "thinking" or getattr(delta, "type", None) == "thinking_delta":
                            thinking_text += getattr(delta, "thinking", "") or ""
                        elif btype == "text" or getattr(delta, "type", None) == "text_delta":
                            answer_text += getattr(delta, "text", "") or ""
                try:
                    final_msg = await stream.get_final_message()
                    usage_obj = getattr(final_msg, "usage", None)
                except Exception:
                    usage_obj = None

        prompt = _prompt_text(system, messages)
        _record_usage(prompt, answer_text, usage=usage_obj, model=model)

        # Build an LLMResponse with reasoning + answer preserved separately.
        if thinking_text:
            resp = create_llm_response_from_parts(thinking_text, answer_text)
        else:
            resp = split_thinking(answer_text)
        # Defensive: downstream code expects a dataclass; coerce any stray
        # string into the same shape so the MMLU evaluator receives a stable
        # ``LLMResponse`` (with reasoning stripped, full_text = answer).
        if not isinstance(resp, LLMResponse):
            resp = LLMResponse(reasoning=None, answer=str(resp), full_text=str(resp))
        return resp
    except Exception as e:
        print(f"Error in achat_mm: {e}")
        # Guarantee: downstream code (AnalyzeAgent / BenchmarkAgent) consumes
        # LLMResponse. Return a degraded LLMResponse rather than raising,
        # so a single failed sample doesn't kill an entire evaluation batch.
        return LLMResponse(reasoning=None, answer="", full_text="")


# Backwards-compatible alias.
achat_minimax = achat_mm


@retry(wait=wait_random_exponential(min=5, max=90), stop=stop_after_attempt(RETRIES), reraise=True)
def chat_mm(model: str, msg: List[Dict], max_tokens=None, temperature=None, top_p=None):
    """Synchronous streaming variant of :func:`achat_mm` using ``anthropic.Anthropic``."""
    try:
        system, messages = _prepare_msgs(msg)
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max(1024, int(max_tokens or MAX_TOKENS)),
            "system": system,
            "messages": messages,
        }
        if LLM_ENABLE_THINKING and LLM_THINKING_BUDGET > 0:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": int(LLM_THINKING_BUDGET)}
        if temperature is not None:
            kwargs["temperature"] = float(temperature)

        client = anthropic.Anthropic(api_key=MM_KEY, base_url=_anthropic_base_url())
        thinking_text = ""
        answer_text = ""
        usage_obj = None
        with client.messages.stream(**kwargs) as stream:
            for event in stream:
                etype = getattr(event, "type", None)
                if etype == "content_block_start":
                    block = getattr(event, "content_block", None)
                    btype = getattr(block, "type", None)
                    if btype == "thinking":
                        thinking_text = ""
                    elif btype == "text":
                        answer_text = ""
                elif etype == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    btype = getattr(getattr(event, "content_block", None), "type", None)
                    if btype == "thinking" or getattr(delta, "type", None) == "thinking_delta":
                        thinking_text += getattr(delta, "thinking", "") or ""
                    elif btype == "text" or getattr(delta, "type", None) == "text_delta":
                        answer_text += getattr(delta, "text", "") or ""
            try:
                final_msg = stream.get_final_message()
                usage_obj = getattr(final_msg, "usage", None)
            except Exception:
                usage_obj = None

        prompt = _prompt_text(system, messages)
        _record_usage(prompt, answer_text, usage=usage_obj, model=model)
        if thinking_text:
            return create_llm_response_from_parts(thinking_text, answer_text)
        return split_thinking(answer_text)
    except Exception as e:
        print(f"Error in chat_mm: {e}")
        raise


# Backwards-compatible alias.
chat_minimax = chat_mm


@LLMRegistry.register('minimax')
class MiniMaxChat(LLM):
    """Client for the MiniMax-M3 chat-completions endpoint.

    Uses the OpenAI-compatible request schema against
    ``${MINIMAX_BASE_URL}/chat/completions`` (defaults to
    ``https://api.minimaxi.com/v1``). Default sampling parameters match the
    MMLU-Pro official recipe (T=0, top_p=0.95) so reproductions of
    ``run_*.py`` use identical decoding settings.
    """

    # Override the base-class defaults so ``_defaults()`` returns the env-
    # driven values (16384 / 0.0 / 0.95) instead of the generic 1000 / 0.2.
    DEFAULT_MAX_TOKENS = MAX_TOKENS
    DEFAULT_TEMPERATURE = TEMPERATURE

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(self, messages: List[Message], max_tokens=None, temperature=None, num_comps=None, top_p=None):
        max_tokens, temperature, num_comps = _defaults(self, max_tokens, temperature, num_comps)
        if top_p is None:
            top_p = TOP_P
        return await achat_mm(self.model_name, messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    def gen(self, messages: List[Message], max_tokens=None, temperature=None, num_comps=None, top_p=None):
        max_tokens, temperature, num_comps = _defaults(self, max_tokens, temperature, num_comps)
        if top_p is None:
            top_p = TOP_P
        return chat_mm(self.model_name, messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
