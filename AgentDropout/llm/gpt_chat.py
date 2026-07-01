import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import anthropic
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

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
MAX_TOKENS = max(1, int(os.getenv("MINIMAX_MAX_TOKENS", "2048")))
MINIMAX_DEFAULT_MAX_TOKENS = MAX_TOKENS
# Official MMLU-Pro inference params: temperature=0.0, top_p=0.95.
TEMPERATURE = float(os.getenv("MINIMAX_TEMPERATURE", "0.0"))
MINIMAX_DEFAULT_TEMPERATURE = TEMPERATURE
TOP_P = float(os.getenv("MINIMAX_TOP_P", "0.95"))
MINIMAX_DEFAULT_TOP_P = TOP_P

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


def _record_usage(prompt: str, response: str, usage: Any = None) -> None:
    _, prompt_tokens, completion_tokens = cost_count(prompt, response, MODEL, usage=usage)
    TokenUsageRecords.instance().value.append({
        "model": MODEL,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
    })
    if LOG_USAGE:
        print(f"{MODEL} input_tokens={prompt_tokens} output_tokens={completion_tokens}")


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
            _record_usage(prompt, text, usage=completion.usage)
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
            _record_usage(prompt, text, usage=completion.usage)
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
async def achat_qwen(model: str, msg: List[Dict]):
    client = AsyncOpenAI(api_key=QWEN_KEY, base_url=QWEN_URL)
    try:
        async with _SEMA:
            await _wait_slot()
            async with _timeout(1000):
                completion = await client.chat.completions.create(model=model, messages=msg)
        text = completion.choices[0].message.content
        if isinstance(text, str):
            prompt = "".join(item["content"] for item in msg)
            cost_count_llama3(prompt, text, model, usage=completion.usage)
            return text
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


@LLMRegistry.register('GPTChat')
class GPTChat(LLM):
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
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(self, messages: List[Message], max_tokens=None, temperature=None, num_comps=None):
        max_tokens, temperature, num_comps = _defaults(self, max_tokens, temperature, num_comps)
        return await achat_qwen(self.model_name, _to_msg_list(messages))

    def gen(self, messages: List[Message], max_tokens=None, temperature=None, num_comps=None):
        pass


@retry(wait=wait_random_exponential(min=5, max=90), stop=stop_after_attempt(RETRIES), reraise=True)
async def achat_mm(model: str, msg: List[Dict], max_tokens=None, temperature=None, top_p=None):
    client = AsyncOpenAI(api_key=MM_KEY, base_url=MM_URL)
    try:
        async with _SEMA:
            await _wait_slot()
            async with _timeout(1000):
                completion = await client.chat.completions.create(
                    model=model,
                    messages=msg,
                    max_tokens=max_tokens or MAX_TOKENS,
                    temperature=temperature if temperature is not None else TEMPERATURE,
                    top_p=top_p if top_p is not None else TOP_P,
                )
        text = completion.choices[0].message.content
        if isinstance(text, str):
            _record_usage("".join(m["content"] for m in msg), text, usage=completion.usage)
            return text
    except Exception as e:
        print(f"Error in achat_mm: {e}")
        raise


# Backwards-compatible alias.
achat_minimax = achat_mm


@retry(wait=wait_random_exponential(min=5, max=90), stop=stop_after_attempt(RETRIES), reraise=True)
def chat_mm(model: str, msg: List[Dict], max_tokens=None, temperature=None, top_p=None):
    client = OpenAI(api_key=MM_KEY, base_url=MM_URL)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=msg,
            max_tokens=max_tokens or MAX_TOKENS,
            temperature=temperature if temperature is not None else TEMPERATURE,
            top_p=top_p if top_p is not None else TOP_P,
        )
        text = completion.choices[0].message.content
        if isinstance(text, str):
            _record_usage("".join(m["content"] for m in msg), text, usage=completion.usage)
            return text
    except Exception as e:
        print(f"Error in chat_mm: {e}")
        raise


# Backwards-compatible alias.
chat_minimax = chat_mm


@LLMRegistry.register('minimax')
class MiniMaxChat(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(self, messages: List[Message], max_tokens=None, temperature=None, num_comps=None, top_p=None):
        max_tokens, temperature, num_comps = _defaults(self, max_tokens, temperature, num_comps)
        if top_p is None:
            top_p = TOP_P
        return await achat_mm(self.model_name, _to_msg_list(messages), max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    def gen(self, messages: List[Message], max_tokens=None, temperature=None, num_comps=None, top_p=None):
        max_tokens, temperature, num_comps = _defaults(self, max_tokens, temperature, num_comps)
        if top_p is None:
            top_p = TOP_P
        return chat_mm(self.model_name, _to_msg_list(messages), max_tokens=max_tokens, temperature=temperature, top_p=top_p)