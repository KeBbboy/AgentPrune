import aiohttp
import asyncio
from contextlib import asynccontextmanager
import time
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
import anthropic

try:
    import async_timeout
except ImportError:
    async_timeout = None

from AgentDropout.llm.format import Message
from AgentDropout.llm.price import cost_count, cost_count_llama3, cost_count_deepseek
from AgentDropout.llm.llm import LLM
from AgentDropout.llm.llm_registry import LLMRegistry
from AgentDropout.utils.globals import TokenUsageRecords


load_dotenv()

def _normalize_base_url(url: str | None) -> str | None:
    if not url:
        return None
    normalized = url.strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    return normalized


MINE_BASE_URL = _normalize_base_url(os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL"))
MINE_API_KEYS = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
MINIMAX_MODEL_NAME = "MiniMax-M3"
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or MINE_API_KEYS
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL")
MINIMAX_SYSTEM_PROMPT = "You are a helpful assistant."
LOCAL_BASE_URL = _normalize_base_url(os.getenv("LOCAL_BASE_URL") or MINE_BASE_URL) or "http://localhost:6789/v1"
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY") or MINE_API_KEYS or "API-KEY"
LLM_MAX_CONCURRENT_REQUESTS = max(1, int(os.getenv("LLM_MAX_CONCURRENT_REQUESTS", "10")))
LLM_REQUEST_INTERVAL_SECONDS = max(0.0, float(os.getenv("LLM_REQUEST_INTERVAL_SECONDS", "0")))
LLM_RETRY_ATTEMPTS = max(1, int(os.getenv("LLM_RETRY_ATTEMPTS", "8")))
LOG_MINIMAX_TOKEN_USAGE = os.getenv("LOG_MINIMAX_TOKEN_USAGE", "0").lower() in {"1", "true", "yes"}
MINIMAX_DEFAULT_MAX_TOKENS = max(1, int(os.getenv("MINIMAX_MAX_TOKENS", "2048")))
# 官方 MMLU-Pro 推理参数：temperature=0.0, top_p=0.95
MINIMAX_DEFAULT_TEMPERATURE = float(os.getenv("MINIMAX_TEMPERATURE", "0.0"))
MINIMAX_DEFAULT_TOP_P = float(os.getenv("MINIMAX_TOP_P", "0.95"))

_LLM_REQUEST_SEMAPHORE = asyncio.Semaphore(LLM_MAX_CONCURRENT_REQUESTS)
_LLM_REQUEST_LOCK = asyncio.Lock()
_LAST_LLM_REQUEST_TS = 0.0


@asynccontextmanager
async def _noop_async_timeout():
    yield


def _llm_timeout(seconds: float):
    if async_timeout is not None:
        return async_timeout.timeout(seconds)
    if hasattr(asyncio, "timeout"):
        return asyncio.timeout(seconds)
    return _noop_async_timeout()


async def _wait_for_llm_slot() -> None:
    global _LAST_LLM_REQUEST_TS
    if LLM_REQUEST_INTERVAL_SECONDS <= 0:
        return

    async with _LLM_REQUEST_LOCK:
        elapsed = time.monotonic() - _LAST_LLM_REQUEST_TS
        wait_seconds = LLM_REQUEST_INTERVAL_SECONDS - elapsed
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)
        _LAST_LLM_REQUEST_TS = time.monotonic()


def _message_value(message: Any, key: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
                elif "content" in block:
                    parts.append(_content_to_text(block["content"]))
                else:
                    parts.append(str(block))
            else:
                text = getattr(block, "text", None)
                if text is not None:
                    parts.append(str(text))
                else:
                    parts.append(str(block))
        return "\n".join(part for part in parts if part)
    return str(content)


def _anthropic_content_blocks(content: Any) -> List[Dict[str, str]]:
    text = _content_to_text(content)
    return [{"type": "text", "text": text if text else " "}]


def _append_anthropic_message(messages: List[Dict[str, Any]], role: str, content: Any) -> None:
    if role not in {"user", "assistant"}:
        role = "user"
    blocks = _anthropic_content_blocks(content)
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"].extend(blocks)
    else:
        messages.append({"role": role, "content": blocks})


def _anthropic_client_kwargs() -> Dict[str, str]:
    kwargs = {}
    if MINIMAX_API_KEY:
        kwargs["api_key"] = MINIMAX_API_KEY
    if MINIMAX_BASE_URL:
        kwargs["base_url"] = MINIMAX_BASE_URL.strip().rstrip("/")
    return kwargs


def _prepare_anthropic_messages(msg: Union[str, List[Dict]]) -> tuple[str, List[Dict[str, Any]]]:
    if isinstance(msg, str):
        msg = [{"role": "user", "content": msg}]

    system_parts = []
    messages: List[Dict[str, Any]] = []
    for item in msg:
        role = str(_message_value(item, "role", "user"))
        content = _message_value(item, "content", "")
        if role == "system":
            system_text = _content_to_text(content)
            if system_text:
                system_parts.append(system_text)
            continue
        _append_anthropic_message(messages, role, content)

    if not messages:
        messages.append({"role": "user", "content": [{"type": "text", "text": " "}]})

    system = "\n\n".join(system_parts).strip() or MINIMAX_SYSTEM_PROMPT
    return system, messages


def _anthropic_prompt_text(system: str, messages: List[Dict[str, Any]]) -> str:
    parts = [system] if system else []
    for message in messages:
        parts.append(f"{message['role']}: {_content_to_text(message.get('content', ''))}")
    return "\n\n".join(parts)


def _extract_anthropic_text(message: Any) -> str:
    text_parts = []
    for block in getattr(message, "content", []) or []:
        block_type = getattr(block, "type", None)
        if isinstance(block, dict):
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(str(block.get("text", "")))
        elif block_type == "text":
            text_parts.append(str(getattr(block, "text", "")))

    response = "\n".join(part for part in text_parts if part).strip()
    if response:
        return response
    return _content_to_text(getattr(message, "content", ""))


def _minimax_request_kwargs(
    msg: Union[str, List[Dict]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> tuple[Dict[str, Any], str]:
    system, messages = _prepare_anthropic_messages(msg)
    request_kwargs: Dict[str, Any] = {
        "model": MINIMAX_MODEL_NAME,
        "max_tokens": max_tokens or 1000,
        "system": system,
        "messages": messages,
    }
    if temperature is not None:
        request_kwargs["temperature"] = temperature
    return request_kwargs, _anthropic_prompt_text(system, messages)


def _record_minimax_usage(prompt: str, response: str, usage: Any = None) -> None:
    _, prompt_tokens, completion_tokens = cost_count(
        prompt,
        response,
        MINIMAX_MODEL_NAME,
        usage=usage,
    )
    record = {
        "model": MINIMAX_MODEL_NAME,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
    }
    TokenUsageRecords.instance().value.append(record)
    if LOG_MINIMAX_TOKEN_USAGE:
        print(
            f"{MINIMAX_MODEL_NAME} input_tokens={prompt_tokens} "
            f"output_tokens={completion_tokens}"
        )

# print(MINE_BASE_URL)


# @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
# async def achat(
#     model: str,
#     msg: List[Dict],):
#     request_url = MINE_BASE_URL
#     authorization_key = MINE_API_KEYS
#     headers = {
#         'Content-Type': 'application/json',
#         'authorization': authorization_key
#     }
#     data = {
#         "name": model,
#         "inputs": {
#             "stream": False,
#             "msg": repr(msg),
#         }
#     }
#     async with aiohttp.ClientSession() as session:
#         async with session.post(request_url, headers=headers ,json=data) as response:
#             response_data = await response.json()
#             if isinstance(response_data['data'],str):
#                 prompt = "".join([item['content'] for item in msg])
#                 cost_count(prompt,response_data['data'],model)
#                 return response_data['data']
#             else:
#                 raise Exception("api error")

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(
    model: str,
    msg: List[Dict],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
):
    request_kwargs, prompt = _minimax_request_kwargs(
        msg,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    aclient = anthropic.AsyncAnthropic(**_anthropic_client_kwargs())
    try:
        async with _llm_timeout(1000):
            completion = await aclient.messages.create(**request_kwargs)
        response_message = _extract_anthropic_text(completion)
        
        if isinstance(response_message, str):
            _record_minimax_usage(prompt, response_message, usage=completion.usage)
            return response_message

    except Exception as e:
        raise RuntimeError(f"Failed to complete the async chat request: {e}")


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
def chat(
    model: str,
    msg: List[Dict],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
):
    request_kwargs, prompt = _minimax_request_kwargs(
        msg,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    client = anthropic.Anthropic(**_anthropic_client_kwargs())
    try:
        completion = client.messages.create(**request_kwargs)
        response_message = _extract_anthropic_text(completion)

        if isinstance(response_message, str):
            _record_minimax_usage(prompt, response_message, usage=completion.usage)
            return response_message

    except Exception as e:
        raise RuntimeError(f"Failed to complete the chat request: {e}")

# @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(6))
async def achat_deepseek(model: str, msg: List[Dict],):
    model = ''
    # print(1111111)
    api_kwargs = dict(api_key = deepseek_api, base_url = deepseek_url)
    aclient = AsyncOpenAI(**api_kwargs)
    try:
        async with _llm_timeout(1000):
            completion = await aclient.chat.completions.create(model=model,messages=msg)
        # print(completion)
        response_message = completion.choices[0].message.content
        
        if isinstance(response_message, str):
            prompt = "".join([item['content'] for item in msg])
            cost_count_deepseek(prompt, response_message, model, usage=completion.usage)
            return response_message

    except Exception as e:
        raise RuntimeError(f"Failed to complete the async chat request: {e}")

# @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
@retry(wait=wait_random_exponential(min=5, max=90), stop=stop_after_attempt(LLM_RETRY_ATTEMPTS), reraise=True)
async def achat_llama(model: str, msg: List[Dict]):
    # print(111111111111)
    api_kwargs = dict(api_key = LOCAL_API_KEY, base_url = LOCAL_BASE_URL)
    aclient = AsyncOpenAI(**api_kwargs)
    try:
        async with _LLM_REQUEST_SEMAPHORE:
            await _wait_for_llm_slot()
            async with _llm_timeout(1000):
                completion = await aclient.chat.completions.create(model=model,messages=msg)
        response_message = completion.choices[0].message.content
        
        if isinstance(response_message, str):
            prompt = "".join([item['content'] for item in msg])
            cost_count_llama3(prompt, response_message, model, usage=completion.usage)
            return response_message

    except Exception as e:
        print(f"Error in achat_llama: {e}")
        raise
    

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat(
            self.model_name,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return chat(
            self.model_name,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

@LLMRegistry.register('deepseek')
class DeepseekChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat_deepseek(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass

@LLMRegistry.register('llama')
class LlamaChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name
        # print(11111111111111111111)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat_llama(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass


@retry(wait=wait_random_exponential(min=5, max=90), stop=stop_after_attempt(LLM_RETRY_ATTEMPTS), reraise=True)
async def achat_minimax(model: str, msg: List[Dict], max_tokens: Optional[int] = None, temperature: Optional[float] = None, top_p: Optional[float] = None):
    api_kwargs = dict(api_key=MINE_API_KEYS, base_url=MINE_BASE_URL)
    aclient = AsyncOpenAI(**api_kwargs)
    try:
        async with _LLM_REQUEST_SEMAPHORE:
            await _wait_for_llm_slot()
            async with _llm_timeout(1000):
                completion = await aclient.chat.completions.create(
                    model=model,
                    messages=msg,
                    max_tokens=max_tokens or MINIMAX_DEFAULT_MAX_TOKENS,
                    temperature=temperature if temperature is not None else MINIMAX_DEFAULT_TEMPERATURE,
                    top_p=top_p if top_p is not None else MINIMAX_DEFAULT_TOP_P,
                )
        response_message = completion.choices[0].message.content

        if isinstance(response_message, str):
            prompt = "".join([item['content'] for item in msg])
            _record_minimax_usage(prompt, response_message, usage=completion.usage)
            return response_message

    except Exception as e:
        print(f"Error in achat_minimax: {e}")
        raise


@retry(wait=wait_random_exponential(min=5, max=90), stop=stop_after_attempt(LLM_RETRY_ATTEMPTS), reraise=True)
def chat_minimax(model: str, msg: List[Dict], max_tokens: Optional[int] = None, temperature: Optional[float] = None, top_p: Optional[float] = None):
    api_kwargs = dict(api_key=MINE_API_KEYS, base_url=MINE_BASE_URL)
    client = AsyncOpenAI(**api_kwargs)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=msg,
            max_tokens=max_tokens or MINIMAX_DEFAULT_MAX_TOKENS,
            temperature=temperature if temperature is not None else MINIMAX_DEFAULT_TEMPERATURE,
            top_p=top_p if top_p is not None else MINIMAX_DEFAULT_TOP_P,
        )
        response_message = completion.choices[0].message.content

        if isinstance(response_message, str):
            prompt = "".join([item['content'] for item in msg])
            _record_minimax_usage(prompt, response_message, usage=completion.usage)
            return response_message

    except Exception as e:
        print(f"Error in chat_minimax: {e}")
        raise


@LLMRegistry.register('minimax')
class MiniMaxChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        top_p: Optional[float] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if top_p is None:
            top_p = MINIMAX_DEFAULT_TOP_P
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat_minimax(self.model_name, messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Union[List[str], str]:
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if top_p is None:
            top_p = MINIMAX_DEFAULT_TOP_P
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return chat_minimax(self.model_name, messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
