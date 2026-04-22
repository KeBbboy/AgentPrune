import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv
import os
from contextvars import ContextVar
from openai import OpenAI, AsyncOpenAI

from AgentPrune.llm.format import Message
from AgentPrune.llm.price import cost_count
from AgentPrune.llm.llm import LLM
from AgentPrune.llm.llm_registry import LLMRegistry
from AgentPrune.utils.globals import PromptTokens, CompletionTokens


OPENAI_API_KEYS = ['']
BASE_URL = ''

load_dotenv()
MINE_BASE_URL = os.getenv('BASE_URL')
MINE_API_KEY = os.getenv('API_KEY')
ACTIVE_USAGE_RECORDER: ContextVar[Optional[Dict[str, int]]] = ContextVar("active_usage_recorder", default=None)


def _normalize_base_url(base_url: Optional[str]) -> Optional[str]:
    if not base_url:
        return base_url
    return base_url.removesuffix('/chat/completions')


def set_active_usage_recorder(recorder: Optional[Dict[str, int]]):
    return ACTIVE_USAGE_RECORDER.set(recorder)


def reset_active_usage_recorder(token):
    ACTIVE_USAGE_RECORDER.reset(token)


@retry(wait=wait_random_exponential(max=300), stop=stop_after_attempt(3))
async def achat(
    model: str,
    msg: List[Dict],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    num_comps: Optional[int] = None,
):
    client = AsyncOpenAI(base_url = _normalize_base_url(MINE_BASE_URL), api_key = MINE_API_KEY,)
    chat_completion = await client.chat.completions.create(
        messages=msg,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        n=num_comps,
    )
    usage = chat_completion.usage
    if usage is not None:
        PromptTokens.instance().value += usage.prompt_tokens or 0
        CompletionTokens.instance().value += usage.completion_tokens or 0
        recorder = ACTIVE_USAGE_RECORDER.get()
        if recorder is not None:
            recorder["prompt_tokens"] = usage.prompt_tokens or 0
            recorder["completion_tokens"] = usage.completion_tokens or 0
    response = chat_completion.choices[0].message.content
    return response
    

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
            messages = [{'role':'user', 'content':'messages'}]
        return await achat(
            self.model_name,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            num_comps=num_comps,
        )
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass
