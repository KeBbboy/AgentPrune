from AgentDropout.utils.globals import Cost, PromptTokens, CompletionTokens, CachedTokens, ReasoningTokens, BilledPromptTokens, BilledCompletionTokens, Tokenizer, Deepseek_Tokenizer
import tiktoken

# GPT-4:  https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
# GPT3.5: https://platform.openai.com/docs/models/gpt-3-5
# DALL-E: https://openai.com/pricing

def cal_token(model:str, text:str):
    try:
        encoder = tiktoken.encoding_for_model(model)
    except KeyError:
        encoder = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoder.encode(text))
    return num_tokens

def cal_token_llama3(tokenizer, text:str):
    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    # encoder = tiktoken.encoding_for_model(model)
    num_tokens = len(tokenizer(text, return_tensors='pt')['input_ids'][0])
    return num_tokens

def _usage_value(usage, names):
    if usage is None:
        return None
    for name in names:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if value is not None:
            return value
    return None

def _usage_int(usage, names):
    value = _usage_value(usage, names)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _usage_token_counts(usage):
    prompt_len = _usage_int(usage, ("prompt_tokens", "input_tokens"))
    completion_len = _usage_int(usage, ("completion_tokens", "output_tokens"))
    total_len = _usage_int(usage, ("total_tokens",))
    if completion_len is None and prompt_len is not None and total_len is not None:
        completion_len = max(total_len - prompt_len, 0)
    
    prompt_tokens_details = _usage_value(usage, ("prompt_tokens_details",))
    cached_tokens = None
    if prompt_tokens_details is not None:
        if isinstance(prompt_tokens_details, dict):
            cached_tokens = prompt_tokens_details.get("cached_tokens")
        else:
            cached_tokens = getattr(prompt_tokens_details, "cached_tokens", None)
    if cached_tokens is not None:
        cached_tokens = int(cached_tokens)
    
    completion_tokens_details = _usage_value(usage, ("completion_tokens_details",))
    reasoning_tokens = None
    if completion_tokens_details is not None:
        if isinstance(completion_tokens_details, dict):
            reasoning_tokens = completion_tokens_details.get("reasoning_tokens")
        else:
            reasoning_tokens = getattr(completion_tokens_details, "reasoning_tokens", None)
    if reasoning_tokens is not None:
        reasoning_tokens = int(reasoning_tokens)
    
    return prompt_len, completion_len, cached_tokens, reasoning_tokens

def cost_count(prompt, response, model_name, usage=None):
    branch: str
    prompt_len: int
    completion_len: int
    price: float

    usage_prompt_len, usage_completion_len, cached_tokens, reasoning_tokens = _usage_token_counts(usage)
    if usage_prompt_len is None or usage_completion_len is None:
        print(f"Warning: cost_count received incomplete usage for {model_name}: prompt={usage_prompt_len}, completion={usage_completion_len}")
    prompt_len = usage_prompt_len or 0
    completion_len = usage_completion_len or 0
    cached_tokens = cached_tokens or 0
    reasoning_tokens = reasoning_tokens or 0

    billed_prompt = prompt_len - cached_tokens
    billed_completion = completion_len

    if "gpt-4" in model_name:
        branch = "gpt-4"
        model_info = OPENAI_MODEL_INFO[branch].get(model_name)
        price = 0.0 if model_info is None else (
            billed_prompt * model_info["input"] /1000 +
            billed_completion * model_info["output"] /1000
        )
    elif "gpt-3.5" in model_name:
        branch = "gpt-3.5"
        model_info = OPENAI_MODEL_INFO[branch].get(model_name)
        price = 0.0 if model_info is None else (
            billed_prompt * model_info["input"] /1000 +
            billed_completion * model_info["output"] /1000
        )
    elif "minimax" in model_name.lower() or "MiniMax" in model_name:
        branch = "minimax"
        pricing = MODEL_PRICING["minimax-m3"]
        non_cached_prompt = billed_prompt
        price = (non_cached_prompt * pricing["input"] + 
                 cached_tokens * pricing["cache_read"] + 
                 billed_completion * pricing["output"]) / 1000000
    elif "dall-e" in model_name:
        branch = "dall-e"
        price = 0.0
        prompt_len = 0
        completion_len = 0
        billed_prompt = 0
        billed_completion = 0
    else:
        branch = "other"
        price = 0.0

    Cost.instance().value += price
    PromptTokens.instance().value += prompt_len
    CompletionTokens.instance().value += completion_len
    CachedTokens.instance().value += cached_tokens
    ReasoningTokens.instance().value += reasoning_tokens
    BilledPromptTokens.instance().value += billed_prompt
    BilledCompletionTokens.instance().value += billed_completion

    # print(f"Prompt Tokens: {prompt_len}, Completion Tokens: {completion_len}")
    return price, prompt_len, completion_len

def cost_count_llama3(prompt, response, model, usage=None):
    branch: str
    prompt_len: int
    completion_len: int
    price: float

    usage_prompt_len, usage_completion_len, cached_tokens, reasoning_tokens = _usage_token_counts(usage)
    if usage_prompt_len is None or usage_completion_len is None:
        print(f"Warning: cost_count_llama3 received incomplete usage for {model}: prompt={usage_prompt_len}, completion={usage_completion_len}")
    prompt_len = usage_prompt_len or 0
    completion_len = usage_completion_len or 0
    cached_tokens = cached_tokens or 0
    reasoning_tokens = reasoning_tokens or 0
    price = 0

    billed_prompt = prompt_len - cached_tokens
    billed_completion = completion_len

    model_lower = model.lower()
    if "qwen3-14b" in model_lower:
        pricing = MODEL_PRICING["qwen3-14b"]
        price = (billed_prompt * pricing["input"] + 
                 billed_completion * pricing["output"]) / 1000000
    elif "qwen3-32b" in model_lower:
        pricing = MODEL_PRICING["qwen3-32b"]
        price = (billed_prompt * pricing["input"] + 
                 billed_completion * pricing["output"]) / 1000000

    Cost.instance().value += price
    PromptTokens.instance().value += prompt_len
    CompletionTokens.instance().value += completion_len
    CachedTokens.instance().value += cached_tokens
    ReasoningTokens.instance().value += reasoning_tokens
    BilledPromptTokens.instance().value += billed_prompt
    BilledCompletionTokens.instance().value += billed_completion

    # print(f"Prompt Tokens: {prompt_len}, Completion Tokens: {completion_len}")
    return price, prompt_len, completion_len

def cost_count_deepseek(prompt, response, model, usage=None):
    branch: str
    prompt_len: int
    completion_len: int
    price: float

    usage_prompt_len, usage_completion_len, cached_tokens, reasoning_tokens = _usage_token_counts(usage)
    if usage_prompt_len is None or usage_completion_len is None:
        print(f"Warning: cost_count_deepseek received incomplete usage for {model}: prompt={usage_prompt_len}, completion={usage_completion_len}")
    prompt_len = usage_prompt_len or 0
    completion_len = usage_completion_len or 0
    cached_tokens = cached_tokens or 0
    reasoning_tokens = reasoning_tokens or 0
    price = 0

    billed_prompt = prompt_len - cached_tokens
    billed_completion = completion_len

    pricing = MODEL_PRICING["deepseek"]
    price = (billed_prompt * pricing["input"] + 
             billed_completion * pricing["output"]) / 1000000
    Cost.instance().value += price
    PromptTokens.instance().value += prompt_len
    CompletionTokens.instance().value += completion_len
    CachedTokens.instance().value += cached_tokens
    ReasoningTokens.instance().value += reasoning_tokens
    BilledPromptTokens.instance().value += billed_prompt
    BilledCompletionTokens.instance().value += billed_completion

    # print(f"Prompt Tokens: {prompt_len}, Completion Tokens: {completion_len}")
    return price, prompt_len, completion_len

OPENAI_MODEL_INFO ={
    "gpt-4": {
        "current_recommended": "gpt-4-1106-preview",
        "gpt-4-0125-preview": {
            "context window": 128000, 
            "training": "Jan 2024", 
            "input": 0.01, 
            "output": 0.03
        },      
        "gpt-4-1106-preview": {
            "context window": 128000, 
            "training": "Apr 2023", 
            "input": 0.01, 
            "output": 0.03
        },
        "gpt-4-vision-preview": {
            "context window": 128000, 
            "training": "Apr 2023", 
            "input": 0.01, 
            "output": 0.03
        },
        "gpt-4": {
            "context window": 8192, 
            "training": "Sep 2021", 
            "input": 0.03, 
            "output": 0.06
        },
        "gpt-4-0314": {
            "context window": 8192, 
            "training": "Sep 2021", 
            "input": 0.03, 
            "output": 0.06
        },
        "gpt-4-32k": {
            "context window": 32768, 
            "training": "Sep 2021", 
            "input": 0.06, 
            "output": 0.12
        },
        "gpt-4-32k-0314": {
            "context window": 32768, 
            "training": "Sep 2021", 
            "input": 0.06, 
            "output": 0.12
        },
        "gpt-4-0613": {
            "context window": 8192, 
            "training": "Sep 2021", 
            "input": 0.06, 
            "output": 0.12
        },
        "gpt-4o": {
            "context window": 128000, 
            "training": "Jan 2024", 
            "input": 0.005, 
            "output": 0.015
        }, 
    },
    "gpt-3.5": {
        "current_recommended": "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0125": {
            "context window": 16385, 
            "training": "Jan 2024", 
            "input": 0.0010, 
            "output": 0.0020
        },
        "gpt-3.5-turbo-1106": {
            "context window": 16385, 
            "training": "Sep 2021", 
            "input": 0.0010, 
            "output": 0.0020
        },
        "gpt-3.5-turbo-instruct": {
            "context window": 4096, 
            "training": "Sep 2021", 
            "input": 0.0015, 
            "output": 0.0020
        },
        "gpt-3.5-turbo": {
            "context window": 4096, 
            "training": "Sep 2021", 
            "input": 0.0015, 
            "output": 0.0020
        },
        "gpt-3.5-turbo-0301": {
            "context window": 4096, 
            "training": "Sep 2021", 
            "input": 0.0015, 
            "output": 0.0020
        },
        "gpt-3.5-turbo-0613": {
            "context window": 16384, 
            "training": "Sep 2021", 
            "input": 0.0015, 
            "output": 0.0020
        },
        "gpt-3.5-turbo-16k-0613": {
            "context window": 16384, 
            "training": "Sep 2021", 
            "input": 0.0015, 
            "output": 0.0020
        }
    },
    "dall-e": {
        "current_recommended": "dall-e-3",
        "dall-e-3": {
            "release": "Nov 2023",
            "standard": {
                "1024×1024": 0.040,
                "1024×1792": 0.080,
                "1792×1024": 0.080
            },
            "hd": {
                "1024×1024": 0.080,
                "1024×1792": 0.120,
                "1792×1024": 0.120
            }
        },
        "dall-e-2": {
            "release": "Nov 2022",
            "1024×1024": 0.020,
            "512×512": 0.018,
            "256×256": 0.016
        }
    }
}

MODEL_PRICING = {
    "minimax-m3": {
        "input": 2.10,
        "output": 8.40,
        "cache_read": 0.42,
        "unit": "元/百万tokens"
    },
    "qwen3-14b": {
        "input": 0.50,
        "output": 2.00,
        "cache_read": 0,
        "unit": "元/百万tokens"
    },
    "qwen3-32b": {
        "input": 1.00,
        "output": 4.00,
        "cache_read": 0,
        "unit": "元/百万tokens"
    },
    "deepseek": {
        "input": 2.00,
        "output": 8.00,
        "cache_read": 0,
        "unit": "元/百万tokens"
    }
}
