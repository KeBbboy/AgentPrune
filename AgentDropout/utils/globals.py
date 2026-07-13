import sys
import random
import contextvars
from typing import Any, Dict, Union, Literal, List
from transformers import AutoTokenizer

class Singleton:
    _instance = None

    @classmethod
    def instance(cls,*args, **kwargs):
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance
    
    def reset(self):
        self.value = 0.0

class Cost(Singleton):
    def __init__(self):
        self.value = 0.0

class PromptTokens(Singleton):
    def __init__(self):
        self.value = 0.0

class CompletionTokens(Singleton):
    def __init__(self):
        self.value = 0.0

class CachedTokens(Singleton):
    def __init__(self):
        self.value = 0.0

class ReasoningTokens(Singleton):
    def __init__(self):
        self.value = 0.0

class BilledPromptTokens(Singleton):
    def __init__(self):
        self.value = 0.0

class BilledCompletionTokens(Singleton):
    def __init__(self):
        self.value = 0.0

class TokenUsageRecords(Singleton):
    def __init__(self):
        self.value = []

    def reset(self):
        self.value = []


TokenUsageContext = contextvars.ContextVar("token_usage_context", default={})


def set_token_usage_context(context: Dict[str, Any]):
    """设置当前异步任务的 token 归因上下文。"""
    return TokenUsageContext.set(dict(context))


def reset_token_usage_context(token) -> None:
    """恢复上一个 token 归因上下文。"""
    TokenUsageContext.reset(token)


def get_token_usage_context() -> Dict[str, Any]:
    """读取当前异步任务的 token 归因上下文。"""
    value = TokenUsageContext.get()
    return dict(value) if isinstance(value, dict) else {}

class Time(Singleton):
    def __init__(self):
        self.value = ""

class Mode(Singleton):
    def __init__(self):
        self.value = ""

class Tokenizer(Singleton):
    def __init__(self, model):
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        self.value = ""

class Deepseek_Tokenizer(Singleton):
    def __init__(self, model):
        self.tokenizer = AutoTokenizer.from_pretrained('/path/to/deepseek_v3_tokenizer', use_fast=True)
        self.value = ""


def print_token_stats(prefix: str = "") -> None:
    """Print all token usage statistics collected so far.

    Fields printed:
        Cost, PromptTokens, CompletionTokens, CachedTokens, ReasoningTokens,
        BilledPromptTokens, BilledCompletionTokens.

    Args:
        prefix: Optional label prepended to each line (e.g. "Train batch", "Final").
    """
    p = f"{prefix} " if prefix else ""
    print(f"{p}Cost {Cost.instance().value}")
    print(f"{p}PromptTokens {PromptTokens.instance().value}")
    print(f"{p}CompletionTokens {CompletionTokens.instance().value}")
    print(f"{p}CachedTokens {CachedTokens.instance().value}")
    print(f"{p}ReasoningTokens {ReasoningTokens.instance().value}")
    print(f"{p}BilledPromptTokens {BilledPromptTokens.instance().value}")
    print(f"{p}BilledCompletionTokens {BilledCompletionTokens.instance().value}")
