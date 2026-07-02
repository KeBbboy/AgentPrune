import sys
import random
from typing import Union, Literal, List
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
