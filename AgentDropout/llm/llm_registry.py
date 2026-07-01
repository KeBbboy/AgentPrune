from typing import Optional
from class_registry import ClassRegistry

from AgentDropout.llm.llm import LLM


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        model_name = "MiniMax-M3"

        model = cls.registry.get('GPTChat', model_name)

        return model
