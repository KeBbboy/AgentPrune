from typing import Optional
from class_registry import ClassRegistry

from AgentDropout.llm.llm import LLM


def _pick_registry_key(model_name: Optional[str]) -> str:
    """Pick the LLM implementation based on the requested model name.

    Routing rules (first match wins):
      * Empty / None / ``"MiniMax-M3"`` / contains ``"minimax"`` → ``minimax`` client
        (Anthropic-compatible protocol against the MiniMax-M3 endpoint).
      * Contains ``"qwen"`` (e.g. ``"Qwen3-14B"``, ``"Qwen/Qwen3-14B"``) or
        ``"siliconflow"`` / ``"llama"`` → ``llama`` client, which dispatches to
        the OpenAI-compatible ``achat_qwen`` caller.
      * Anything else → ``minimax`` (keeps the previous default).
    """
    name = (model_name or "").strip()
    if not name or name == "MiniMax-M3" or "minimax" in name.lower():
        return "minimax"
    lowered = name.lower()
    if "qwen" in lowered or "siliconflow" in lowered or "llama" in lowered:
        return "llama"
    return "minimax"


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
        # Resolve the concrete implementation key from the requested model name,
        # then keep using the original ``model_name`` as the upstream identifier
        # so the per-client ``model_name`` field (and pricing / thinking-mode
        # switches) reflects what the user actually asked for.
        key = _pick_registry_key(model_name)
        resolved_name = model_name if model_name else "MiniMax-M3"
        return cls.registry.get(key, resolved_name)
