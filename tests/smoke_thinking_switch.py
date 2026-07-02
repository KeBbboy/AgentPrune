"""Smoke test: verify the new thinking-mode switches reach both providers."""
import os
import sys
import asyncio
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AgentDropout.llm.gpt_chat import (
    _thinking_extra_body,
    configure_thinking,
    LLM_ENABLE_THINKING,
    LLM_THINKING_BUDGET,
    MINIMAX_THINKING_TYPE,
)

print("=== arg-style default config ===")
print(f"  LLM_ENABLE_THINKING  = {LLM_ENABLE_THINKING}")
print(f"  LLM_THINKING_BUDGET  = {LLM_THINKING_BUDGET}")
print(f"  MINIMAX_THINKING_TYPE= {MINIMAX_THINKING_TYPE}")

print("\n=== _thinking_extra_body() outputs ===")
for m in ["MiniMax-M3", "Qwen/Qwen3-14B", "gpt-4o-mini", "Qwen2-72B-Instruct"]:
    print(f"  {m!r:30s} -> {_thinking_extra_body(m)}")


async def call_real(label, model, **extra):
    extra_body = _thinking_extra_body(model)
    if "qwen" in model.lower():
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"))
    else:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=os.getenv("MINIMAX_API_KEY"),
            base_url=os.getenv("MINIMAX_BASE_URL"),
        )
    completion = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What is 2 + 2? Reply in one word."}],
        max_tokens=128,
        temperature=0.0,
        extra_body=extra_body,
    )
    raw = completion.choices[0].message.model_dump()
    rc = raw.get("reasoning_content")
    ct = raw.get("content")
    print(f"\n=== {label} ({model}) ===")
    print(f"  extra_body sent     = {extra_body}")
    print(f"  reasoning_content   = {(rc or '')[:120]!r}")
    print(f"  content             = {ct!r}")


async def main():
    # Default (ON).
    configure_thinking(enable_thinking=True, thinking_budget=8192, minimax_thinking_type="adaptive")
    await call_real("ON (default)", "Qwen/Qwen3-14B")
    # Forced OFF via the same in-process config path used by argparse runners.
    configure_thinking(enable_thinking=False, thinking_budget=8192, minimax_thinking_type="adaptive")
    print(f"\n  (configure_thinking(enable_thinking=False) -> helper returns {_thinking_extra_body('Qwen/Qwen3-14B')})")
    await call_real("OFF (arg override)", "Qwen/Qwen3-14B")

    configure_thinking(enable_thinking=True, thinking_budget=8192, minimax_thinking_type="adaptive")
    await call_real("ON  (MiniMax-M3)", "MiniMax-M3")


asyncio.run(main())
