from types import SimpleNamespace

from AgentDropout.llm.gpt_chat import (
    MINIMAX_MODEL_NAME,
    _extract_anthropic_text,
    _minimax_request_kwargs,
    _record_minimax_usage,
)
from AgentDropout.utils.globals import CompletionTokens, Cost, PromptTokens, TokenUsageRecords


def test_minimax_request_uses_anthropic_messages_shape():
    request_kwargs, prompt_text = _minimax_request_kwargs(
        [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
        ],
        max_tokens=123,
    )

    assert request_kwargs == {
        "model": MINIMAX_MODEL_NAME,
        "max_tokens": 123,
        "system": "system prompt",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            }
        ],
    }
    assert "system prompt" in prompt_text
    assert "user: hello" in prompt_text


def test_extract_anthropic_text_ignores_thinking_blocks():
    message = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="internal reasoning"),
            SimpleNamespace(type="text", text="final answer"),
        ]
    )

    assert _extract_anthropic_text(message) == "final answer"


def test_record_minimax_usage_accumulates_per_call_tokens():
    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()
    TokenUsageRecords.instance().reset()

    _record_minimax_usage(
        "prompt",
        "response",
        SimpleNamespace(input_tokens=11, output_tokens=22),
    )

    assert PromptTokens.instance().value == 11
    assert CompletionTokens.instance().value == 22
    assert TokenUsageRecords.instance().value == [
        {
            "model": MINIMAX_MODEL_NAME,
            "input_tokens": 11,
            "output_tokens": 22,
        }
    ]
