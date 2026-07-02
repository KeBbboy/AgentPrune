"""Probe MiniMax-M3 thinking-mode toggle.

Calls the MiniMax-M3 endpoint with thinking.type=adaptive (on, the default) and
thinking.type=disabled (off), using the same questions the MMLU AnalyzeAgent
would see. Prints the full content field, the extracted reasoning content, and
the usage breakdown so we can verify both modes behave as documented.
"""
import os
import sys
import re
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai import OpenAI
from dataset.mmlu_dataset import MMLUDataset
from dataset.MMLU.download import download


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_thinking(text: str):
    """Return (thinking, answer) extracted from a content field that may
    contain a single <think>...</think> block. If no block is present,
    thinking is None and the full text is treated as the answer."""
    if not text:
        return None, text
    m = THINK_RE.search(text)
    if not m:
        return None, text
    thinking = m.group(1).strip()
    answer = (text[: m.start()] + text[m.end():]).strip()
    return thinking, answer


def build_prompt(question, A, B, C, D):
    return (
        f"Answer the following multiple choice question.\n\n"
        f"Question: {question}\n"
        f"A) {A}\n"
        f"B) {B}\n"
        f"C) {C}\n"
        f"D) {D}\n\n"
        f"Reply with only the single letter (A, B, C, or D)."
    )


def call(client, model, prompt, thinking_type):
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.0,
        extra_body={"thinking": {"type": thinking_type}},
    )
    raw = completion.choices[0].message.model_dump()
    u = completion.usage
    content = raw.get("content") or ""
    thinking, answer = split_thinking(content)
    rt = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None)
    return {
        "raw_keys": list(raw.keys()),
        "message_full": raw,
        "content": content,
        "thinking": thinking,
        "answer": answer,
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.total_tokens,
        "reasoning_tokens": rt,
        "usage_full": u.model_dump(),
    }


def main():
    download()
    ds = MMLUDataset("dev")
    print(f"Loaded MMLU dev split: {len(ds)} questions\n")

    client = OpenAI(
        api_key=os.getenv("MINIMAX_API_KEY"),
        base_url=os.getenv("MINIMAX_BASE_URL"),
    )
    MODEL = "MiniMax-M3"
    n = 1
    for i in range(n):
        row = ds[i]
        prompt = build_prompt(row["question"], row["A"], row["B"], row["C"], row["D"])
        print(f"========== Question {i} ==========")
        print(f"Q: {row['question'][:120]}{'...' if len(row['question']) > 120 else ''}")
        print(f"Correct: {row['correct_answer']}")

        for label, mode in [("THINKING ON  (adaptive)", "adaptive"),
                             ("THINKING OFF (disabled)", "disabled")]:
            r = call(client, MODEL, prompt, mode)
            print(f"\n  --- {label} ---")
            print(f"  msg keys: {r['raw_keys']}")
            print(f"  content ({len(r['content'])} chars) = {r['content']!r}")
            if r["thinking"]:
                print(f"  extracted thinking ({len(r['thinking'])} chars):")
                print(f"    {r['thinking'][:300]}{'...' if len(r['thinking']) > 300 else ''}")
            else:
                print(f"  extracted thinking: <none>")
            print(f"  final answer: {r['answer']!r}")
            print(f"  usage: prompt={r['prompt_tokens']} completion={r['completion_tokens']} "
                  f"total={r['total_tokens']} reasoning_tokens={r['reasoning_tokens']}")
            print(f"  usage FULL     = {r['usage_full']}")
            print(f"  message FULL   = {r['message_full']}")
        print()


if __name__ == "__main__":
    main()