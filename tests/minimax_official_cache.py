"""Probe MiniMax-M3 on MiniMax's own official API to see whether prompt cache is reported.

Two consecutive calls with the same long prefix to maximize the chance of cache hit.
Dumps the full usage + message dict so we can see exactly which fields are returned.
"""
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai import OpenAI
from dataset.mmlu_dataset import MMLUDataset
from dataset.MMLU.download import download


def build_long_prefix(question, A, B, C, D):
    # A long system prompt + identical user prefix so the second call hits the cache.
    long_sys = (
        "You are a careful exam grader. Read the multiple-choice question below, "
        "think step by step, then reply with exactly one of the four letters A, B, C, or D. "
        "Do not add any extra text or punctuation. " * 20  # repeat to inflate tokens
    )
    user = (
        f"Answer the following multiple choice question.\n\n"
        f"Question: {question}\n"
        f"A) {A}\n"
        f"B) {B}\n"
        f"C) {C}\n"
        f"D) {D}\n\n"
        f"Reply with only the single letter (A, B, C, or D)."
    )
    return long_sys, user


def call(client, model, sys_msg, user_msg, call_idx):
    last_err = None
    completion = None
    for attempt in range(6):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=2048,
                temperature=0.0,
            )
            break
        except Exception as e:
            last_err = e
            wait_s = min(30, 5 * (attempt + 1))
            print(f"  ! attempt {attempt+1} failed: {type(e).__name__}: {str(e)[:200]} -- retry in {wait_s}s")
            time.sleep(wait_s)
    if completion is None:
        raise last_err

    raw = completion.choices[0].message.model_dump()
    u = completion.usage
    print(f"\n  --- call {call_idx} ---")
    print(f"  msg keys         = {list(raw.keys())}")
    print(f"  content          ({len(raw.get('content') or '')} chars) = {raw.get('content')!r}")
    print(f"  reasoning_content ({len(raw.get('reasoning_content') or '')} chars) = {(raw.get('reasoning_content') or '')[:120]!r}{'...' if raw.get('reasoning_content') and len(raw['reasoning_content']) > 120 else ''}")
    print(f"  usage            = {u.model_dump()}")
    return raw, u


def main():
    download()
    ds = MMLUDataset("dev")
    print(f"Loaded MMLU dev split: {len(ds)} questions\n")

    client = OpenAI(
        api_key=os.getenv("MINIMAX_API_KEY"),
        base_url=os.getenv("MINIMAX_BASE_URL"),
    )
    MODEL = "MiniMax-M3"

    row = ds[0]
    sys_msg, user_msg = build_long_prefix(
        row["question"], row["A"], row["B"], row["C"], row["D"]
    )
    print(f"========== Question 0 ==========")
    print(f"Q: {row['question'][:120]}{'...' if len(row['question']) > 120 else ''}")
    print(f"Correct: {row['correct_answer']}")
    print(f"sys prefix length (chars) = {len(sys_msg)}")

    call(client, MODEL, sys_msg, user_msg, 1)
    call(client, MODEL, sys_msg, user_msg, 2)


if __name__ == "__main__":
    main()