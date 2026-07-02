"""Probe Qwen/Qwen3.5-397B-A17B on SiliconFlow with thinking ON/OFF.

Verifies (1) what fields the API returns, (2) whether prompt cache is hit.
Calls the same prompt twice in each mode to maximize the chance of a cache hit.
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai import OpenAI
import time
from dataset.mmlu_dataset import MMLUDataset
from dataset.MMLU.download import download


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


def call(client, model, prompt, extra_body, call_idx):
    # Retry on 503/429 transient errors (SiliconFlow often returns 503 for large MoE models).
    last_err = None
    completion = None
    for attempt in range(6):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.0,
                extra_body=extra_body,
            )
            break
        except Exception as e:
            last_err = e
            wait_s = min(30, 5 * (attempt + 1))
            print(f"  ! attempt {attempt+1} failed: {type(e).__name__}: {str(e)[:160]} -- retry in {wait_s}s")
            time.sleep(wait_s)
    if completion is None:
        raise last_err
    raw = completion.choices[0].message.model_dump()
    u = completion.usage
    rc = raw.get("reasoning_content")
    ct = raw.get("content")
    print(f"\n  --- call {call_idx} (extra_body={extra_body}) ---")
    print(f"  msg keys         = {list(raw.keys())}")
    print(f"  reasoning_content ({len(rc or '')} chars) = {(rc or '')[:80]!r}{'...' if rc and len(rc) > 80 else ''}")
    print(f"  content           ({len(ct or '')} chars) = {ct!r}")
    print(f"  usage             = {u.model_dump()}")
    return raw, u


def main():
    download()
    ds = MMLUDataset("dev")
    print(f"Loaded MMLU dev split: {len(ds)} questions\n")

    client = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"))
    MODEL = "Qwen/Qwen3.5-397B-A17B"
    n = 1
    for i in range(n):
        row = ds[i]
        prompt = build_prompt(row["question"], row["A"], row["B"], row["C"], row["D"])
        print(f"========== Question {i} ==========")
        print(f"Q: {row['question'][:120]}{'...' if len(row['question']) > 120 else ''}")
        print(f"Correct: {row['correct_answer']}")

        for label, extra_body in [
            ("THINKING ON  (enable_thinking=True)",  {"enable_thinking": True,  "thinking_budget": 4096}),
            ("THINKING OFF (enable_thinking=False)", {"enable_thinking": False}),
        ]:
            print(f"\n=== {label} ===")
            # Two consecutive calls with the same prompt → second should hit cache if any.
            call(client, MODEL, prompt, extra_body, 1)
            call(client, MODEL, prompt, extra_body, 2)
        print()


if __name__ == "__main__":
    main()