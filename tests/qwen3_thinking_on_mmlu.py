"""Probe Qwen3-14B on real MMLU questions and dump the API response fields.

Loads the MMLU dev split, builds the same prompt the AnalyzeAgent uses, and
calls SiliconFlow with thinking ON. Prints reasoning_content and content for
each of the first few questions so we can see the raw API surface.
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai import OpenAI
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


def main():
    download()
    ds = MMLUDataset("dev")
    print(f"Loaded MMLU dev split: {len(ds)} questions\n")

    client = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"))
    MODEL = "Qwen/Qwen3-14B"
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
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.0,
                extra_body=extra_body,
            )
            raw = completion.choices[0].message.model_dump()
            u = completion.usage
            rc = raw.get("reasoning_content")
            ct = raw.get("content")
            rt = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None)

            print(f"\n  --- {label} ---")
            print(f"  extra_body sent = {extra_body}")
            print(f"  API fields (msg keys) = {list(raw.keys())}")
            print(f"  reasoning_content ({len(rc or '')} chars) = {rc!r}")
            print(f"  content           ({len(ct or '')} chars) = {ct!r}")
            print(f"  usage: prompt={u.prompt_tokens} completion={u.completion_tokens} "
                  f"total={u.total_tokens} reasoning_tokens={rt}")
            print(f"  usage FULL     = {u.model_dump()}")
            print(f"  message FULL   = {raw}")
        print()


if __name__ == "__main__":
    main()