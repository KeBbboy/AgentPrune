"""Apply the hardened SWEBenchLiteDataset.postprocess_answer to an existing
predictions.jsonl produced by run_swe_bench_lite.py and dump the rewritten file.

Usage:
    python scripts/repatch_predictions.py \
        --input  result/swe_bench_lite/direct_full/predictions.jsonl \
        --output result/swe_bench_lite/direct_full/predictions_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.swe_bench_lite_dataset import SWEBenchLiteDataset  # noqa: E402


def _looks_like_diff(patch: str) -> bool:
    # Match either real newlines or the literal '\n' that survives bad JSON dumps.
    norm = patch.replace("\\n", "\n")
    return (
        "diff --git" in norm
        and "\n--- " in norm
        and "\n+++ " in norm
        and "@@ -" in norm
    )


def _bucket(patch: str) -> str:
    if not patch:
        return "empty"
    norm = patch.replace("\\n", "\n")
    if _looks_like_diff(patch):
        first_line = norm.splitlines()[0] if norm.splitlines() else ""
        if "__model_unparsed__" in first_line:
            return "synthetic"
        return "real-diff"
    return "not-diff"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    lengths_after = []
    with args.input.open("r", encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            original = item.get("model_patch", "") or ""

            # postprocess_answer expects a single string; pass raw answer for maximal recall
            new_patch = SWEBenchLiteDataset.postprocess_answer(original)
            item["model_patch"] = new_patch

            counts[_bucket(new_patch)] += 1
            lengths_after.append(len(new_patch))
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    total = sum(counts.values())
    print(f"rewrote {total} predictions -> {args.output}")
    for k, v in counts.most_common():
        print(f"  {k:>12s}: {v} ({v / total:.1%})")
    if lengths_after:
        avg = sum(lengths_after) / len(lengths_after)
        print(f"  avg patch length: {avg:.0f} chars")


if __name__ == "__main__":
    main()
