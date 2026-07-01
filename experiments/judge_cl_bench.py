import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()


def load_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    raise ValueError(f"Unsupported CL-bench judge input structure: {path}")


def append_jsonl(item: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            completed.add(str(get_task_id(item)))
    return completed


def get_task_id(item: Dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    return str(metadata.get("task_id") or item.get("idx") or item.get("id"))


def model_output_from(item: Dict[str, Any]) -> str:
    output = item.get("model_output")
    if output is None:
        output = item.get("response")
    if output is None:
        output = item.get("prediction")
    return "" if output is None else str(output)


def build_rubrics_text(rubrics: Any) -> str:
    if not rubrics:
        return "No specific rubrics provided."

    lines = []
    for i, rubric in enumerate(rubrics, 1):
        if isinstance(rubric, dict):
            criteria = str(rubric.get("rubric_criteria") or rubric.get("criteria") or rubric).strip()
        else:
            criteria = str(rubric).strip()
        if criteria:
            lines.append(f"{i}. {criteria}")
    return "\n".join(lines) if lines else "No specific rubrics provided."


def build_grading_prompt(rubrics_text: str, model_output: str) -> str:
    return (
        "Starting now, you are a rigorous instruction-following grading teacher. Your task is to accurately grade and score student answers based on the 【Rubrics】.\n\n"
        "Grading Criteria\n"
        "This is a strict, all-or-nothing grading system. The final score is binary.\n"
        "To receive a score of 1, the student's answer must perfectly satisfy every single requirement listed in the 【Rubrics】.\n"
        "If even one requirement is not fully met, the final score will be 0.\n"
        "Grading Process\n"
        "Please strictly follow the steps below for analysis—no steps may be skipped:\n"
        "Step 1: Analyze the Standard Answer\n"
        "List all explicit requirements in the 【Rubrics】 item by item (including format, content, quantity, order, etc.).\n"
        "Identify implicit requirements in the 【Rubrics】 (e.g., language style, logical structure).\n"
        "Define specific evaluation criteria for each requirement (e.g., \"must include X,\" \"must not exceed Y\").\n"
        "Step 2: Check Each Requirement Against the Student's Answer\n"
        "For every requirement in the 【Rubrics】, verify one by one whether the student's answer fully satisfies it.\n"
        "Step 3: Self-Reflection\n"
        "Before giving the final score, you must conduct the following checks:\n"
        "  Completeness Check: Whether all requirements in the standard answer have been reviewed with no omissions.\n"
        "  Strictness Check: Whether the evaluation strictly adheres to the \"fully satisfied\" standard without relaxing requirements due to subjective judgment.\n"
        "  Consistency Check: Whether the grading rationale aligns logically with the final score.\n"
        "  Objectivity Check: Whether judgments are based on objective facts rather than subjective speculation.\n"
        "Output Format Requirements\n"
        "【Grading Rationale】: xxx\n"
        "【List of Requirement Satisfaction Status】: [x1, x2, ..., xn] (where n is the total number of requirements in the 【Rubrics】, and xi is \"yes\" or \"no\")\n"
        "【Overall Score】: x points (x is an integer, either 0 or 1.)\n\n"
        "Content to Be Graded\n"
        f"【Rubrics】:\n{rubrics_text}\n"
        f"【Student Response】:\n{model_output}\n"
        "\nPlease strictly output ONLY the following JSON format (do not output any other content):\n"
        "{\n"
        '  "Grading Rationale": "Your detailed grading rationale",\n'
        '  "List of Requirement Satisfaction Status": ["yes", "no", "..."],\n'
        '  "Overall Score": 0 or 1\n'
        "}\n"
    )


def strip_json_wrapper(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def parse_judge_json(text: str) -> Dict[str, Any]:
    cleaned = strip_json_wrapper(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def call_judge_api(
    client: OpenAI,
    model: str,
    rubrics_text: str,
    model_output: str,
    reasoning_effort: Optional[str],
    max_retries: int,
    retry_delay: float,
) -> Dict[str, Any]:
    messages = [{"role": "user", "content": build_grading_prompt(rubrics_text, model_output)}]
    last_error = None
    for attempt in range(max_retries):
        try:
            kwargs = {"model": model, "messages": messages}
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
            response = client.chat.completions.create(**kwargs)
            result_text = response.choices[0].message.content or ""
            result_json = parse_judge_json(result_text)
            if "Overall Score" not in result_json:
                raise ValueError("Missing 'Overall Score' field")
            return result_json
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    raise RuntimeError(str(last_error))


def normalize_score(value: Any) -> int:
    try:
        return 1 if int(value) == 1 else 0
    except Exception:
        return 0


def judge_one(args_tuple):
    item, client, args = args_tuple
    task_id = get_task_id(item)
    model_output = model_output_from(item)
    rubrics = item.get("rubrics", [])
    if not model_output.strip():
        result = {
            **item,
            "idx": task_id,
            "grading_rationale": "No model output (counted as score 0)",
            "requirement_status": [],
            "score": 0,
        }
        return task_id, result, None

    try:
        judge_json = call_judge_api(
            client=client,
            model=args.judge_model,
            rubrics_text=build_rubrics_text(rubrics),
            model_output=model_output,
            reasoning_effort=args.reasoning_effort,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )
        result = {
            **item,
            "idx": task_id,
            "model_output": model_output,
            "grading_rationale": judge_json.get("Grading Rationale", ""),
            "requirement_status": judge_json.get("List of Requirement Satisfaction Status", []),
            "score": normalize_score(judge_json.get("Overall Score", 0)),
            "judge_model": args.judge_model,
        }
        return task_id, result, None
    except Exception as exc:
        result = {
            **item,
            "idx": task_id,
            "model_output": model_output,
            "grading_rationale": f"Judge failed: {exc}",
            "requirement_status": [],
            "score": 0,
            "judge_model": args.judge_model,
        }
        return task_id, result, str(exc)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def print_summary(output_path: Path) -> Dict[str, Any]:
    graded = load_jsonl(output_path)
    total = len(graded)
    score_sum = sum(normalize_score(item.get("score", 0)) for item in graded)
    summary = {
        "total": total,
        "score_sum": score_sum,
        "average_score": score_sum / total if total else None,
        "output_file": str(output_path),
    }
    print(f"Summary: {summary}")
    return summary


def normalize_base_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    normalized = url.strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    return normalized


def main():
    parser = argparse.ArgumentParser(description="Judge CL-bench model outputs with rubric-based binary grading.")
    parser.add_argument("--input", type=str, required=True, help="Input JSON/JSONL from run_cl_bench.py or official infer.py.")
    parser.add_argument("--output", type=str, default=None, help="Output graded JSONL path.")
    parser.add_argument("--judge_model", type=str, default=os.getenv("JUDGE_MODEL", "deepseek-chat"))
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_delay", type=float, default=3.0)
    parser.add_argument("--reasoning_effort", type=str, default=None, choices=["low", "medium", "high"])
    parser.add_argument("--limit", type=int, default=0, help="0 means grade all pending records.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_graded.jsonl")
    records = load_records(input_path)
    if args.limit:
        records = records[: args.limit]

    completed = load_completed_ids(output_path)
    pending = [item for item in records if get_task_id(item) not in completed]
    print(f"Input records: {len(records)}")
    print(f"Completed records: {len(completed)}")
    print(f"Pending records: {len(pending)}")

    base_url = normalize_base_url(args.base_url or os.getenv("JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL"))
    api_key = args.api_key or os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise ValueError("Missing judge API key. Set --api_key, JUDGE_API_KEY, OPENAI_API_KEY, or API_KEY.")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    if args.workers == 1:
        for item in tqdm(pending, desc="Judging"):
            task_id, result, error = judge_one((item, client, args))
            append_jsonl(result, output_path)
            if error:
                print(f"[{task_id}] judge error: {error}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(judge_one, (item, client, args)): get_task_id(item) for item in pending}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Judging"):
                task_id, result, error = future.result()
                append_jsonl(result, output_path)
                if error:
                    print(f"[{task_id}] judge error: {error}")

    print_summary(output_path)


if __name__ == "__main__":
    main()
