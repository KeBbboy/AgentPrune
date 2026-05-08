import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _read_records(dataset_path: str | Path) -> List[Dict[str, Any]]:
    path = Path(dataset_path)
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "examples", "items", "records"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError(f"Unsupported JSON structure in {path}")
    raise ValueError(f"Unsupported file format for MultiArith dataset: {path.suffix}")


def _first_non_empty(record: Dict[str, Any], keys: Iterable[str], default: str = "") -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
    return default


def _extract_question(record: Dict[str, Any]) -> str:
    direct_question = _first_non_empty(record, ("question", "Question", "sQuestion"))
    if direct_question:
        return direct_question

    body = _first_non_empty(record, ("body", "Body"))
    question = _first_non_empty(record, ("query", "Query", "prompt", "Prompt"))
    if body and question:
        return f"{body} {question}".strip()
    if body:
        return body
    if question:
        return question
    raise KeyError(f"Unable to find a question field in record: {record}")


def _extract_answer(record: Dict[str, Any]) -> str:
    direct_answer = _first_non_empty(record, ("answer", "Answer", "target", "Target"))
    if direct_answer:
        return direct_answer.replace(",", "")

    for key in ("lSolutions", "solutions", "solution_values", "answers"):
        value = record.get(key)
        if isinstance(value, list) and len(value) > 0:
            return str(value[0]).replace(",", "").strip()

    raise KeyError(f"Unable to find an answer field in record: {record}")


def _extract_step(record: Dict[str, Any]) -> str:
    direct_step = _first_non_empty(
        record,
        ("solution", "Solution", "rationale", "Rationale", "equation", "Equation"),
    )
    if direct_step:
        return direct_step

    for key in ("lEquations", "equations", "formulae"):
        value = record.get(key)
        if isinstance(value, list) and len(value) > 0:
            return " ; ".join(str(item).strip() for item in value if str(item).strip())

    return ""


def load_multiarith_dataset(dataset_path: str | Path) -> List[Dict[str, str]]:
    processed_records: List[Dict[str, str]] = []
    for raw_record in _read_records(dataset_path):
        processed_records.append(
            {
                "task": _extract_question(raw_record),
                "step": _extract_step(raw_record),
                "answer": _extract_answer(raw_record),
            }
        )
    return processed_records
