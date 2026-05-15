import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from dataset.gsm8k_dataset import gsm_get_predict


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
                "answer": gsm_get_predict(_extract_answer(raw_record)),
            }
        )
    return processed_records


class MultiArithDataset:
    def __init__(
        self,
        split: str,
        dataset_dir: str = "dataset/multiarith",
        dataset_path: str | Path | None = None,
        shuffle_seed: int = 888,
    ) -> None:
        self._split = split
        self._dataset_dir = Path(dataset_dir)
        self._dataset_path = Path(dataset_path) if dataset_path is not None else None
        self._shuffle_seed = shuffle_seed
        self._records = self._load_data()

    @staticmethod
    def get_domain() -> str:
        return "multiarith"

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, str]:
        return self._records[index]

    def __iter__(self) -> Iterator[Dict[str, str]]:
        return iter(self._records)

    def _candidate_paths(self) -> List[Path]:
        split_aliases = {
            "train": ["train", "dev"],
            "dev": ["dev", "train"],
            "val": ["val", "validation", "test"],
            "validation": ["validation", "val", "test"],
            "test": ["test", "val", "validation"],
        }
        split_names = split_aliases.get(self._split, [self._split])
        candidates: List[Path] = []
        for split_name in split_names:
            candidates.extend([
                self._dataset_dir / f"{split_name}.json",
                self._dataset_dir / f"{split_name}.jsonl",
                self._dataset_dir / f"multiarith_{split_name}.json",
                self._dataset_dir / f"multiarith_{split_name}.jsonl",
                self._dataset_dir / f"MultiArith_{split_name}.json",
                self._dataset_dir / f"MultiArith_{split_name}.jsonl",
            ])
        return candidates

    def _resolve_dataset_path(self) -> Path:
        for path in self._candidate_paths():
            if path.exists():
                return path
        if self._dataset_path is not None and self._dataset_path.exists():
            print(
                f"No split-specific file found for '{self._split}'. "
                f"Falling back to {self._dataset_path}."
            )
            return self._dataset_path
        raise FileNotFoundError(
            f"No dataset split file found for '{self._split}'. Tried: "
            + ", ".join(str(path) for path in self._candidate_paths())
        )

    def _load_data(self) -> List[Dict[str, str]]:
        records = load_multiarith_dataset(self._resolve_dataset_path())
        random.Random(self._shuffle_seed).shuffle(records)
        print("Total number of questions: ", len(records))
        return records

    @staticmethod
    def record_to_input(record: Dict[str, str]) -> Dict[str, str]:
        return {"task": record["task"]}

    def postprocess_answer(self, answer) -> str:
        if isinstance(answer, list):
            answer = answer[0] if len(answer) > 0 else ""
        return gsm_get_predict(answer)

    @staticmethod
    def record_to_target_answer(record: Dict[str, str]) -> str:
        return str(record["answer"])
