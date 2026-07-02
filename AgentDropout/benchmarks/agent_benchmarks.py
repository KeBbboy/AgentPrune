import json
import random
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from AgentDropout.benchmarks.evaluators.metrics import (
    evaluate_gaia,
    evaluate_gpqa,
    evaluate_math,
    evaluate_musique,
    evaluate_qa,
)


QA_DATASETS = {"gaia", "hle", "musique"}
MC_DATASETS = {"gpqa"}
MATH_DATASETS = {"aime"}


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    data_subdir: str
    default_split: str
    default_tools: tuple[str, ...]
    answer_kind: str
    expected_samples: Optional[int] = None
    notes: str = ""


BENCHMARK_SPECS: Dict[str, BenchmarkSpec] = {
    "gaia": BenchmarkSpec(
        name="gaia",
        data_subdir="GAIA",
        default_split="all_validation",
        default_tools=("web_search", "code_generator", "text_inspector"),
        answer_kind="qa",
        expected_samples=165,
        notes="msc-thesis runs the GAIA all_validation file.",
    ),
    "gpqa": BenchmarkSpec(
        name="gpqa",
        data_subdir="GPQA",
        default_split="diamond",
        default_tools=("web_search", "code_generator"),
        answer_kind="multiple_choice",
        expected_samples=198,
        notes="msc-thesis config uses GPQA diamond.",
    ),
    "aime": BenchmarkSpec(
        name="aime",
        data_subdir="AIME",
        default_split="train",
        default_tools=("code_generator",),
        answer_kind="math",
        expected_samples=60,
        notes="msc-thesis downloader combines AIME 2024 and 2025.",
    ),
    "musique": BenchmarkSpec(
        name="musique",
        data_subdir="MuSiQue",
        default_split="validation_subset_200",
        default_tools=("web_search", "code_generator"),
        answer_kind="qa",
        expected_samples=200,
        notes="msc-thesis runs a 200-example answerable validation subset.",
    ),
    "hle": BenchmarkSpec(
        name="hle",
        data_subdir="HLE",
        default_split="test_subset_200",
        default_tools=("web_search", "code_generator", "text_inspector"),
        answer_kind="qa",
        expected_samples=200,
        notes="msc-thesis runs a 200-example HLE test subset.",
    ),
}


@dataclass
class BenchmarkRecord:
    question_id: Any
    question: str
    answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_answer_text(answer: Any) -> str:
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    if answer is None:
        return ""
    if not isinstance(answer, str):
        answer = str(answer)
    answer = answer.strip()

    xml_match = re.search(r"<answer>\s*(.*?)\s*</answer>", answer, flags=re.IGNORECASE | re.DOTALL)
    if xml_match is not None:
        return _clean_answer(xml_match.group(1))

    boxed_match = re.search(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", answer)
    if boxed_match is not None:
        return _clean_answer(boxed_match.group(1))

    final_match = re.search(
        r"(?:final\s+answer|answer)\s*(?:is|:)\s*(.+)$",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if final_match is not None:
        return _clean_answer(final_match.group(1))

    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    return _clean_answer(lines[-1]) if lines else ""


def extract_choice(answer: Any) -> str:
    text = extract_answer_text(answer)
    for pattern in (
        r"^\(?\s*([A-J])\s*\)?$",
        r"answer\s*(?:is|:)?\s*\(?\s*([A-J])\s*\)?",
        r"\b([A-J])\b",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is not None:
            return match.group(1).upper()
    return ""


def extract_numeric_answer(answer: Any) -> str:
    text = extract_answer_text(answer)
    if not text:
        return ""
    text = text.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match is None:
        return ""
    value = match.group(0)
    if value.endswith(".0"):
        value = value[:-2]
    return value


# AIME / MATH official final-answer markers. Order matters: the last-seen
# marker in the body wins, matching the convention used by lm-eval-harness
# MATH and `math_verify` (downstream scoring treats the most recent
# `\boxed{...}` or `#### N` as the authoritative final answer).
_OFFICIAL_HASH_ANSWER_RE = re.compile(r"####\s*(-?\d[\d,]*)(?:\.\d+)?")
_OFFICIAL_FINAL_ANSWER_RE = re.compile(
    r"(?:final\s+answer|the\s+answer|answer)\s*(?:is|:)\s*\$?\s*(-?\d[\d,]*)(?:\.\d+)?",
    re.IGNORECASE,
)
_OFFICIAL_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def _last_int_in(text: str) -> str:
    cleaned = (text or "").replace(",", "")
    for candidate in reversed(re.findall(r"-?\d+(?:\.\d+)?", cleaned)):
        if candidate.endswith(".0"):
            candidate = candidate[:-2]
        try:
            return str(int(candidate))
        except ValueError:
            continue
    return ""


def extract_official_numeric_answer(answer: Any) -> str:
    """Numerically extract the model's answer using the conventions upstream
    AIME / MATH scorers recognise.

    Priority order (last occurrence wins, to match ``math_verify``):

        1. ``#### N``  — AIME / GSM8K CoT-style terminator.
        2. ``Final Answer is N`` / ``The answer is N``.
        3. ``\\boxed{N}``  — canonical LaTeX answer wrapper.
        4. Last integer on the last non-empty line.
    """

    if answer is None:
        return ""
    raw = str(answer)

    for pattern in (_OFFICIAL_HASH_ANSWER_RE, _OFFICIAL_FINAL_ANSWER_RE):
        candidates = pattern.findall(raw)
        if candidates:
            value = _last_int_in(candidates[-1])
            if value:
                return value

    boxed_matches = _OFFICIAL_BOXED_RE.findall(raw)
    if boxed_matches:
        value = _last_int_in(boxed_matches[-1])
        if value:
            return value

    return extract_numeric_answer(answer)


def score_benchmark_prediction(
    dataset_name: str,
    prediction: Any,
    ground_truth: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = metadata or {}
    pred_text = extract_answer_text(prediction)
    if dataset_name == "gpqa":
        result = evaluate_gpqa(pred_text, ground_truth, metadata.get("choices", []))
        result["prediction"] = extract_choice(prediction) or pred_text
        result["score"] = result.get("accuracy", float(result.get("correct", False)))
        return result

    if dataset_name == "aime":
        result = evaluate_math(pred_text, ground_truth)
        # Use the robust extractor first; only fall back to the raw prediction
        # text when no integer could be salvaged from the model's output.
        result["prediction"] = extract_official_numeric_answer(prediction) or pred_text
        result["score"] = result.get("accuracy", float(result.get("correct", False)))
        return result

    if dataset_name == "musique":
        aliases = metadata.get("answer_aliases") or []
        result = evaluate_musique(pred_text, ground_truth, aliases)
    elif dataset_name == "gaia":
        result = evaluate_gaia(pred_text, ground_truth)
    elif dataset_name == "hle":
        result = evaluate_qa(pred_text, ground_truth)
    else:
        result = evaluate_qa(pred_text, ground_truth)

    result["prediction"] = pred_text
    result["score"] = result.get("accuracy", float(result.get("correct", False)))
    return result


class AgentBenchmarkDataset:
    def __init__(
        self,
        name: str,
        split: Optional[str] = None,
        data_dir: str | Path = "dataset",
        dataset_path: str | Path | None = None,
        shuffle_seed: int = 0,
    ) -> None:
        name = name.lower()
        if name not in BENCHMARK_SPECS:
            raise ValueError(f"Unsupported benchmark dataset: {name}")
        self.name = name
        self.spec = BENCHMARK_SPECS[name]
        self.split = split or self.spec.default_split
        self.data_dir = Path(data_dir)
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.shuffle_seed = shuffle_seed
        self.records = self._load_records()

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[BenchmarkRecord]:
        return iter(self.records)

    def __getitem__(self, index: int) -> BenchmarkRecord:
        return self.records[index]

    @property
    def default_tools(self) -> List[str]:
        return list(self.spec.default_tools)

    def expected_samples(self) -> Optional[int]:
        return self.spec.expected_samples

    def record_to_input(self, record: BenchmarkRecord) -> Dict[str, Any]:
        metadata = dict(record.metadata)
        payload = {
            "task": record.question,
            "question": record.question,
            "dataset_name": self.name,
            "attachments": list(metadata.get("attachments", [])),
            "metadata": metadata,
        }
        return payload

    def record_to_target_answer(self, record: BenchmarkRecord) -> str:
        return record.answer

    def postprocess_answer(self, answer: Any) -> str:
        if self.name == "gpqa":
            return extract_choice(answer)
        if self.name == "aime":
            return extract_numeric_answer(answer)
        return extract_answer_text(answer)

    def score(self, prediction: Any, record: BenchmarkRecord) -> Dict[str, Any]:
        return score_benchmark_prediction(self.name, prediction, record.answer, record.metadata)

    def _candidate_paths(self) -> List[Path]:
        candidates: List[Path] = []
        if self.dataset_path is not None:
            candidates.append(self.dataset_path)

        filenames = [f"{self.split}.jsonl"]
        if self.name == "musique":
            filenames.append("validation_subset_200.jsonl")
        if self.name == "hle" and self.split == "test":
            filenames.append("test_subset_200.jsonl")

        roots = [self.data_dir]
        if self.data_dir.name != "data":
            roots.append(Path("data"))
        if self.data_dir.name != "dataset":
            roots.append(Path("dataset"))

        seen: set[Path] = set()
        for root in roots:
            for filename in filenames:
                path = root / self.spec.data_subdir / filename
                if path not in seen:
                    candidates.append(path)
                    seen.add(path)
        return candidates

    def _resolve_path(self) -> Path:
        for path in self._candidate_paths():
            if path.exists():
                return path
        raise FileNotFoundError(
            f"No {self.name.upper()} split file found for split '{self.split}'. Tried: "
            + ", ".join(str(path) for path in self._candidate_paths())
        )

    def _load_records(self) -> List[BenchmarkRecord]:
        path = self._resolve_path()
        with open(path, "r", encoding="utf-8") as f:
            raw_records = [json.loads(line) for line in f if line.strip()]

        records = [self._normalize_record(idx, data) for idx, data in enumerate(raw_records)]
        random.Random(self.shuffle_seed).shuffle(records)
        print(f"Loaded {len(records)} {self.name.upper()} records from {path}")
        return records

    def _normalize_record(self, idx: int, data: Dict[str, Any]) -> BenchmarkRecord:
        if self.name == "gaia":
            return self._normalize_gaia(idx, data)
        if self.name == "gpqa":
            return self._normalize_gpqa(idx, data)
        if self.name == "aime":
            return self._normalize_aime(idx, data)
        if self.name == "musique":
            return self._normalize_musique(idx, data)
        if self.name == "hle":
            return self._normalize_hle(idx, data)
        raise ValueError(f"Unsupported benchmark dataset: {self.name}")

    @staticmethod
    def _attachments(data: Dict[str, Any]) -> List[str]:
        attachments = []
        full_file_path = data.get("full_file_path")
        if full_file_path:
            attachments.append(str(full_file_path))
        return attachments

    def _normalize_gaia(self, idx: int, data: Dict[str, Any]) -> BenchmarkRecord:
        metadata = {
            "level": data.get("Level", data.get("level", 1)),
            "file_name": data.get("file_name", ""),
            "file_path": data.get("file_path", ""),
            "full_file_path": data.get("full_file_path", ""),
            "attachments": self._attachments(data),
        }
        return BenchmarkRecord(
            question_id=data.get("task_id", data.get("id", idx)),
            question=str(data.get("Question", data.get("question", ""))),
            answer=str(data.get("Answer", data.get("answer", ""))),
            metadata=metadata,
        )

    def _normalize_gpqa(self, idx: int, data: Dict[str, Any]) -> BenchmarkRecord:
        choices = list(data.get("Choices") or data.get("choices") or [])
        question = str(data.get("Question", data.get("question", ""))).strip()
        if choices:
            question += "\n\nChoices:"
            for i, choice in enumerate(choices):
                question += f"\n{chr(ord('A') + i)}. {choice}"
        return BenchmarkRecord(
            question_id=data.get("id", idx),
            question=question,
            answer=str(data.get("Answer", data.get("answer", ""))).strip().upper(),
            metadata={
                "choices": choices,
                "answer_text": data.get("AnswerText", data.get("answer_text", "")),
                "domain": data.get("Domain", data.get("domain", "")),
            },
        )

    def _normalize_aime(self, idx: int, data: Dict[str, Any]) -> BenchmarkRecord:
        return BenchmarkRecord(
            question_id=data.get("ID", data.get("id", idx)),
            question=str(data.get("Question", data.get("question", data.get("problem", "")))),
            answer=str(data.get("Answer", data.get("answer", ""))),
            metadata={"year": data.get("Year", data.get("year", "")), "problem_id": data.get("ID", idx)},
        )

    def _normalize_musique(self, idx: int, data: Dict[str, Any]) -> BenchmarkRecord:
        return BenchmarkRecord(
            question_id=data.get("ID", data.get("id", idx)),
            question=str(data.get("Question", data.get("question", ""))),
            answer=str(data.get("Answer", data.get("answer", ""))),
            metadata={
                "answer_aliases": data.get("answer_aliases") or data.get("golden_answers") or [],
                "answerable": data.get("answerable", True),
            },
        )

    def _normalize_hle(self, idx: int, data: Dict[str, Any]) -> BenchmarkRecord:
        metadata = {
            "id": data.get("id", f"hle_{self.split}_{idx}"),
            "answer_type": data.get("Answer_type", data.get("answer_type", "")),
            "category": data.get("category", ""),
            "raw_subject": data.get("raw_subject", ""),
            "file_name": data.get("file_name", ""),
            "file_path": data.get("file_path", ""),
            "full_file_path": data.get("full_file_path", ""),
            "attachments": self._attachments(data),
        }
        return BenchmarkRecord(
            question_id=metadata["id"],
            question=str(data.get("Question", data.get("question", ""))),
            answer=str(data.get("Answer", data.get("answer", ""))),
            metadata=metadata,
        )


def _clean_answer(answer: str) -> str:
    answer = str(answer).strip()
    answer = re.sub(r"^[-*\s]+", "", answer)
    answer = answer.strip().strip("\"'")
    if answer.endswith("."):
        answer = answer[:-1].strip()
    return answer


def iter_batches(records: Iterable[BenchmarkRecord], batch_size: int) -> Iterator[List[BenchmarkRecord]]:
    batch: List[BenchmarkRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
