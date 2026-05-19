import json
import random
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd


SPLIT_ALIASES = {
    "train": ["hotpot_train_v1.1", "train", "hotpot_train"],
    "dev": ["hotpot_dev_distractor_v1", "dev_distractor", "validation", "dev"],
    "validation": ["hotpot_dev_distractor_v1", "dev_distractor", "validation", "dev"],
    "val": ["hotpot_dev_distractor_v1", "dev_distractor", "validation", "dev"],
    "dev_distractor": ["hotpot_dev_distractor_v1", "dev_distractor", "validation", "dev"],
    "dev_fullwiki": ["hotpot_dev_fullwiki_v1", "dev_fullwiki"],
    "test": ["hotpot_test_fullwiki_v1", "test_fullwiki", "test"],
    "test_fullwiki": ["hotpot_test_fullwiki_v1", "test_fullwiki", "test"],
}


def normalize_answer(answer: str) -> str:
    """Official HotpotQA answer normalization."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(answer.lower())))


def hotpot_f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    zero_metric = (0.0, 0.0, 0.0)

    if normalized_prediction in {"yes", "no", "noanswer"} and normalized_prediction != normalized_ground_truth:
        return zero_metric
    if normalized_ground_truth in {"yes", "no", "noanswer"} and normalized_prediction != normalized_ground_truth:
        return zero_metric

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0 or len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return zero_metric

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def hotpot_exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def hotpot_answer_metrics(prediction: str, ground_truth: str) -> Dict[str, float]:
    f1, precision, recall = hotpot_f1_score(prediction, ground_truth)
    return {
        "em": float(hotpot_exact_match_score(prediction, ground_truth)),
        "f1": f1,
        "prec": precision,
        "recall": recall,
    }


class HotpotQADataset:
    def __init__(
        self,
        split: str,
        dataset_dir: str = "dataset/hotpotqa",
        shuffle_seed: int = 888,
        max_context_paragraphs: Optional[int] = None,
        max_sentences_per_paragraph: Optional[int] = None,
    ) -> None:
        self._split = split
        self._dataset_dir = Path(dataset_dir)
        self._shuffle_seed = shuffle_seed
        self._max_context_paragraphs = max_context_paragraphs
        self._max_sentences_per_paragraph = max_sentences_per_paragraph
        self._records = self._load_data()

    @staticmethod
    def get_domain() -> str:
        return "hotpotqa"

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._records[index]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._records)

    def _candidate_paths(self) -> List[Path]:
        split_stems = SPLIT_ALIASES.get(self._split, [self._split])
        candidates: List[Path] = []
        for stem in split_stems:
            candidates.extend([
                self._dataset_dir / f"{stem}.json",
                self._dataset_dir / f"{stem}.jsonl",
                self._dataset_dir / f"{stem}.parquet",
            ])
        return candidates

    def _resolve_dataset_path(self) -> Path:
        for path in self._candidate_paths():
            if path.exists():
                return path
        raise FileNotFoundError(
            f"No HotpotQA split file found for '{self._split}'. Tried: "
            + ", ".join(str(path) for path in self._candidate_paths())
        )

    def _load_data(self) -> List[Dict[str, Any]]:
        dataset_path = self._resolve_dataset_path()
        if dataset_path.suffix.lower() == ".json":
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_records = json.load(f)
            if isinstance(raw_records, dict):
                raw_records = raw_records.get("data") or raw_records.get("records") or []
        elif dataset_path.suffix.lower() == ".jsonl":
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_records = [json.loads(line) for line in f if line.strip()]
        elif dataset_path.suffix.lower() == ".parquet":
            raw_records = pd.read_parquet(dataset_path).to_dict("records")
        else:
            raise ValueError(f"Unsupported HotpotQA file format: {dataset_path.suffix}")

        records = [self._normalize_record(record) for record in raw_records]
        random.Random(self._shuffle_seed).shuffle(records)
        print("Total number of questions: ", len(records))
        return records

    @staticmethod
    def _tolist(value: Any) -> Any:
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    @classmethod
    def _normalize_context(cls, raw_context: Any) -> List[Dict[str, Any]]:
        raw_context = cls._tolist(raw_context)
        context: List[Dict[str, Any]] = []

        if isinstance(raw_context, dict):
            titles = cls._tolist(raw_context.get("title", []))
            sentences = cls._tolist(raw_context.get("sentences", []))
            for title, sentence_group in zip(titles, sentences):
                sentence_group = cls._tolist(sentence_group)
                context.append({
                    "title": str(title),
                    "sentences": [str(sentence) for sentence in sentence_group],
                })
            return context

        if isinstance(raw_context, list):
            for paragraph in raw_context:
                paragraph = cls._tolist(paragraph)
                if isinstance(paragraph, dict):
                    title = paragraph.get("title", "")
                    sentences = cls._tolist(paragraph.get("sentences", []))
                elif isinstance(paragraph, (list, tuple)) and len(paragraph) >= 2:
                    title = paragraph[0]
                    sentences = cls._tolist(paragraph[1])
                else:
                    continue
                context.append({
                    "title": str(title),
                    "sentences": [str(sentence) for sentence in sentences],
                })

        return context

    @classmethod
    def _normalize_supporting_facts(cls, raw_supporting_facts: Any) -> List[List[Any]]:
        raw_supporting_facts = cls._tolist(raw_supporting_facts)
        if raw_supporting_facts is None:
            return []

        if isinstance(raw_supporting_facts, dict):
            titles = cls._tolist(raw_supporting_facts.get("title", []))
            sent_ids = cls._tolist(raw_supporting_facts.get("sent_id", []))
            return [[str(title), int(sent_id)] for title, sent_id in zip(titles, sent_ids)]

        supporting_facts: List[List[Any]] = []
        if isinstance(raw_supporting_facts, list):
            for fact in raw_supporting_facts:
                fact = cls._tolist(fact)
                if isinstance(fact, (list, tuple)) and len(fact) >= 2:
                    supporting_facts.append([str(fact[0]), int(fact[1])])
        return supporting_facts

    @classmethod
    def _normalize_record(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        record_id = record.get("_id") or record.get("id")
        if record_id is None:
            record_id = str(abs(hash(json.dumps(record, sort_keys=True, default=str))))

        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise KeyError(f"Unable to extract HotpotQA question from record: {record}")

        answer = record.get("answer")
        normalized = {
            "_id": str(record_id),
            "question": question.strip(),
            "answer": None if answer is None else str(answer).strip(),
            "supporting_facts": cls._normalize_supporting_facts(record.get("supporting_facts")),
            "context": cls._normalize_context(record.get("context", [])),
            "type": str(record.get("type", "") or "").strip(),
            "level": str(record.get("level", "") or "").strip(),
        }
        return normalized

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        selected_context = context
        if self._max_context_paragraphs is not None:
            selected_context = selected_context[:self._max_context_paragraphs]

        paragraphs: List[str] = []
        for paragraph_index, paragraph in enumerate(selected_context, start=1):
            sentences = paragraph["sentences"]
            if self._max_sentences_per_paragraph is not None:
                sentences = sentences[:self._max_sentences_per_paragraph]

            sentence_lines = [
                f"{sentence_index}. {sentence}"
                for sentence_index, sentence in enumerate(sentences)
            ]
            paragraphs.append(
                f"Paragraph {paragraph_index}: {paragraph['title']}\n"
                + "\n".join(sentence_lines)
            )
        return "\n\n".join(paragraphs)

    def record_to_input(self, record: Dict[str, Any]) -> Dict[str, str]:
        task = (
            "Question:\n"
            f"{record['question']}\n\n"
            "Context:\n"
            f"{self._format_context(record['context'])}"
        )
        return {"task": task}

    @staticmethod
    def record_to_id(record: Dict[str, Any]) -> str:
        return record["_id"]

    @staticmethod
    def record_to_target_answer(record: Dict[str, Any]) -> Optional[str]:
        return record.get("answer")

    @staticmethod
    def postprocess_answer(answer: Any, record: Optional[Dict[str, Any]] = None) -> str:
        if isinstance(answer, list):
            answer = answer[0] if len(answer) > 0 else ""
        if answer is None:
            return ""
        if not isinstance(answer, str):
            answer = str(answer)

        answer = answer.strip()
        answer_match = re.search(
            r"<answer>\s*(.*?)\s*</answer>",
            answer,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if answer_match is not None:
            return HotpotQADataset._clean_prediction(answer_match.group(1))

        final_answer_match = re.search(
            r"(?:final\s+answer|answer)\s*(?:is|:)\s*(.+)$",
            answer,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if final_answer_match is not None:
            return HotpotQADataset._clean_prediction(final_answer_match.group(1))

        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        if len(lines) > 0:
            return HotpotQADataset._clean_prediction(lines[-1])
        return ""

    @staticmethod
    def _clean_prediction(answer: str) -> str:
        answer = answer.strip()
        answer = re.sub(r"^[-*\s]+", "", answer)
        answer = answer.strip().strip("\"'")
        if answer.endswith("."):
            answer = answer[:-1].strip()
        return answer

    @staticmethod
    def answer_metrics(prediction: str, ground_truth: str) -> Dict[str, float]:
        return hotpot_answer_metrics(prediction, ground_truth)
