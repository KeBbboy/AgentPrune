import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Union, Literal


OPTION_LETTERS = ["A", "B", "C", "D", "E"]


class AQuADataset:
    def __init__(
        self,
        split: Union[Literal["train"], Literal["dev"], Literal["val"], Literal["test"]],
        dataset_dir: str = "dataset/aqua",
    ) -> None:
        self._split = split
        self._dataset_dir = Path(dataset_dir)
        self._records = self._load_data()

    @staticmethod
    def get_domain() -> str:
        return "aqua"

    def _candidate_paths(self) -> List[Path]:
        return [
            self._dataset_dir / f"{self._split}.json",
            self._dataset_dir / f"{self._split}.jsonl",
            self._dataset_dir / f"AQuA_{self._split}.json",
            self._dataset_dir / f"AQuA_{self._split}.jsonl",
            self._dataset_dir / f"{self._split}_data.json",
            self._dataset_dir / f"{self._split}_data.jsonl",
        ]

    def _load_data(self) -> List[Dict[str, Any]]:
        dataset_path = next((path for path in self._candidate_paths() if path.exists()), None)
        if dataset_path is None:
            tried = ", ".join(str(path) for path in self._candidate_paths())
            raise FileNotFoundError(f"No AQuA split file found for '{self._split}'. Tried: {tried}")

        if dataset_path.suffix.lower() == ".jsonl":
            with open(dataset_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
        elif dataset_path.suffix.lower() == ".json":
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = None
                for key in ("data", "examples", "items", "records"):
                    value = data.get(key)
                    if isinstance(value, list):
                        records = value
                        break
                if records is None:
                    raise ValueError(f"Unsupported JSON structure in {dataset_path}")
            else:
                raise ValueError(f"Unsupported JSON structure in {dataset_path}")
        else:
            raise ValueError(f"Unsupported AQuA file format: {dataset_path.suffix}")

        return [self._normalize_record(record) for record in records]

    @staticmethod
    def _clean_option_text(option: str) -> str:
        return re.sub(r"^\s*\(?[A-Ea-e]\)?[\.\):\-]?\s*", "", option).strip()

    @classmethod
    def _extract_options(cls, record: Dict[str, Any]) -> Dict[str, str]:
        options: Dict[str, str] = {}

        for letter in OPTION_LETTERS:
            value = record.get(letter) or record.get(letter.lower())
            if isinstance(value, str) and value.strip():
                options[letter] = cls._clean_option_text(value)
        if len(options) == len(OPTION_LETTERS):
            return options

        raw_options = record.get("options") or record.get("choices")
        if isinstance(raw_options, dict):
            for letter in OPTION_LETTERS:
                value = raw_options.get(letter) or raw_options.get(letter.lower())
                if isinstance(value, str) and value.strip():
                    options[letter] = cls._clean_option_text(value)
        elif isinstance(raw_options, list):
            for letter, value in zip(OPTION_LETTERS, raw_options):
                options[letter] = cls._clean_option_text(str(value))

        if len(options) != len(OPTION_LETTERS):
            raise KeyError(f"Unable to extract 5 options from record: {record}")
        return options

    @staticmethod
    def _extract_question(record: Dict[str, Any]) -> str:
        for key in ("question", "Question", "problem", "Problem"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise KeyError(f"Unable to extract question from record: {record}")

    @classmethod
    def _extract_correct_answer(cls, record: Dict[str, Any], options: Dict[str, str]) -> str:
        for key in ("correct", "correct_option", "answer", "Answer", "target"):
            value = record.get(key)
            if value is None:
                continue
            candidate = str(value).strip()
            if len(candidate) == 1 and candidate.upper() in OPTION_LETTERS:
                return candidate.upper()
            for letter, option_text in options.items():
                if candidate == option_text:
                    return letter
        raise KeyError(f"Unable to extract correct option from record: {record}")

    @classmethod
    def _normalize_record(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        options = cls._extract_options(record)
        question = cls._extract_question(record)
        correct_answer = cls._extract_correct_answer(record, options)
        normalized = {
            "question": question,
            "correct_answer": correct_answer,
        }
        normalized.update(options)
        return normalized

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._records[index]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._records)

    @staticmethod
    def record_to_input(record: Dict[str, Any]) -> Dict[str, Any]:
        demo_question = (
            f"{record['question']}\n"
            f"Option A: {record['A']}\n"
            f"Option B: {record['B']}\n"
            f"Option C: {record['C']}\n"
            f"Option D: {record['D']}\n"
            f"Option E: {record['E']}\n"
        )
        return {"task": demo_question}

    def postprocess_answer(self, answer) -> str:
        if isinstance(answer, list):
            answer = answer[0] if len(answer) > 0 else ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if not answer:
            return ""

        xml_match = re.search(r"<answer>\s*([ABCDE])\s*</answer>", answer, flags=re.IGNORECASE)
        if xml_match is not None:
            return xml_match.group(1).upper()

        answer_is_match = re.search(r"answer is\s*[:\-]?\s*([ABCDE])", answer, flags=re.IGNORECASE)
        if answer_is_match is not None:
            return answer_is_match.group(1).upper()

        for char in answer:
            upper_char = char.upper()
            if upper_char in OPTION_LETTERS:
                return upper_char
        return answer[:1].upper()

    @staticmethod
    def record_to_target_answer(record: Dict[str, Any]) -> str:
        correct_answer = record["correct_answer"]
        if not isinstance(correct_answer, str):
            raise TypeError(f"String expected but got {type(correct_answer)}")
        return correct_answer.upper()
