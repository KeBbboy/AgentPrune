import json
import re
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
import pandas as pd


OPTION_LETTERS = list("ABCDEFGHIJ")
SPLIT_ALIASES = {
    "dev": ["dev", "validation", "train"],
    "val": ["val", "test", "validation"],
    "validation": ["validation", "dev"],
    "test": ["test", "val"],
    "train": ["train", "dev", "validation"],
}


class MMLUProDataset:
    def __init__(
        self,
        split: str,
        dataset_dir: str = "dataset/mmlu_pro",
        shuffle_seed: int = 888,
    ) -> None:
        self._split = split
        self._dataset_dir = Path(dataset_dir)
        self._shuffle_seed = shuffle_seed
        self._records = self._load_data()

    @staticmethod
    def get_domain() -> str:
        return "mmlu_pro"

    def _candidate_paths(self) -> List[Path]:
        split_names = SPLIT_ALIASES.get(self._split, [self._split])
        candidates: List[Path] = []
        for split_name in split_names:
            candidates.extend([
                self._dataset_dir / f"{split_name}.json",
                self._dataset_dir / f"{split_name}.jsonl",
                self._dataset_dir / f"{split_name}.parquet",
                self._dataset_dir / f"MMLU-Pro_{split_name}.json",
                self._dataset_dir / f"MMLU-Pro_{split_name}.jsonl",
                self._dataset_dir / f"MMLU-Pro_{split_name}.parquet",
                self._dataset_dir / f"mmlu_pro_{split_name}.json",
                self._dataset_dir / f"mmlu_pro_{split_name}.jsonl",
                self._dataset_dir / f"mmlu_pro_{split_name}.parquet",
            ])
        return candidates

    def _load_data(self) -> List[Dict[str, Any]]:
        dataset_path = next((path for path in self._candidate_paths() if path.exists()), None)
        if dataset_path is None:
            tried = ", ".join(str(path) for path in self._candidate_paths())
            raise FileNotFoundError(f"No MMLU-Pro split file found for '{self._split}'. Tried: {tried}")

        if dataset_path.suffix.lower() == ".jsonl":
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_records = [json.loads(line) for line in f if line.strip()]
        elif dataset_path.suffix.lower() == ".json":
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                raw_records = data
            elif isinstance(data, dict):
                raw_records = None
                for key in ("data", "examples", "items", "records"):
                    value = data.get(key)
                    if isinstance(value, list):
                        raw_records = value
                        break
                if raw_records is None:
                    raise ValueError(f"Unsupported JSON structure in {dataset_path}")
            else:
                raise ValueError(f"Unsupported JSON structure in {dataset_path}")
        elif dataset_path.suffix.lower() == ".parquet":
            raw_records = pd.read_parquet(dataset_path).to_dict("records")
        else:
            raise ValueError(f"Unsupported MMLU-Pro file format: {dataset_path.suffix}")

        normalized_records = [self._normalize_record(record) for record in raw_records]

        rng = np.random.default_rng(self._shuffle_seed)
        permutation = rng.permutation(len(normalized_records))
        records = [normalized_records[idx] for idx in permutation]

        print("Total number of questions: ", len(records))
        return records

    @staticmethod
    def _extract_question(record: Dict[str, Any]) -> str:
        for key in ("question", "Question", "problem", "Problem"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise KeyError(f"Unable to extract question from record: {record}")

    @staticmethod
    def _clean_option_text(option: Any) -> str:
        option_text = str(option).strip()
        option_text = re.sub(r"^\s*\([A-Ja-j]\)\s*", "", option_text)
        option_text = re.sub(r"^\s*[A-Ja-j][\.\):\-]\s*", "", option_text)
        return option_text.strip()

    @classmethod
    def _extract_options(cls, record: Dict[str, Any]) -> List[tuple[int, str]]:
        raw_options = record.get("options")
        if raw_options is None:
            raw_options = record.get("choices")
        if isinstance(raw_options, list):
            raw_option_list = raw_options
        elif isinstance(raw_options, np.ndarray):
            raw_option_list = raw_options.tolist()
        elif isinstance(raw_options, dict):
            raw_option_list = []
            for letter in OPTION_LETTERS:
                value = raw_options.get(letter) or raw_options.get(letter.lower())
                if value is not None and str(value).strip():
                    raw_option_list.append(value)
        else:
            raw_option_list = []
            for letter in OPTION_LETTERS:
                value = record.get(letter) or record.get(letter.lower())
                if value is not None and str(value).strip():
                    raw_option_list.append(value)

        options = []
        for original_index, option in enumerate(raw_option_list):
            cleaned_option = cls._clean_option_text(option)
            if not cleaned_option or cleaned_option == "N/A":
                continue
            options.append((original_index, cleaned_option))

        if len(options) < 2 or len(options) > len(OPTION_LETTERS):
            raise ValueError(
                f"MMLU-Pro expects between 2 and {len(OPTION_LETTERS)} options, got {len(options)}. "
                f"Record: {record}"
            )
        return options

    @classmethod
    def _extract_correct_answer(cls, record: Dict[str, Any], options: List[tuple[int, str]]) -> str:
        answer_index = record.get("answer_index")
        if answer_index is not None:
            original_index = int(answer_index)
            for filtered_index, (candidate_index, _) in enumerate(options):
                if candidate_index == original_index:
                    return OPTION_LETTERS[filtered_index]

        answer = record.get("answer")
        if answer is not None:
            answer_str = str(answer).strip()
            if len(answer_str) == 1 and answer_str.upper() in OPTION_LETTERS:
                original_index = OPTION_LETTERS.index(answer_str.upper())
                for filtered_index, (candidate_index, _) in enumerate(options):
                    if candidate_index == original_index:
                        return OPTION_LETTERS[filtered_index]
                if answer_str.upper() in OPTION_LETTERS[:len(options)]:
                    return answer_str.upper()
            cleaned_answer = cls._clean_option_text(answer_str)
            for index, (_, option_text) in enumerate(options):
                if cleaned_answer == option_text:
                    return OPTION_LETTERS[index]

        raise KeyError(f"Unable to extract correct option from record: {record}")

    @classmethod
    def _normalize_record(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        question = cls._extract_question(record)
        options = cls._extract_options(record)
        correct_answer = cls._extract_correct_answer(record, options)
        option_texts = [option_text for _, option_text in options]

        normalized = {
            "question": question,
            "options": option_texts,
            "correct_answer": correct_answer,
        }
        for letter, option_text in zip(OPTION_LETTERS, option_texts):
            normalized[letter] = option_text
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
        question_lines = [record["question"]]
        for index, option_text in enumerate(record["options"]):
            question_lines.append(f"Option {OPTION_LETTERS[index]}: {option_text}")
        return {"task": "\n".join(question_lines) + "\n"}

    @staticmethod
    def _get_valid_option_letters(record: Dict[str, Any] | None) -> List[str]:
        if record is None:
            return OPTION_LETTERS.copy()
        return OPTION_LETTERS[:len(record["options"])]

    @staticmethod
    def _extract_answer_letter(answer: str, valid_letters: List[str]) -> str:
        valid_letter_group = "".join(valid_letters)

        answer_is_match = re.search(
            rf"answer is\s*\(?([{valid_letter_group}])\)?",
            answer,
            flags=re.IGNORECASE,
        )
        if answer_is_match is not None:
            return answer_is_match.group(1).upper()

        answer_colon_match = re.search(
            rf"answer:\s*\(?([{valid_letter_group}])\)?",
            answer,
            flags=re.IGNORECASE,
        )
        if answer_colon_match is not None:
            return answer_colon_match.group(1).upper()

        standalone_matches = re.findall(
            rf"\b([{valid_letter_group}])\b",
            answer,
            flags=re.IGNORECASE,
        )
        if standalone_matches:
            return standalone_matches[-1].upper()

        return random.Random(answer).choice(valid_letters)

    def postprocess_answer(self, answer, record: Dict[str, Any] | None = None) -> str:
        if isinstance(answer, list):
            answer = answer[0] if len(answer) > 0 else ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if not answer:
            return ""
        valid_letters = self._get_valid_option_letters(record)
        return self._extract_answer_letter(answer, valid_letters)

    @staticmethod
    def record_to_target_answer(record: Dict[str, Any]) -> str:
        correct_answer = record["correct_answer"]
        if not isinstance(correct_answer, str):
            raise TypeError(f"String expected but got {type(correct_answer)}")
        return correct_answer.upper()
