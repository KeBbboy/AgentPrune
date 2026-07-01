import json
import re
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
import pandas as pd


OPTION_LETTERS = list("ABCDEFGHIJ")
DEFAULT_FEWSHOT_K = 1
MMLU_PRO_COT_INSTRUCTION = (
    "The following are multiple choice questions (with answers) about {category}. "
    "Think step by step and then output the answer in the format of \"The answer is (X)\" at the end."
)
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
        few_shot_examples: int = DEFAULT_FEWSHOT_K,
    ) -> None:
        self._split = split
        self._dataset_dir = Path(dataset_dir)
        self._shuffle_seed = shuffle_seed
        self._few_shot_examples = few_shot_examples
        self._records = self._load_data()
        self._support_records = self._load_support_records()
        self._category_examples = self._build_category_examples()

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
        return self._load_records_from_path(dataset_path)

    def _load_records_from_path(self, dataset_path: Path) -> List[Dict[str, Any]]:
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

    def _load_support_records(self) -> List[Dict[str, Any]]:
        if self._split in {"dev", "validation", "train"}:
            return self._records

        support_split_names = SPLIT_ALIASES.get("validation", ["validation"])
        for split_name in support_split_names:
            for dataset_path in [
                self._dataset_dir / f"{split_name}.json",
                self._dataset_dir / f"{split_name}.jsonl",
                self._dataset_dir / f"{split_name}.parquet",
                self._dataset_dir / f"MMLU-Pro_{split_name}.json",
                self._dataset_dir / f"MMLU-Pro_{split_name}.jsonl",
                self._dataset_dir / f"MMLU-Pro_{split_name}.parquet",
                self._dataset_dir / f"mmlu_pro_{split_name}.json",
                self._dataset_dir / f"mmlu_pro_{split_name}.jsonl",
                self._dataset_dir / f"mmlu_pro_{split_name}.parquet",
            ]:
                if dataset_path.exists():
                    return self._load_records_from_path(dataset_path)

        return self._records

    def _build_category_examples(self) -> Dict[str, List[Dict[str, Any]]]:
        category_examples: Dict[str, List[Dict[str, Any]]] = {}
        for record in self._support_records:
            category = record.get("category")
            cot_content = record.get("cot_content", "")
            if not category or not cot_content:
                continue
            category_examples.setdefault(category, []).append(record)
        return category_examples

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
            "cot_content": str(record.get("cot_content", "") or "").strip(),
            "category": str(record.get("category", "") or "").strip(),
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
    def _format_question_block(record: Dict[str, Any]) -> str:
        question_lines = [record["question"]]
        for index, option_text in enumerate(record["options"]):
            question_lines.append(f"{OPTION_LETTERS[index]}. {option_text}")
        return "\n".join(question_lines)

    @classmethod
    def _format_cot_example(cls, record: Dict[str, Any], include_answer: bool) -> str:
        prompt = cls._format_question_block(record)
        if not include_answer:
            return prompt

        cot_content = record.get("cot_content", "").strip()
        if cot_content:
            return f"{prompt}\nAnswer: Let's think step by step.\n{cot_content}\nThe answer is ({record['correct_answer']}).\n\n"
        return f"{prompt}\nAnswer: The answer is ({record['correct_answer']}).\n\n"

    def _select_few_shot_examples(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        category = record.get("category", "")
        examples = self._category_examples.get(category, [])
        filtered_examples = [example for example in examples if example["question"] != record["question"]]
        return filtered_examples[:self._few_shot_examples]

    def record_to_input(self, record: Dict[str, Any]) -> Dict[str, Any]:
        question_block = self._format_cot_example(record, include_answer=False)
        category = record.get("category", "miscellaneous")
        header = MMLU_PRO_COT_INSTRUCTION.format(category=category)
        few_shot_examples = self._select_few_shot_examples(record)

        prompt_parts = [header, ""]
        for example in few_shot_examples:
            prompt_parts.append(self._format_cot_example(example, include_answer=True))
        prompt_parts.append(f"{question_block}\nAnswer: Let's think step by step.")

        return {"task": "\n".join(prompt_parts).strip() + "\n"}

    def get_system_prompt(self) -> str:
        # 官方 MMLU-Pro system prompt（与 TIGER-Lab/MMLU-Pro evaluation/eval.py 对齐）
        return (
            "The following are multiple choice questions (with answers) about a specific subject. "
            "Think step by step and then output the answer in the format of "
            "\"The answer is (X)\" at the end, where X is the letter of the correct option."
        )

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

    def postprocess_answer(
        self,
        answer,
        record: Dict[str, Any] | None = None,
        llm_name: str | None = None,
    ) -> str:
        if isinstance(answer, list):
            answer = answer[0] if len(answer) > 0 else ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        answer = answer.strip()
        if not answer:
            return ""

        # MiniMax-M3 wraps its reasoning in ``...``; strip the block
        # when we know that model produced the reply. Other models keep
        # their original answer text.
        is_minimax = bool(llm_name) and "MiniMax-M3" in llm_name
        if is_minimax:
            OPEN_T = "think"
            CLOSE_T = "/" + OPEN_T
            lower = answer.lower()
            if OPEN_T in lower:
                open_idx = lower.index(OPEN_T) + len(OPEN_T)
                close_idx = lower.find(CLOSE_T, open_idx)
                if close_idx != -1:
                    answer = answer[close_idx + len(CLOSE_T):].strip()
                # else: unclosed block; let the regex still try.

        valid_letters = self._get_valid_option_letters(record)
        return self._extract_answer_letter(answer, valid_letters)

    @staticmethod
    def record_to_target_answer(record: Dict[str, Any]) -> str:
        correct_answer = record["correct_answer"]
        if not isinstance(correct_answer, str):
            raise TypeError(f"String expected but got {type(correct_answer)}")
        return correct_answer.upper()

    def stratified_sample(self, ratio: float = 0.1, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        
        category_groups = {}
        for record in self._records:
            category = record.get("category", "miscellaneous")
            category_groups.setdefault(category, []).append(record)
        
        sampled_records = []
        category_sample_counts = {}
        
        for category, records in category_groups.items():
            group_size = len(records)
            sample_size = max(1, int(round(group_size * ratio)))
            if sample_size > group_size:
                sample_size = group_size
            
            permutation = rng.permutation(len(records))
            sampled_indices = permutation[:sample_size]
            sampled_records.extend([records[idx] for idx in sampled_indices])
            category_sample_counts[category] = sample_size
        
        rng.shuffle(sampled_records)
        
        print(f"Stratified sampling: {ratio*100:.0f}% of original")
        print(f"Original total: {len(self._records)}")
        print(f"Sampled total: {len(sampled_records)}")
        print("Category sample counts:")
        for category, count in sorted(category_sample_counts.items()):
            original = len(category_groups[category])
            print(f"  {category}: {count}/{original} ({count/original*100:.1f}%)")
        
        self._records = sampled_records
