import glob
import re
import pandas as pd
from typing import Union, List, Literal, Any, Dict
import numpy as np
from abc import ABC

class MMLUDataset(ABC):
    def __init__(self,
        split: Union[Literal['dev'], Literal['val'], Literal['test']],
        ) -> None:

        self._split = split

        data_path = f"dataset/MMLU/data/{self._split}/"
        self._total_df: pd.DataFrame = self._load_data(data_path)

    @staticmethod
    def get_domain() -> str:
        return 'mmlu'

    @staticmethod
    def _load_data(
        data_path: str,
        ) -> pd.DataFrame:

        rng = np.random.default_rng(888)

        csv_paths = glob.glob(data_path + "*.csv")
        csv_paths = sorted(csv_paths)
        print("Number of topics: ", len(csv_paths))

        names = ['question', 'A', 'B', 'C', 'D', 'correct_answer']

        total_df = pd.DataFrame(columns=names)
        for path in csv_paths:
            single_df = pd.read_csv(path, header=None,
                            names=names,encoding='utf-8')
            total_df = pd.concat([total_df, single_df])

        total_df = total_df.reset_index(drop=True)

        # Pseudorandom shuffle
        total_df = total_df.reindex(rng.permutation(total_df.index))

        print("Total number of questions: ", len(total_df))

        return total_df

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> pd.DataFrame:
        record = self._total_df.iloc[index]
        assert isinstance(record, pd.DataFrame) or isinstance(record, pd.Series)
        return record

    @staticmethod
    def record_to_input(record: pd.DataFrame) -> Dict[str, Any]:
        demo_question = (
            f"{record['question']}\n"
            f"Option A: {record['A']}\n"
            f"Option B: {record['B']}\n"
            f"Option C: {record['C']}\n"
            f"Option D: {record['D']}\n"
            )
        input_dict = {"task": demo_question}
        return input_dict

    def postprocess_answer(self, answer):
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        answer = answer.strip()
        if not answer:
            return answer

        TAG_OPEN = "think"
        TAG_CLOSE = "/think"
        LOW_OPEN = "<" + TAG_OPEN + ">"
        LOW_CLOSE = "<" + TAG_CLOSE + ">"
        OPEN_LEN = len(LOW_OPEN)
        CLOSE_LEN = len(LOW_CLOSE)

        lower = answer.lower()
        # `` blocks first
        if LOW_OPEN in lower:
            open_idx = lower.index(LOW_OPEN) + OPEN_LEN
            close_idx = lower.find(LOW_CLOSE, open_idx)
            if close_idx != -1:
                answer = answer[close_idx + CLOSE_LEN:].strip()
            else:
                # thinking block never closed -> drop everything
                answer = ""
        if not answer:
            return ""

        # first non-empty line of the remainder
        first_line = ""
        for line in answer.splitlines():
            if line.strip():
                first_line = line.strip()
                break
        if not first_line:
            return ""

        # 1) leading "A" / "**B**" / "(C)" etc.
        match = re.match(
            r"^\**\s*([A-D])(?:\s|$|[).:：,-])", first_line, flags=re.IGNORECASE
        )
        if match:
            return match.group(1).upper()

        # 2) descriptive form: "The answer is B", "choose D"
        match = re.search(
            r"\b(?:answer|option|choose)\b[^A-D\n]{0,20}([A-D])\b",
            first_line, flags=re.IGNORECASE,
        )
        if not match:
            match = re.search(
                r"\b(?:answer|option|choose)\s*(?:is|:|：)?\s*(?:option\s+)?([A-D])\b",
                first_line, flags=re.IGNORECASE,
            )
        if match:
            return match.group(1).upper()

        # 3) last standalone A-D anywhere in the response (covers "C")
        for scope in (first_line, answer):
            candidates = re.findall(
                r"(?<![A-Za-z0-9])([A-D])(?![A-Za-z0-9])", scope
            )
            if candidates:
                return candidates[-1].upper()

        return first_line[0]  # last-ditch fallback

    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        correct_answer = record['correct_answer']
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} "
            f"of type {type(correct_answer)} (2)" \
            f" record={record}")
        return correct_answer
