import glob
import pandas as pd
from typing import Union, List, Literal, Any, Dict
import numpy as np
from abc import ABC
import re

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

    @staticmethod
    def _extract_option_answer(answer: str, valid_letters: str = "ABCD") -> str:
        escaped_letters = re.escape(valid_letters)
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        if not lines:
            return ""

        explicit_patterns = (
            rf"(?:final\s+answer|correct\s+answer|answer)\s*(?:is|:)?\s*(?:\*\*)?\s*(?:option\s*)?([{escaped_letters}])(?:\*\*)?\b",
            rf"(?:therefore|thus|so),?\s*(?:the\s+)?(?:answer|correct\s+answer)\s*(?:is|:)?\s*(?:\*\*)?\s*(?:option\s*)?([{escaped_letters}])(?:\*\*)?\b",
        )
        for pattern in explicit_patterns:
            match = re.search(pattern, answer, flags=re.IGNORECASE)
            if match is not None:
                return match.group(1).upper()

        first_line = lines[0].strip()
        first_line_match = re.match(
            rf"^(?:\*\*)?\s*(?:option\s*)?([{escaped_letters}])(?:\s*(?:\*\*)?)?(?:[\s\).:：-]|$)",
            first_line,
            flags=re.IGNORECASE,
        )
        if first_line_match is not None:
            return first_line_match.group(1).upper()

        option_mentions = {
            match.group(1).upper()
            for match in re.finditer(
                rf"\boption\s+([{escaped_letters}])\b",
                answer,
                flags=re.IGNORECASE,
            )
        }
        if len(option_mentions) == 1:
            return next(iter(option_mentions))

        return ""

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        return self._extract_option_answer(answer)

    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        correct_answer = record['correct_answer']
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} "
            f"of type {type(correct_answer)} (2)" \
            f" record={record}")
        return correct_answer
