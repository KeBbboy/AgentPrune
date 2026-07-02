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

    def postprocess_answer(self, answer, llm_name: str | None = None):
        """Extract the predicted option letter for MMLU.

        Unified implementation aligned with the official lm-evaluation-harness
        MMLU ``process_results`` behaviour:

        1. If a ``think``...``/think`` block is present (e.g. MiniMax-M3 replies),
           strip it before extraction so reasoning does not contaminate the
           final pick.
        2. From the post-strip text, take the first standalone A-D letter
           (matches anything not bordered by alphanumerics). This mirrors the
           official "first isolated A-D" rule, instead of the previous
           "last A-D" heuristic which biased MiniMax-M3 outputs differently
           from other models.
        3. Fallback: first non-whitespace character of the first non-empty line.

        The ``llm_name`` argument is kept for backward compatibility with
        callers (e.g. ``experiments/evaluate_mmlu.py``) but no longer alters
        the extraction logic — behaviour only depends on the response text,
        not on the producing model name.

        This method is only used by the evaluator/scoring pipeline. Agent-to-
        agent communication goes through ``MMLUPromptSet.postprocess_answer``
        and the raw ``full_text`` payloads, so changing this implementation
        has no effect on inter-agent message passing.
        """
        del llm_name  # behaviour is model-agnostic; argument retained for API stability.

        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        # ``LLMResponse`` (dataclass with ``reasoning`` / ``answer`` / ``full_text``)
        # and any dict with ``full_text`` should be coerced to their full_text string.
        if not isinstance(answer, str):
            if hasattr(answer, "full_text") and isinstance(answer.full_text, str):
                answer = answer.full_text
            elif isinstance(answer, dict) and isinstance(answer.get("full_text"), str):
                answer = answer["full_text"]
            else:
                answer = str(answer)
        answer = answer.strip()
        if not answer:
            return ""

        # 1) Strip a ``think``...``/think`` reasoning block if present.
        think_re = re.compile(r"think\b.*?/think", re.IGNORECASE | re.DOTALL)
        stripped = think_re.sub("", answer, count=1).strip()
        text = stripped or answer

        # 2) Look for the explicit ``Answer: X`` line introduced by the new
        # MMLU prompt. We pick the LAST occurrence to match the convention
        # "the most recent Answer: line is the authoritative final pick".
        answer_line_match = None
        for line in reversed(text.splitlines()):
            m = re.match(r"^\s*(?:final\s+)?answer\s*[:：]\s*\(?\s*([A-D])\s*\)?\s*$",
                         line.strip(), flags=re.IGNORECASE)
            if m:
                answer_line_match = m
                break
        if answer_line_match is not None:
            return answer_line_match.group(1).upper()

        # 3) Fallback: first standalone A-D letter anywhere in the cleaned text.
        for match in re.finditer(r"(?<![A-Za-z0-9])([A-D])(?![A-Za-z0-9])", text):
            return match.group(1).upper()

        # 4) Last-resort fallback: first non-whitespace char of the first non-empty line.
        for line in text.splitlines():
            stripped_line = line.strip()
            if stripped_line:
                return stripped_line[0].upper()
        return ""

    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        correct_answer = record['correct_answer']
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} "
            f"of type {type(correct_answer)} (2)" \
            f" record={record}")
        return correct_answer
