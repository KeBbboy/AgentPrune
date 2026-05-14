import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from AgentPrune.prompt.prompt_set_registry import PromptSetRegistry
from AgentPrune.tools.coding.python_executor import PyExecutor
from AgentPrune.tools.reader.readers import JSONLReader


def extract_humaneval_code(output: str) -> str:
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", output, flags=re.IGNORECASE | re.DOTALL)
    if answer_match is not None:
        return answer_match.group(1).strip()
    return output.strip()


class HumanEvalDataset:
    def __init__(self, dataset_json: str = "dataset/humaneval/humaneval-py.jsonl") -> None:
        self._dataset_json = dataset_json
        self._records: List[Dict[str, Any]] = JSONLReader.parse_file(Path(dataset_json))
        self._prompt_set = PromptSetRegistry.get("humaneval")
        self._split = "test"

    @staticmethod
    def get_domain() -> str:
        return "humaneval"

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
        return {"task": record["prompt"]}

    @staticmethod
    def record_to_target_answer(record: Dict[str, Any]) -> str:
        return record["test"]

    def postprocess_answer(self, answer: Any, record: Optional[Dict[str, Any]] = None) -> str:
        processed = self._prompt_set.postprocess_answer(answer)
        if isinstance(processed, list):
            processed = processed[0] if processed else ""
        if not isinstance(processed, str):
            raise TypeError(f"Expected string answer, got {type(processed)}")
        code = extract_humaneval_code(processed)
        if code.startswith("```python") and code.endswith("```"):
            code = code.lstrip("```python\n").rstrip("\n```")
        return code

    def compute_reward(self, raw_answer: Any, record: Dict[str, Any]) -> float:
        answer = self.postprocess_answer(raw_answer, record=record)
        test = record["test"]
        is_solved, _, _ = PyExecutor().execute(answer, [test], timeout=100)
        return float(is_solved)
