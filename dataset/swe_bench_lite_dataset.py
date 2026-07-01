import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


class SWEBenchLiteDataset:
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        dataset_dir: str = "dataset/swe_bench_lite",
        split: str = "test",
        use_huggingface: bool = True,
    ) -> None:
        self._split = split
        self._dataset_path = Path(dataset_path) if dataset_path else None
        self._dataset_dir = Path(dataset_dir)
        self._use_huggingface = use_huggingface
        self._records = self._load_data()

    @staticmethod
    def get_domain() -> str:
        return "swe_bench_lite"

    def _load_data(self) -> List[Dict[str, Any]]:
        if self._use_huggingface:
            return self._load_from_huggingface()
        return self._load_from_local()

    def _load_from_huggingface(self) -> List[Dict[str, Any]]:
        try:
            from datasets import load_dataset

            logger.info("Loading SWE-bench Lite from HuggingFace...")
            ds = load_dataset("princeton-nlp/SWE-bench_Lite", split=self._split)
            records = []
            for idx, item in enumerate(ds):
                record = self._normalize_record(item, idx)
                records.append(record)
            logger.info(f"Loaded {len(records)} records from HuggingFace")
            return records
        except Exception as e:
            logger.warning(f"Failed to load from HuggingFace: {e}")
            return self._load_from_local()

    def _candidate_paths(self) -> List[Path]:
        if self._dataset_path is not None:
            return [self._dataset_path]
        return [
            self._dataset_dir / f"{self._split}.jsonl",
            self._dataset_dir / f"{self._split}.json",
            self._dataset_dir / f"swe-bench-lite_{self._split}.jsonl",
            self._dataset_dir / f"swe-bench-lite_{self._split}.json",
            self._dataset_dir / f"SWE-bench_Lite_{self._split}.jsonl",
            self._dataset_dir / f"SWE-bench_Lite_{self._split}.json",
        ]

    def _load_from_local(self) -> List[Path]:
        dataset_path = next((path for path in self._candidate_paths() if path.exists()), None)
        if dataset_path is None:
            tried = ", ".join(str(path) for path in self._candidate_paths())
            raise FileNotFoundError(
                f"No SWE-bench Lite data file found. Tried: {tried}\n"
                f"Please download the dataset from HuggingFace: princeton-nlp/SWE-bench_Lite"
            )

        suffix = dataset_path.suffix.lower()
        if suffix == ".jsonl":
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_records = [json.loads(line) for line in f if line.strip()]
        elif suffix == ".json":
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_records = self._extract_record_list(data, dataset_path)
        else:
            raise ValueError(f"Unsupported SWE-bench Lite file format: {dataset_path.suffix}")

        records = [self._normalize_record(record, idx) for idx, record in enumerate(raw_records)]
        print(f"Loaded SWE-bench Lite records from {dataset_path}")
        print("Total number of tasks: ", len(records))
        return records

    @staticmethod
    def _extract_record_list(data: Any, dataset_path: Path) -> List[Dict[str, Any]]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "examples", "items", "records", "tasks", "instances"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError(f"Unsupported JSON structure in {dataset_path}")

    @staticmethod
    def _first_present(record: Dict[str, Any], keys: List[str], default: Any = "") -> Any:
        for key in keys:
            value = record.get(key)
            if value is not None:
                if isinstance(value, str) and not value.strip():
                    continue
                return value
        return default

    @classmethod
    def _normalize_record(cls, record: Dict[str, Any], index: int) -> Dict[str, Any]:
        if not isinstance(record, dict):
            raise ValueError(f"Expected a dict record, got {type(record).__name__}: {record}")

        instance_id = cls._first_present(record, ["instance_id", "task_id", "id", "uid", "example_id"], default=str(index))
        problem_statement = cls._first_present(record, ["problem_statement", "issue", "problem", "description", "prompt"], default="")
        repo = cls._first_present(record, ["repo", "repository", "repo_name"], default="")
        patch = cls._first_present(record, ["patch", "solution", "fix", "ground_truth", "model_patch"], default=None)
        test_patch = cls._first_present(record, ["test_patch", "tests", "test_code", "test_case"], default="")
        base_commit = cls._first_present(record, ["base_commit", "commit", "commit_hash"], default="")
        fail_to_pass = cls._first_present(record, ["fail_to_pass", "fail_tests", "failing_tests"], default=[])
        pass_to_pass = cls._first_present(record, ["pass_to_pass", "pass_tests", "passing_tests"], default=[])
        context = cls._first_present(record, ["context", "code_context", "files", "file_contents"], default="")

        if isinstance(fail_to_pass, str):
            fail_to_pass = json.loads(fail_to_pass) if fail_to_pass.strip() else []
        if isinstance(pass_to_pass, str):
            pass_to_pass = json.loads(pass_to_pass) if pass_to_pass.strip() else []

        return {
            "id": str(instance_id),
            "instance_id": str(instance_id),
            "task_id": str(instance_id),
            "repo": str(repo).strip(),
            "problem_statement": str(problem_statement).strip(),
            "issue": str(problem_statement).strip(),
            "patch": None if patch is None else str(patch).strip(),
            "test_patch": str(test_patch).strip(),
            "base_commit": str(base_commit).strip(),
            "fail_to_pass": fail_to_pass,
            "pass_to_pass": pass_to_pass,
            "context": str(context).strip(),
            "metadata": cls._first_present(record, ["metadata", "meta"], default={}),
            "raw": record,
        }

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
        instance_id = record.get("instance_id", "")
        repo = record.get("repo", "")
        problem_statement = record.get("problem_statement", "")
        context = record.get("context", "")

        prompt = (
            f"Instance ID: {instance_id}\n"
            f"Repository: {repo}\n\n"
            f"Issue Description:\n{problem_statement}\n\n"
        )

        if context:
            prompt += f"Code Context:\n{context}\n\n"

        # Few-shot example with concrete diff format
        few_shot_example = """
## Example of a correct unified diff patch:

```diff
diff --git a/src/calculator.py b/src/calculator.py
index abc1234..def5678 100644
--- a/src/calculator.py
+++ b/src/calculator.py
@@ -10,7 +10,7 @@ def add(a, b):
     return a + b

 def subtract(a, b):
-    return a - b
+    return a - b  # Fixed edge case for negative numbers
```

## Important Requirements:
1. The output MUST be a valid unified diff (git diff format)
2. The diff MUST start with `diff --git` line
3. The diff MUST contain `--- a/...` and `+++ b/...` lines
4. The diff MUST contain `@@ -... +... @@` hunk headers
5. Use `-` for removed lines and `+` for added lines
6. The patch will be applied with `git apply`, so it must be syntactically correct
7. Do NOT include explanations, comments, or any text outside the diff block
8. Wrap the entire diff in triple backticks (```diff ... ```)

## Your Task:
Analyze the issue above and generate a unified diff patch that fixes it.
The patch should only modify the necessary lines to fix the issue.
"""

        prompt += "Your task is to fix this issue by modifying the code.\n"
        prompt += few_shot_example

        return {"task": prompt}

    # 常见围栏/语言标签(模型经常回 "```python" / "```diff" / "```text")
    _FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)?[ \t]*\n?(.*?)```", flags=re.DOTALL)
    _DIFF_START_RE = re.compile(r"\bdiff --git\s+")
    _HUNK_RE = re.compile(r"@@ -\d")

    @staticmethod
    def _extract_fenced(text: str) -> Optional[str]:
        """Return the most relevant fenced code block (prefer diff, then code)."""
        candidates = SWEBenchLiteDataset._FENCE_RE.findall(text)
        if not candidates:
            return None
        blocks = [body for (_lang, body) in candidates]
        # 1) prefer a block whose language tag is "diff"
        for i, (lang, _body) in enumerate(candidates):
            if (lang or "").strip().lower() in ("diff", "patch", "gitdiff"):
                return blocks[i].strip()
        # 2) prefer a block that already contains a unified diff
        for body in blocks:
            if "diff --git" in body:
                return body.strip()
        # 3) fall back to the first fenced block
        return blocks[0].strip()

    @staticmethod
    def _looks_like_diff(text: str) -> bool:
        return (
            "diff --git" in text
            and "\n--- " in text
            and "\n+++ " in text
            and bool(SWEBenchLiteDataset._HUNK_RE.search(text))
        )

    @staticmethod
    def _wrap_as_synthetic_diff(text: str) -> str:
        """把"修改后代码片段"包成最小的 unified diff。

        SWE-bench harness 接收任意 model_patch,只要它能 `git apply` 通过且通过测试。
        对于完全不是 diff 的输出,这里给一个保守兜底:声明一个文件占位,内容是原文本,
        这样打分时 "patch apply failed" 会显式记录,而不是被吃掉或误判通过。
        """
        cleaned = text.replace("\r\n", "\n").rstrip() + "\n"
        return (
            "diff --git a/__model_unparsed__/snippet.py b/__model_unparsed__/snippet.py\n"
            "new file mode 100644\n"
            "index 0000000..1111111\n"
            "--- /dev/null\n"
            "+++ b/__model_unparsed__/snippet.py\n"
            "@@ -0,0 +1,{} @@\n".format(cleaned.count("\n"))
        ) + "".join("+" + line for line in cleaned.split("\n"))

    @classmethod
    def postprocess_answer(cls, answer: Any) -> str:
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not isinstance(answer, str):
            answer = str(answer)

        answer = answer.strip()
        if not answer:
            return answer

        # 0) Normalize: upstream sometimes embeds literal backslash-n sequences
        # instead of real newlines. Unescape them so git/diff heuristics work.
        if "\\n" in answer:
            answer = answer.replace("\\n", "\n").replace("\\t", "\t").strip()
        if not answer:
            return answer

        # 1) fenced code block wins
        fenced = cls._extract_fenced(answer)
        if fenced is not None:
            if cls._looks_like_diff(fenced):
                return fenced
            # block exists but is not diff -> treat its body as code snippet
            return cls._wrap_as_synthetic_diff(fenced)

        # 2) free-form answer containing a diff
        diff_match = cls._DIFF_START_RE.search(answer)
        if diff_match:
            tail = answer[diff_match.start():].strip()
            if cls._looks_like_diff(tail):
                return tail
            return cls._wrap_as_synthetic_diff(tail)

        # 3) nothing diff-like -> synthetic diff placeholder so harness records "failed"
        return cls._wrap_as_synthetic_diff(answer)

    @staticmethod
    def record_to_target_answer(record: Dict[str, Any]) -> Optional[str]:
        return record.get("patch")

    @staticmethod
    def normalize_for_match(value: Any) -> str:
        text = "" if value is None else str(value)
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
        return text

    @classmethod
    def score(cls, prediction: Any, target: Any) -> Optional[Dict[str, bool]]:
        if target is None or str(target).strip() == "":
            return None
        pred = cls.normalize_for_match(prediction)
        gold = cls.normalize_for_match(target)
        return {
            "exact_match": pred == gold,
            "contains_match": bool(gold) and gold in pred,
        }

    def get_categories(self) -> List[str]:
        categories = set()
        for record in self._records:
            repo = record.get("repo", "")
            if repo:
                categories.add(repo.split("/")[-1] if "/" in repo else repo)
        return sorted(list(categories))

    def get_records_by_category(self, category: str) -> List[Dict[str, Any]]:
        return [
            record for record in self._records
            if category in (record.get("repo", "") or "")
        ]


import logging

logger = logging.getLogger(__name__)