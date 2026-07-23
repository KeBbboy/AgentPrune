"""测试新增 benchmark 数据集配置。"""

import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from AgentDropout.benchmarks.agent_benchmarks import AgentBenchmarkDataset
from AgentDropout.tools.benchmark_tools import CodeGeneratorTool


def test_math500_default_tools_and_loader(tmp_path):
    data_dir = tmp_path / "dataset"
    math_dir = data_dir / "MATH500"
    math_dir.mkdir(parents=True)
    sample = {
        "Question": "Compute 2+2.",
        "Answer": "4",
        "type": "algebra",
        "level": "Level 1",
    }
    (math_dir / "test_subset_200.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")

    dataset = AgentBenchmarkDataset(name="math500", data_dir=data_dir)

    assert dataset.default_tools == ["web_search", "code_generator"]
    assert dataset.split == "test_subset_200"
    assert len(dataset) == 1
    assert dataset[0].answer == "4"
    assert dataset.score("\\boxed{4}", dataset[0])["correct"] is True


def test_bigcodebench_loader_and_scorer(tmp_path):
    data_dir = tmp_path / "dataset"
    bcb_dir = data_dir / "BigCodeBench"
    bcb_dir.mkdir(parents=True)
    sample = {
        "task_id": "bigcodebench/0",
        "instruct_prompt": "Implement add_one(x) that returns x + 1.",
        "code_prompt": "def add_one(x):",
        "entry_point": "add_one",
        "libs": [],
        "test": (
            "import unittest\n"
            "class TestAddOne(unittest.TestCase):\n"
            "    def test_basic(self):\n"
            "        self.assertEqual(add_one(2), 3)\n"
        ),
    }
    (bcb_dir / "v0.1.4_subset_200.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")

    dataset = AgentBenchmarkDataset(name="bigcodebench", data_dir=data_dir)
    prediction = "```python\ndef add_one(x):\n    return x + 1\n```"
    score = dataset.score(prediction, dataset[0])

    assert dataset.default_tools == ["code_generator"]
    assert dataset.split == "v0.1.4_subset_200"
    assert score["correct"] is True
    assert score["accuracy"] == 1.0


def test_code_generator_return_code_mode_does_not_execute():
    tool = CodeGeneratorTool(llm_name="unused", direct_tool_call=True, return_code=True)
    result = tool.execute(code="def add_one(x):\n    return x + 1\n")

    assert result.success is True
    assert result.metadata["mode"] == "return_code"
    assert "[Syntax check: OK]" in result.output
    assert "def add_one" in result.output
