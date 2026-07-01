import json

from dataset.cl_bench_dataset import CLBenchDataset


def test_cl_bench_dataset_loads_json_and_scores_contains(tmp_path):
    dataset_file = tmp_path / "test.json"
    dataset_file.write_text(
        json.dumps([
            {
                "id": "case-1",
                "context": "Alice has a blue badge. Bob has a red badge.",
                "task": "What color badge does Alice have?",
                "answer": "blue",
                "rubrics": ["The answer should identify Alice's badge color."],
            }
        ]),
        encoding="utf-8",
    )

    dataset = CLBenchDataset(dataset_path=str(dataset_file))
    record = dataset[0]

    assert len(dataset) == 1
    assert record["id"] == "case-1"
    assert "Context:" in dataset.record_to_input(record)["task"]
    assert dataset.record_to_target_answer(record) == "blue"
    assert dataset.score("The answer is blue.", "blue") == {
        "exact_match": False,
        "contains_match": True,
    }


def test_cl_bench_dataset_accepts_common_field_aliases(tmp_path):
    dataset_file = tmp_path / "test.jsonl"
    dataset_file.write_text(
        json.dumps({
            "uid": "alias-1",
            "Background": "The launch window is Friday.",
            "Question": "When is the launch window?",
            "gold_answer": "Friday",
            "verification_rubrics": ["mentions Friday"],
        }) + "\n",
        encoding="utf-8",
    )

    dataset = CLBenchDataset(dataset_path=str(dataset_file))
    record = dataset[0]

    assert record["id"] == "alias-1"
    assert record["context"] == "The launch window is Friday."
    assert record["task"] == "When is the launch window?"
    assert record["answer"] == "Friday"
