import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

from AgentDropout.utils import role_aware_compressor as role_aware
from AgentDropout.utils.role_aware_compressor import (
    RoleAwareEdgeCompressor,
    RoleAwareNodeAggregator,
    add_role_aware_args,
    role_aware_graph_kwargs,
)


def load_node_class():
    node_path = Path(__file__).resolve().parents[1] / "AgentDropout" / "graph" / "node.py"
    module_spec = importlib.util.spec_from_file_location("node_under_role_aware_test", node_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.Node


def install_fake_transformers(monkeypatch):
    role_aware._EMBEDDER_CACHE.clear()
    last_texts = []

    def vector(text):
        lower = text.lower()
        if "code" in lower:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        if "overall answer" in lower:
            return np.asarray([0.0, 0.99, 0.01], dtype=np.float32)
        if "logic is sound" in lower:
            return np.asarray([0.0, 0.65, -0.76], dtype=np.float32)
        if "variable" in lower:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        if "boundary" in lower:
            return np.asarray([0.95, 0.31, 0.0], dtype=np.float32)
        if "loop" in lower:
            return np.asarray([0.85, 0.53, 0.0], dtype=np.float32)
        if "final answer" in lower or "decision" in lower or "conclusion" in lower:
            return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        if "answer to the equation" in lower:
            return np.asarray([0.0, 0.98, 0.02], dtype=np.float32)
        if "simpson" in lower:
            return np.asarray([0.0, 0.45, 0.89], dtype=np.float32)
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_name):
            return cls()

        def __call__(self, texts, **kwargs):
            last_texts[:] = texts
            return {
                "input_ids": torch.arange(len(texts)).view(-1, 1),
                "attention_mask": torch.ones(len(texts), 1, dtype=torch.long),
            }

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_name):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **encoded):
            vectors = np.asarray([vector(text) for text in last_texts], dtype=np.float32)
            return types.SimpleNamespace(last_hidden_state=torch.tensor(vectors).unsqueeze(1))

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=FakeTokenizer, AutoModel=FakeModel),
    )


def test_role_aware_compressor_selects_receiver_relevant_sentences(monkeypatch):
    install_fake_transformers(monkeypatch)
    compressor = RoleAwareEdgeCompressor(
        model_name="fake-model",
        sentence_budget=3,
        min_sentences=1,
        dedup_threshold=0.99,
    )
    message = """
The answer to the equation is x=42.
The variable should be initialized before use.
We computed the integral using Simpson's method.
The boundary condition fails when input is negative.
Therefore the final answer is 42.
The loop should handle edge cases properly.
"""

    compressed = compressor.compress(
        message,
        receiver_role="Code Reviewer",
        receiver_description="Reviews code logic, boundary conditions, and variable handling",
    )

    assert "variable should be initialized" in compressed
    assert "boundary condition fails" in compressed
    assert "loop should handle edge cases" in compressed
    assert "answer to the equation" not in compressed
    assert len(compressor.last_sentence_scores) == 6


def test_role_aware_compressor_deduplicates_within_one_message(monkeypatch):
    install_fake_transformers(monkeypatch)
    compressor = RoleAwareEdgeCompressor(
        model_name="fake-model",
        sentence_budget=2,
        min_sentences=1,
        dedup_threshold=0.88,
    )
    message = """
The answer to the equation is x=42.
We computed the integral using Simpson's method.
Therefore the final answer is 42.
"""

    compressed = compressor.compress(
        message,
        receiver_role="Final Decision Maker",
        receiver_description="Makes final decisions based on conclusions and numerical results",
    )

    assert "final answer is 42" in compressed
    assert "Simpson's method" in compressed
    assert "answer to the equation" not in compressed


def test_role_aware_graph_kwargs_defaults_to_minilm_model():
    args = types.SimpleNamespace(role_aware_compression=True)

    kwargs = role_aware_graph_kwargs(args)

    assert kwargs["role_aware_compression"] is True
    assert kwargs["role_aware_model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert kwargs["role_aware_rate"] == 0.5
    assert kwargs["node_aware_total_budget"] == 10


def test_role_aware_cli_exposes_only_two_switches():
    parser = argparse.ArgumentParser()

    add_role_aware_args(parser)

    option_dests = [
        action.dest
        for action in parser._actions
        if action.option_strings and action.dest != "help"
    ]
    assert option_dests == ["role_aware_compression", "node_aware_aggregation"]


def test_node_aggregator_deduplicates_across_senders(monkeypatch):
    install_fake_transformers(monkeypatch)
    aggregator = RoleAwareNodeAggregator(
        model_name="fake-model",
        total_budget=3,
        dedup_threshold=0.88,
        source_weight=0.3,
    )
    spatial_info = {
        "A2": {
            "role": "Code Reviewer",
            "output": "Overall answer is 42.\nLogic is sound.",
            "message_type": "full",
        },
        "A1": {
            "role": "Math Solver",
            "output": "The answer to the equation is x=42.\nWe computed the integral using Simpson's method.",
            "message_type": "full",
        },
    }

    aggregated = aggregator.aggregate(
        spatial_info,
        receiver_role="Final Decision Maker",
        receiver_description="Makes final decisions based on conclusions and numerical results",
    )

    combined_output = "\n".join(info["output"] for info in aggregated.values())
    assert "Overall answer is 42" in combined_output
    assert "Simpson's method" in combined_output
    assert "answer to the equation" not in combined_output
    assert aggregator.num_aggregated == 1


def test_node_aggregator_only_runs_for_spatial_info():
    Node = load_node_class()

    class DummyAggregator:
        def __init__(self):
            self.calls = 0

        def aggregate(self, spatial_info, **kwargs):
            self.calls += 1
            return spatial_info

    class DummyNode(Node):
        def _execute(self, input, spatial_info, temporal_info, **kwargs):
            return None

        async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
            return None

        def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
            return None

    sender = DummyNode("sender", domain="mmlu")
    sender.role = "Solver"
    sender.outputs = ["spatial output"]
    sender.last_memory["outputs"] = ["temporal output"]
    receiver = DummyNode("receiver", domain="mmlu")
    aggregator = DummyAggregator()
    receiver.spatial_aggregator = aggregator
    sender.add_successor(receiver, "spatial")
    sender.add_successor(receiver, "temporal")

    receiver.get_spatial_info()
    receiver.get_temporal_info()

    assert aggregator.calls == 1
