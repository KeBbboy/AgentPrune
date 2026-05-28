import importlib.util
from pathlib import Path


_NODE_PATH = Path(__file__).resolve().parents[1] / "AgentDropout" / "graph" / "node.py"
_SPEC = importlib.util.spec_from_file_location("agentdropout_graph_node_for_tests", _NODE_PATH)
node_module = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(node_module)
Node = node_module.Node


class DummyNode(Node):
    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        return None

    async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        return None

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        return None


class FakeCompressor:
    def __init__(self):
        self.calls = 0

    def compress_payload(self, output):
        self.calls += 1
        return f"COMPRESSED::{output}"


def test_regular_spatial_and_temporal_inputs_are_compressed(monkeypatch):
    fake_compressor = FakeCompressor()
    monkeypatch.setattr(node_module, "get_selective_context_compressor", lambda: fake_compressor)
    producer = DummyNode("producer", agent_name="AnalyzeAgent")
    producer.role = "Producer"
    producer.outputs = ["B\nreasoning"]
    producer.last_memory["outputs"] = ["B\nprevious reasoning"]

    consumer = DummyNode("consumer", agent_name="AnalyzeAgent")
    consumer.spatial_predecessors = [producer]
    consumer.temporal_predecessors = [producer]

    assert consumer.get_spatial_info()["producer"]["output"] == "COMPRESSED::B\nreasoning"
    assert consumer.get_temporal_info()["producer"]["output"] == "COMPRESSED::B\nprevious reasoning"
    assert fake_compressor.calls == 2


def test_final_major_vote_receives_uncompressed_inputs(monkeypatch):
    fake_compressor = FakeCompressor()
    monkeypatch.setattr(node_module, "get_selective_context_compressor", lambda: fake_compressor)
    producer = DummyNode("producer", agent_name="AnalyzeAgent")
    producer.role = "Producer"
    producer.outputs = ["B\nreasoning"]
    producer.last_memory["outputs"] = ["B\nprevious reasoning"]

    final_vote = DummyNode("final", agent_name="FinalMajorVote")
    final_vote.spatial_predecessors = [producer]
    final_vote.temporal_predecessors = [producer]

    assert final_vote.get_spatial_info()["producer"]["output"] == "B\nreasoning"
    assert final_vote.get_temporal_info()["producer"]["output"] == "B\nprevious reasoning"
    assert fake_compressor.calls == 0
