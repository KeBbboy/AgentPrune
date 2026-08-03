import sys
import types
from types import SimpleNamespace

from AgentDropout.graph.message_compression import (
    MessageCompressionConfig,
    build_message_compression_config_from_args,
)
from AgentDropout.graph.graph import Graph
from AgentDropout.graph.node import Node
from AgentDropout.utils import llmlingua2_compressor as llm2
from AgentDropout.utils.llmlingua2_compressor import LLMLingua2MessageCompressor


def test_llmlingua2_compressor_uses_llmlingua2_prompt_compressor(monkeypatch):
    llm2._PROMPT_COMPRESSOR_CACHE.clear()
    calls = {}

    class FakePromptCompressor:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def compress_prompt(self, prompt, **kwargs):
            calls["compress"] = kwargs
            return {"compressed_prompt": "alpha gamma"}

    monkeypatch.setitem(
        sys.modules,
        "llmlingua",
        types.SimpleNamespace(PromptCompressor=FakePromptCompressor),
    )

    compressor = LLMLingua2MessageCompressor(rate=0.5, min_tokens=1)
    compressed = compressor.compress("alpha beta gamma delta", domain="mmlu")

    assert compressed == "alpha gamma"
    assert calls["init"]["use_llmlingua2"] is True
    assert calls["compress"]["rate"] == 0.5
    assert compressor.num_compressed == 1


def test_llmlingua2_compressor_reuses_loaded_model(monkeypatch):
    llm2._PROMPT_COMPRESSOR_CACHE.clear()
    calls = {"init": 0}

    class FakePromptCompressor:
        def __init__(self, **kwargs):
            calls["init"] += 1

        def compress_prompt(self, prompt, **kwargs):
            return {"compressed_prompt": "alpha gamma"}

    monkeypatch.setitem(
        sys.modules,
        "llmlingua",
        types.SimpleNamespace(PromptCompressor=FakePromptCompressor),
    )

    first = LLMLingua2MessageCompressor(rate=0.5, min_tokens=1, model_name="fake-model")
    second = LLMLingua2MessageCompressor(rate=0.5, min_tokens=1, model_name="fake-model")

    assert first.compress("alpha beta gamma delta", domain="mmlu") == "alpha gamma"
    assert second.compress("alpha beta gamma delta", domain="mmlu") == "alpha gamma"
    assert calls["init"] == 1


def test_llmlingua2_compressor_preserves_leading_answer_letter(monkeypatch):
    llm2._PROMPT_COMPRESSOR_CACHE.clear()
    calls = {}

    class FakePromptCompressor:
        def __init__(self, **kwargs):
            pass

        def compress_prompt(self, prompt, **kwargs):
            calls["prompt"] = prompt
            return {"compressed_prompt": "Fine contrasts states and clubs."}

    monkeypatch.setitem(
        sys.modules,
        "llmlingua",
        types.SimpleNamespace(PromptCompressor=FakePromptCompressor),
    )

    compressor = LLMLingua2MessageCompressor(rate=0.5, min_tokens=1, model_name="fake-model")
    compressed = compressor.compress("C\nFine argues that both A and B apply.", domain="mmlu")

    assert calls["prompt"] == "Fine argues that both A and B apply."
    assert compressed.startswith("C\n")


def test_message_compression_config_builds_llmlingua2_defaults():
    args = SimpleNamespace(
        role_aware_compression=False,
        node_aware_aggregation=False,
        temporal_api_summary=False,
        llmlingua2_compression=True,
    )

    cfg = build_message_compression_config_from_args(args)

    assert cfg.llmlingua2_compression is True
    assert not hasattr(cfg, "llmlingua2_scope")


def test_graph_attaches_llmlingua2_only_to_regular_nodes():
    class DummyCompressor:
        pass

    class DummyNode:
        message_compressor = None

    graph = Graph.__new__(Graph)
    graph.ours_message_config = MessageCompressionConfig(llmlingua2_compression=True)
    graph.message_compression_enabled = True
    graph.message_compressor = DummyCompressor()
    graph.decision_node = DummyNode()
    graph.nodes = {"regular": DummyNode()}

    graph.attach_llmlingua2_message_compressor()

    assert graph.nodes["regular"].message_compressor is graph.message_compressor
    assert graph.decision_node.message_compressor is None


def test_node_llmlingua2_compresses_structured_reasoning():
    class DummyCompressor:
        def compress(self, text, **kwargs):
            return f"compressed:{text}"

    class DummyGraph:
        ours_message_config = MessageCompressionConfig(llmlingua2_compression=True)

    class DummyNode(Node):
        def _execute(self, input, spatial_info, temporal_info, **kwargs):
            return None

        async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
            return None

        def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
            return None

    sender = DummyNode("sender", domain="mmlu")
    sender.role = "Solver"
    sender.outputs = ["Answer: C\nReasoning from sender"]
    receiver = DummyNode("receiver", domain="mmlu")
    receiver.bind_to_graph(DummyGraph())
    receiver.message_compressor = DummyCompressor()
    sender.add_successor(receiver, "spatial")

    spatial_info = receiver.get_spatial_info()
    output = spatial_info["sender"]["output"]

    assert output["reasoning"] == "compressed:Reasoning from sender"
    assert output["answer"] == "Answer: C"
    assert "compressed:Reasoning from sender" in output["full_text"]
