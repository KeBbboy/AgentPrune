import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from AgentDropout.utils import llmlingua2_compressor as llm2
from AgentDropout.utils.llmlingua2_compressor import LLMLingua2MessageCompressor, llmlingua2_graph_kwargs


def load_node_class():
    node_path = Path(__file__).resolve().parents[1] / "AgentDropout" / "graph" / "node.py"
    spec = importlib.util.spec_from_file_location("node_under_test", node_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Node


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


def test_llmlingua2_graph_kwargs_defaults_to_all_scope():
    args = SimpleNamespace(llmlingua2_compression=True)

    kwargs = llmlingua2_graph_kwargs(args)

    assert kwargs["llmlingua2_scope"] == "all"


def test_node_compresses_inter_agent_payload():
    Node = load_node_class()

    class DummyCompressor:
        def compress(self, text, **kwargs):
            return f"compressed:{text}"

    class DummyNode(Node):
        def _execute(self, input, spatial_info, temporal_info, **kwargs):
            return None

        async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
            return None

        def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
            return None

    sender = DummyNode("sender", domain="mmlu")
    sender.role = "Solver"
    sender.outputs = ["full answer from sender"]
    receiver = DummyNode("receiver", domain="mmlu")
    receiver.message_compressor = DummyCompressor()
    sender.add_successor(receiver, "spatial")

    spatial_info = receiver.get_spatial_info()

    assert spatial_info["sender"]["output"] == "compressed:full answer from sender"


def test_node_skips_compression_when_compressor_is_detached():
    Node = load_node_class()

    class DummyNode(Node):
        def _execute(self, input, spatial_info, temporal_info, **kwargs):
            return None

        async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
            return None

        def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
            return None

    sender = DummyNode("sender", domain="mmlu")
    sender.role = "Solver"
    sender.outputs = ["full answer from sender"]
    receiver = DummyNode("receiver", domain="mmlu")
    receiver.message_compressor = None
    sender.add_successor(receiver, "spatial")

    spatial_info = receiver.get_spatial_info()

    assert spatial_info["sender"]["output"] == "full answer from sender"
