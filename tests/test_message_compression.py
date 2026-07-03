"""针对 AgentDropout/graph/message_compression 的离线冒烟测试。

不在这里调真实 LLM; 仅测试配置 / 三个模块的纯函数行为。

3 个开关 ↔ 3 个独立模块:
  --role_aware_compression  -> role_aware_compression.py
  --node_aware_aggregation  -> node_aware_aggregation.py
  --temporal_api_summary    -> temporal_api_summary.py
"""
import sys
import os
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from AgentDropout.graph.message_compression import (
    MessageCompressionConfig,
    get_message_compression_config,
    set_message_compression_config,
    add_message_compression_arguments,
    build_message_compression_config_from_args,
    compress_messages_role_aware,
    aggregate_node_aware,
    apply_temporal_api_summary,
    RoleAwareUnstructuredPayloadError,
    AggregationUnstructuredPayloadError,
)


def test_default_config_three_bool_only():
    """配置应该只有 3 个 bool 字段, 没有 temporal_summary_mode / temporal_full_history_edges."""
    cfg = get_message_compression_config()
    assert cfg.role_aware_compression is False
    assert cfg.node_aware_aggregation is False
    assert cfg.temporal_api_summary is False
    # 旧字段名必须不存在
    assert not hasattr(cfg, "temporal_summary_mode")
    assert not hasattr(cfg, "temporal_full_history_edges")


def test_argparser_only_three_cli_flags():
    import argparse
    parser = argparse.ArgumentParser()
    add_message_compression_arguments(parser)
    ns = parser.parse_args([
        "--role_aware_compression", "true",
        "--node_aware_aggregation", "yes",
        "--temporal_api_summary", "1",
    ])
    cfg = build_message_compression_config_from_args(ns)
    assert cfg.role_aware_compression is True
    assert cfg.node_aware_aggregation is True
    assert cfg.temporal_api_summary is True
    # 旧 CLI 字段名应已不存在
    assert not hasattr(ns, "temporal_summary_mode")
    assert not hasattr(ns, "temporal_full_history_edges")


# ---------- 模块 1: role_aware_compression ----------
def test_module1_full_path_raises_unstructured_error():
    """``message_type != "structured"`` 时, 模块 1 报错而非静默截断.

    设计准则: 本模块仅压 reasoning, 不允许任何可能截掉 answer 的隐式行为.
    生产默认 ``strict_unstructured=False`` 不抛错 (与 cfg 关掉等价), 但单元
    测试可通过 ``strict_unstructured=True`` 触发异常以验证设计意图.
    """
    cfg = MessageCompressionConfig(role_aware_compression=True, strict_unstructured=True)
    long_output = "x" * 5000
    payload = {"role": "Coder", "output": long_output, "message_type": "full"}
    info = {"p1": payload}
    try:
        compress_messages_role_aware(info, cfg, self_role="Coder")
    except RoleAwareUnstructuredPayloadError as e:
        assert "message_type" in str(e)
        return
    raise AssertionError("expected RoleAwareUnstructuredPayloadError but no exception raised")


def test_module1_default_message_type_also_raises():
    """未声明 message_type 时亦视为 unstructured, strict 模式抛错."""
    cfg = MessageCompressionConfig(role_aware_compression=True, strict_unstructured=True)
    payload = {"role": "Coder", "output": "x" * 5000}  # 缺 message_type
    info = {"p1": payload}
    try:
        compress_messages_role_aware(info, cfg, self_role="Coder")
    except RoleAwareUnstructuredPayloadError:
        return
    raise AssertionError("expected RoleAwareUnstructuredPayloadError but no exception raised")


def test_module1_runtime_noop_on_unstructured_with_warning():
    """运行时默认 ``strict_unstructured=False``: 非 structured payload 不抛错,
    而是打 warnings.warn 然后原样返回 (与 cfg 关掉等价), 保证生产链路
    (Node.async_execute 压扁 LLMResponse 后 message_type 永远是 ``full``) 不被
    该模块阻塞.
    """
    import warnings as _w
    cfg = MessageCompressionConfig(role_aware_compression=True)  # strict 默认 False
    long_output = "x" * 5000
    payload = {"role": "Coder", "output": long_output, "message_type": "full"}
    info = {"p1": payload}
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        out = compress_messages_role_aware(info, cfg, self_role="Coder")
    # 1) 不抛错
    # 2) 原样返回
    assert out == info
    assert out["p1"]["output"] == long_output
    # 3) 至少一条 RuntimeWarning
    assert any(issubclass(w.category, RuntimeWarning) for w in caught), "expected RuntimeWarning"


def test_module1_off_when_flag_false():
    """role_aware_compression=False 时, 即便字符串超长也不截断."""
    cfg = MessageCompressionConfig()
    payload = {"role": "Coder", "output": "x" * 5000, "message_type": "full"}
    info = {"p1": payload}
    out = compress_messages_role_aware(info, cfg, self_role="Coder")
    assert out["p1"]["output"] == "x" * 5000


def test_module1_structured_nested_form_compresses_reasoning():
    """若上游把 reasoning 塞在 ``payload["output"]["reasoning"]`` (嵌套),
    模块 1 也应识别并压缩 reasoning, 保留 answer 不动.
    """
    cfg = MessageCompressionConfig(role_aware_compression=True)
    payload = {
        "role": "Coder",
        "message_type": "structured",
        "output": {
            "reasoning": "r" * 5000,
            "answer": "FINAL ANSWER X",
            "full_text": "x",
        },
    }
    info = {"p1": payload}
    out = compress_messages_role_aware(info, cfg, self_role="Coder")
    nested = out["p1"]["output"]
    assert isinstance(nested["reasoning"], str) and len(nested["reasoning"]) < 5000
    assert "role-aware reasoning-trunc" in nested["reasoning"]
    assert nested["answer"] == "FINAL ANSWER X"


def test_module1_structured_payload_compresses_reasoning_only():
    """structured payload: 只压 reasoning, answer 必须原样保留."""
    cfg = MessageCompressionConfig(role_aware_compression=True)
    long_reasoning = "r" * 5000
    short_answer = "FINAL ANSWER X"
    payload = {
        "role": "Coder",
        "message_type": "structured",
        "reasoning": long_reasoning,
        "answer": short_answer,
        "full_text": f"<think>{long_reasoning}</think>{short_answer}",
    }
    info = {"p1": payload}
    out = compress_messages_role_aware(info, cfg, self_role="Coder")
    new = out["p1"]
    # reasoning 字段被压
    assert isinstance(new["reasoning"], str) and len(new["reasoning"]) < len(long_reasoning)
    assert "role-aware reasoning-trunc" in new["reasoning"]
    # answer 严格不变
    assert new["answer"] == short_answer


def test_module1_structured_payload_off_preserves_everything():
    """开关关闭时 structured payload 完全原样保留."""
    cfg = MessageCompressionConfig()
    payload = {
        "role": "Coder",
        "message_type": "structured",
        "reasoning": "r" * 5000,
        "answer": "FINAL A",
        "full_text": "x",
    }
    info = {"p1": payload}
    out = compress_messages_role_aware(info, cfg, self_role="Coder")
    assert out["p1"] == payload


# ---------- 模块 2: node_aware_aggregation ----------
def test_module2_node_aware_aggregation_buckets_by_role():
    """structured payload: 同 role 桶内 reasoning/answer 拼接, 不同 role 桶保留."""
    cfg = MessageCompressionConfig(node_aware_aggregation=True)
    info = {
        "k1": {"role": "AnalyzeAgent", "message_type": "structured",
               "reasoning": "A1R", "answer": "A1A", "full_text": "x"},
        "k2": {"role": "AnalyzeAgent", "message_type": "structured",
               "reasoning": "A2R", "answer": "A2A", "full_text": "y"},
        "k3": {"role": "Coder",        "message_type": "structured",
               "reasoning": "C1R", "answer": "C1A", "full_text": "z"},
    }
    out = aggregate_node_aware(info, cfg, self_role="Solver")
    assert len(out) == 2  # 2 buckets: AnalyzeAgent, Coder
    roles = {v["role"] for v in out.values()}
    assert roles == {"AnalyzeAgent", "Coder"}
    # 每个 bucket 都带 aggregated 标记
    assert all(v.get("aggregated") is True for v in out.values())
    # analyze bucket 的 reasoning 同时包含 A1R 和 A2R
    bv = {v["role"]: v for v in out.values()}
    assert "A1R" in bv["AnalyzeAgent"]["reasoning"]
    assert "A2R" in bv["AnalyzeAgent"]["reasoning"]
    assert "A1A" in bv["AnalyzeAgent"]["answer"]
    assert "A2A" in bv["AnalyzeAgent"]["answer"]


def test_module2_off_with_single_bucket_returns_input_unchanged():
    cfg = MessageCompressionConfig(node_aware_aggregation=True)
    info = {
        "k1": {"role": "AnalyzeAgent", "output": "A1", "message_type": "full"},
        "k2": {"role": "AnalyzeAgent", "output": "A2", "message_type": "full"},
    }
    out = aggregate_node_aware(info, cfg, self_role="Solver")
    assert out == info  # 只有 1 个 role 桶, 原样返回


def test_module2_structured_payload_joins_reasoning_and_answer_separately():
    """structured payload: 同 role 桶内, reasoning 与 answer 分别拼接."""
    cfg = MessageCompressionConfig(node_aware_aggregation=True)
    info = {
        "k1": {"role": "AnalyzeAgent", "message_type": "structured",
               "reasoning": "R1", "answer": "A1", "full_text": "x"},
        "k2": {"role": "AnalyzeAgent", "message_type": "structured",
               "reasoning": "R2", "answer": "A2", "full_text": "y"},
        "k3": {"role": "Coder",        "message_type": "structured",
               "reasoning": "RC", "answer": "AC", "full_text": "z"},
    }
    out = aggregate_node_aware(info, cfg, self_role="Solver")
    assert len(out) == 2
    by_role = {v["role"]: v for v in out.values()}
    assert by_role["AnalyzeAgent"]["message_type"] == "structured"
    assert "R1" in by_role["AnalyzeAgent"]["reasoning"]
    assert "R2" in by_role["AnalyzeAgent"]["reasoning"]
    assert "A1" in by_role["AnalyzeAgent"]["answer"]
    assert "A2" in by_role["AnalyzeAgent"]["answer"]
    # Coder 桶应当保留自己的 reasoning + answer
    assert "RC" in by_role["Coder"]["reasoning"]
    assert "AC" in by_role["Coder"]["answer"]


def test_module2_full_message_type_raises_unstructured_error():
    """任何 message_type != structured 的 payload 必须报错 (strict 模式),
    不做字符串拼接; 运行时默认 strict_unstructured=False 走 noop 路径,
    由 :func:`test_module2_runtime_noop_on_unstructured_with_warning` 验证.
    """
    cfg = MessageCompressionConfig(node_aware_aggregation=True, strict_unstructured=True)
    info = {
        "k1": {"role": "AnalyzeAgent", "output": "A1", "message_type": "full"},
        "k2": {"role": "Coder",        "output": "C1", "message_type": "full"},
    }
    try:
        aggregate_node_aware(info, cfg, self_role="Solver")
    except AggregationUnstructuredPayloadError as e:
        assert "2" in str(e)  # 包含计数
        return
    raise AssertionError("expected AggregationUnstructuredPayloadError but no exception raised")


def test_module2_runtime_noop_on_unstructured_with_warning():
    """运行时默认 ``strict_unstructured=False``: 非 structured payload 不抛错,
    而是在跨 role 场景下原样返回 info, 同时打 warnings.warn. 保证生产
    (Node.async_execute 压扁 LLMResponse 后 message_type 永远是 ``full``)
    不被该模块阻塞.
    """
    import warnings as _w
    cfg = MessageCompressionConfig(node_aware_aggregation=True)  # strict 默认 False
    info = {
        "k1": {"role": "AnalyzeAgent", "output": "A1", "message_type": "full"},
        "k2": {"role": "Coder",        "output": "C1", "message_type": "full"},
    }
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        out = aggregate_node_aware(info, cfg, self_role="Solver")
    assert out == info  # 原样返回, 不做拼接
    assert any(issubclass(w.category, RuntimeWarning) for w in caught), "expected RuntimeWarning"


# ---------- 模块 3: temporal_api_summary ----------


async def test_module3_apply_noop_when_flag_false():
    """temporal_api_summary=False 时, apply_* 应当 no-op, 不抛错."""
    cfg = MessageCompressionConfig()

    class _DummyNode:
        def __init__(self):
            self.last_memory = {"outputs": []}
            self.temporal_predecessors = []
            self.role = "Dummy"

    class _DummyGraph:
        llm_name = "qwen3-14b"
        nodes = {}

    g = _DummyGraph()
    g.nodes["n0"] = _DummyNode()
    await apply_temporal_api_summary(g, cfg, round_idx=0)
    assert g.nodes["n0"].last_memory.get("temporal_summary") is None


async def test_module3_first_sentence_extracts_chinese_and_english():
    """_first_sentence 应正确处理中英文句末标点."""
    from AgentDropout.graph.message_compression.temporal_api_summary import _first_sentence
    assert _first_sentence("先思考。再算。") == "先思考。"
    assert _first_sentence("Hello. World.") == "Hello."
    assert _first_sentence("多行\n第二行") == "多行"
    # 没有标点也没有换行: 整段作为一句
    s = "abcdefghij"
    assert _first_sentence(s) == s


async def test_module3_no_llm_call_at_all():
    """模块 3 应该不调 LLM. 通过让 _build_consensus_or_disagreement_summary 走字符路径来确认."""
    from AgentDropout.graph.message_compression.temporal_api_summary import (
        apply_temporal_api_summary,
    )

    cfg = MessageCompressionConfig(temporal_api_summary=True)

    class _DummyNode:
        def __init__(self, role, full_text):
            self.role = role
            self.last_memory = {"outputs": [full_text]}

    class _DummyGraph:
        def __init__(self):
            self.nodes = {}
            self.llm_name = "any"

    g = _DummyGraph()

    tag_open = "<" + "think" + ">"
    tag_close = "<" + "/" + "think" + ">"
    g.nodes["n1"] = _DummyNode("Coder", tag_open + "Reasoning line 1. More text." + tag_close + "42")
    g.nodes["n2"] = _DummyNode("Critic", tag_open + "Different reasoning. Even more." + tag_close + "Final: 43")

    await apply_temporal_api_summary(g, cfg, round_idx=0)
    # 不一致路径
    summary = g.nodes["n1"].last_memory["temporal_summary"]
    assert "不一致" in summary
    # 理由里出现 reasoning 首句 (没有 LLM 调用所以"Reasoning line 1."出现)
    assert "Reasoning line 1" in summary or "Different reasoning" in summary
    # 没有任何 LLM 调用: 当 cfg.temporal_api_summary=True 也走得通
    # (没有 LLMRegistry.get 就会报"未配置 LLM", 因此本次能跑通证伪)


def test_module3_consensus_reason_via_text_uses_first_sentence():
    """单答案路径: '理由' 应是最长 reasoning 的首句, 而不是调 LLM."""
    from AgentDropout.graph.message_compression.temporal_api_summary import (
        _consensus_reason_via_text,
    )
    per_node = [
        ("dummy1", "短理由.", "42"),
        ("dummy2", "这是一个非常长的推理段。它的首句包含完整结论。继续展开。", "42"),
    ]
    reason = _consensus_reason_via_text(per_node)
    # 选的是最长 reasoning, 第一句是 "这是一个非常长的推理段。"
    assert "这是一个非常长的推理段" in reason


def test_module3_per_answer_reason_via_text_first_sentence_only():
    """不一致路径: 每组理由只用 reasoning 首句."""
    from AgentDropout.graph.message_compression.temporal_api_summary import (
        _per_answer_reason_via_text,
    )
    r = "First sentence here. Second sentence also here. Third."
    assert _per_answer_reason_via_text("42", r) == "First sentence here."


def test_module3_split_reasoning_helper():
    """模块 3 内部的 split_reasoning 启发式拆 ``<think>...</think>`` 段."""
    from AgentDropout.graph.message_compression.temporal_api_summary import split_reasoning
    # 使用真实 think 标签, 直接拼字符串绕过 markdown
    tag_open = "<" + "think" + ">"
    tag_close = "<" + "/" + "think" + ">"
    r, a = split_reasoning(tag_open + "abc" + tag_close + "final answer")
    assert r == "abc"
    assert a == "final answer"
    r, a = split_reasoning("just plain text")
    assert r == ""
    assert a == "just plain text"
    # redacted 形式也支持
    red_open = "<" + "redacted_think" + ">"
    r, a = split_reasoning(red_open + "xyz" + tag_close + "ans")
    assert r == "xyz"
    assert a == "ans"


async def test_module3_apply_extracts_reasoning_and_writes_both_fields():
    """apply_temporal_api_summary v2:
    - 单 node 单答案 路径: 应写 temporal_summary, temporal_consensus=True, temporal_answers
    - LLM 不可用时降级仍要写字段
    """
    from AgentDropout.graph.message_compression.temporal_api_summary import apply_temporal_api_summary
    cfg = MessageCompressionConfig(temporal_api_summary=True)

    class _DummyNode:
        def __init__(self, role, full_text):
            self.role = role
            self.last_memory = {"outputs": [full_text]}

    class _DummyGraph:
        llm_name = "__test_no_llm__"  # 触发降级 (LLMRegistry.get 会抛)
        def __init__(self):
            self.nodes = {}

    g = _DummyGraph()
    tag_open = "<" + "think" + ">"
    tag_close = "<" + "/" + "think" + ">"
    full = tag_open + "long reasoning line1\\n\\nline2\\n\\nline3" + tag_close + "final answer YES"
    n = _DummyNode("Coder", full)
    g.nodes["n0"] = n
    await apply_temporal_api_summary(g, cfg, round_idx=0)
    # v2 字段名
    assert "temporal_summary" in n.last_memory
    assert n.last_memory["temporal_summary_round"] == 0
    assert n.last_memory["temporal_consensus"] is True   # 单节点答案唯一
    assert n.last_memory["temporal_answers"] == ["final answer YES"]


async def test_module3_apply_two_nodes_same_answer_consensus():
    """两个 node 给同一答案 -> is_consensus=True, summary 文案以 ``共识`` 开头."""
    from AgentDropout.graph.message_compression.temporal_api_summary import apply_temporal_api_summary
    cfg = MessageCompressionConfig(temporal_api_summary=True)

    class _DummyNode:
        def __init__(self, role, full_text):
            self.role = role
            self.last_memory = {"outputs": [full_text]}

    class _DummyGraph:
        llm_name = "__test_no_llm__"
        def __init__(self):
            self.nodes = {}

    g = _DummyGraph()
    tag_open = "<" + "think" + ">"
    tag_close = "<" + "/" + "think" + ">"
    # 故意让两份 answer 写法不同, 走归一化, 应该被识别为同一答案
    n1 = _DummyNode("Coder",     tag_open + "r1" + tag_close + "The answer is 42.")
    n2 = _DummyNode("Critic",    tag_open + "r2" + tag_close + "FINAL ANSWER: 42")
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    await apply_temporal_api_summary(g, cfg, round_idx=1)
    assert n1.last_memory["temporal_consensus"] is True
    assert n2.last_memory["temporal_consensus"] is True
    # summary 是面向两个 node 公共的那一句
    assert "共识" in n1.last_memory["temporal_summary"]
    assert "42" in n1.last_memory["temporal_summary"]


async def test_module3_apply_two_nodes_diff_answer_disagreement():
    """两个 node 给不同答案 -> is_consensus=False, summary 含多个 '答案 X:' 行."""
    from AgentDropout.graph.message_compression.temporal_api_summary import apply_temporal_api_summary
    cfg = MessageCompressionConfig(temporal_api_summary=True)

    class _DummyNode:
        def __init__(self, role, full_text):
            self.role = role
            self.last_memory = {"outputs": [full_text]}

    class _DummyGraph:
        llm_name = "__test_no_llm__"
        def __init__(self):
            self.nodes = {}

    g = _DummyGraph()
    tag_open = "<" + "think" + ">"
    tag_close = "<" + "/" + "think" + ">"
    n1 = _DummyNode("Coder",  tag_open + "r1a" + tag_close + "The answer is 42.")
    n2 = _DummyNode("Critic", tag_open + "r2b" + tag_close + "Final answer: 43")
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    await apply_temporal_api_summary(g, cfg, round_idx=2)
    assert n1.last_memory["temporal_consensus"] is False
    assert n2.last_memory["temporal_consensus"] is False
    assert "不一致" in n1.last_memory["temporal_summary"]
    # 两个 node 看到的 summary 是同一份
    assert n1.last_memory["temporal_summary"] == n2.last_memory["temporal_summary"]


def test_module3_normalize_answer_dedupes_common_prefixes():
    """_normalize_answer 应该把 'The answer is 42.' / 'final answer 42' / '42' 都归到 '42'."""
    from AgentDropout.graph.message_compression.temporal_api_summary import _normalize_answer as norm
    assert norm("The answer is 42.") == "42"
    assert norm("final answer 42") == "42"
    assert norm("42.") == "42"
    assert norm("答案是 42") == "42"
    assert norm("Final Answer Is 42") == "42"
    # 单独 'final answer' 不该归成空也不该归成非空键 (它不是完整答案)
    # 我们只做"前缀脱掉" — 剩下的字符串就是 normalization key
    assert norm("Final answer") == ""


if __name__ == "__main__":
    test_default_config_three_bool_only()
    test_argparser_only_three_cli_flags()
    test_module1_full_path_raises_unstructured_error()
    test_module1_default_message_type_also_raises()
    test_module1_off_when_flag_false()
    test_module1_structured_payload_compresses_reasoning_only()
    test_module1_structured_payload_off_preserves_everything()
    test_module2_node_aware_aggregation_buckets_by_role()
    test_module2_off_with_single_bucket_returns_input_unchanged()
    test_module2_structured_payload_joins_reasoning_and_answer_separately()
    asyncio.run(test_module3_apply_noop_when_flag_false())
    test_module3_split_reasoning_helper()
    asyncio.run(test_module3_apply_extracts_reasoning_and_writes_both_fields())
    asyncio.run(test_module3_apply_two_nodes_same_answer_consensus())
    asyncio.run(test_module3_apply_two_nodes_diff_answer_disagreement())
    test_module3_normalize_answer_dedupes_common_prefixes()
    print("All message_compression smoke tests passed.")
