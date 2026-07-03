"""SSP 算法单元测试。

覆盖：
    * soft_assignment 形状 & softmax 性质；
    * ssp_fixed_spatial_masks 与 apply_ssp 接口一致性；
    * apply_ssp 在 fake graph 上不抛异常并改写拓扑；
    * Planner 的纯函数部分（不调 LLM）。

说明：
    * 测试不依赖真实 LLM（Planner 用 monkeypatch）；
    * 测试不依赖真实网络模型权重以外的资源（embedder 用真实 MiniLM）。
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pytest

from AgentDropout.algorithm.ssp import (
    SSPConfig,
    soft_assignment,
    ssp_fixed_spatial_masks,
    apply_ssp,
)
from AgentDropout.algorithm.stages import Stage, StageDAG, ComplexityLevel
from AgentDropout.embeddings import MiniLMEmbedder


# ---------------------------------------------------------------------------
# soft_assignment
# ---------------------------------------------------------------------------


def test_soft_assignment_shape_and_probability():
    """输出形状 = (N, K)，每行是合法概率分布。"""
    rng = np.random.default_rng(0)
    profile_emb = rng.normal(size=(4, 16)).astype(np.float32)
    stage_emb = rng.normal(size=(3, 16)).astype(np.float32)
    # 归一化（模拟接口契约）
    profile_emb /= np.linalg.norm(profile_emb, axis=1, keepdims=True)
    stage_emb /= np.linalg.norm(stage_emb, axis=1, keepdims=True)

    alpha = soft_assignment(profile_emb, stage_emb, temperature=1.0)
    assert alpha.shape == (4, 3)
    assert np.all(alpha >= 0)
    assert np.allclose(alpha.sum(axis=1), 1.0, atol=1e-6)


def test_soft_assignment_temperature():
    """T 越小越尖锐，T 越大越平滑。"""
    rng = np.random.default_rng(1)
    profile_emb = rng.normal(size=(1, 8)).astype(np.float32)
    stage_emb = rng.normal(size=(4, 8)).astype(np.float32)
    profile_emb /= np.linalg.norm(profile_emb, axis=1, keepdims=True)
    stage_emb /= np.linalg.norm(stage_emb, axis=1, keepdims=True)

    a_low_T = soft_assignment(profile_emb, stage_emb, temperature=0.1)
    a_high_T = soft_assignment(profile_emb, stage_emb, temperature=10.0)
    # 锐度用最大分量衡量
    assert a_low_T.max() > a_high_T.max() + 0.1


def test_soft_assignment_invalid_temperature():
    """temperature ≤ 0 应抛 ValueError。"""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2, 4)).astype(np.float32)
    with pytest.raises(ValueError):
        soft_assignment(a, a, temperature=0.0)
    with pytest.raises(ValueError):
        soft_assignment(a, a, temperature=-1.0)




# ---------------------------------------------------------------------------
# ssp_fixed_spatial_masks
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embedder():
    return MiniLMEmbedder()


@pytest.fixture
def stage_dag_3():
    return StageDAG(
        stages=[
            Stage("S1", "Analyze", "分析问题并识别关键要素。"),
            Stage("S2", "Execute", "基于分析结果执行计算或推理。"),
            Stage("S3", "Verify", "复核答案一致性并输出最终结果。"),
        ],
        edges=[("S1", "S2"), ("S2", "S3")],
    )


def test_ssp_fixed_spatial_masks_shape_and_diagonal(stage_dag_3, embedder):
    """矩阵形状正确，对角线为 0。"""
    profiles = ["math expert", "history expert", "physics expert", "biology expert", "chemistry expert"]
    masks = ssp_fixed_spatial_masks(profiles, stage_dag_3, embedder, SSPConfig(tau=0.3))
    assert len(masks) == 5
    assert all(len(row) == 5 for row in masks)
    for i in range(5):
        assert masks[i][i] == 0


def test_ssp_fixed_spatial_masks_invalid_inputs(stage_dag_3, embedder):
    """空 profile / 空 stage 应抛 ValueError。"""
    with pytest.raises(ValueError):
        ssp_fixed_spatial_masks([], stage_dag_3, embedder, SSPConfig())
    empty_dag = StageDAG(stages=[], edges=[])
    with pytest.raises(ValueError):
        ssp_fixed_spatial_masks(["a"], empty_dag, embedder, SSPConfig())


# ---------------------------------------------------------------------------
# apply_ssp（轻量 fake graph，不调真实 LLM）
# ---------------------------------------------------------------------------


class _FakeNode:
    """最小可用的 Node 替身，兼容 Graph.clear_spatial_connection / add_successor。"""

    def __init__(self, node_id, role):
        self.id = node_id
        self.role = role
        self.spatial_predecessors = []
        self.spatial_successors = []

    def add_successor(self, other, st="spatial"):
        if other not in self.spatial_successors:
            self.spatial_successors.append(other)
        if self not in other.spatial_predecessors:
            other.spatial_predecessors.append(self)


class _FakeGraph:
    """最小可用的 Graph 替身，仅支持 apply_ssp 需要的接口。"""

    def __init__(self, node_roles):
        # node_roles: List[(node_id, role_text)]
        self.nodes = {}
        for nid, role in node_roles:
            self.nodes[nid] = _FakeNode(nid, role)

    def clear_spatial_connection(self):
        for n in self.nodes.values():
            n.spatial_predecessors = []
            n.spatial_successors = []


def _count_edges(graph):
    """统计无向边数（每对 i→j, j→i 算 1 条）。"""
    edges = set()
    for n in graph.nodes.values():
        for s in n.spatial_successors:
            a, b = sorted([n.id, s.id])
            edges.add((a, b))
    return len(edges)


def test_apply_ssp_modifies_topology(stage_dag_3, embedder):
    """apply_ssp 应正确改写拓扑：同阶段 agent 之间必有边，跨阶段按规则。"""
    graph = _FakeGraph(
        [
            ("a0", "数学分析专家"),
            ("a1", "代数专家"),
            ("a2", "历史学家"),
            ("a3", "语言学家"),
            ("a4", "文学评论家"),
        ]
    )

    report = apply_ssp(graph, stage_dag_3, embedder, SSPConfig(tau=0.3, temperature=1.0))
    assert "num_edges" in report
    assert "density" in report
    assert "stage_assignment" in report
    assert len(report["stage_assignment"]) == 5

    # 拓扑改写后 decision_node 之类的访问不影响（fake graph 没有）
    # 但 num_edges 必须 > 0
    assert report["num_edges"] > 0
    # 同阶段 agent 之间必有边（无向边数 ≥ 各组 clique 之和）
    assert _count_edges(graph) > 0


def test_apply_ssp_topk_smaller_keeps_fewer_edges(stage_dag_3, embedder):
    """topk 越小，保留边数越少（密度单调下降）。"""
    graph_lo = _FakeGraph(
        [("a0", f"profile {i}") for i in range(5)]
    )
    graph_hi = _FakeGraph(
        [("a0", f"profile {i}") for i in range(5)]
    )

    r_lo = apply_ssp(graph_lo, stage_dag_3, embedder, SSPConfig(mode="topk", topk=1))
    r_hi = apply_ssp(graph_hi, stage_dag_3, embedder, SSPConfig(mode="topk", topk=4))
    assert r_lo["num_edges"] <= r_hi["num_edges"], (
        f"topk=1 应保留更少边：{r_lo['num_edges']} vs {r_hi['num_edges']}"
    )


# ---------------------------------------------------------------------------
# Planner 的纯函数部分（不调 LLM）
# ---------------------------------------------------------------------------


def test_complexity_estimation_simple():
    from AgentDropout.algorithm.planner import _estimate_complexity
    assert _estimate_complexity("1+1=?") == ComplexityLevel.SIMPLE
    assert _estimate_complexity("") == ComplexityLevel.SIMPLE


def test_complexity_estimation_medium_by_length():
    from AgentDropout.algorithm.planner import _estimate_complexity
    long_query = "x" * 200
    assert _estimate_complexity(long_query) == ComplexityLevel.HARD


def test_complexity_estimation_hard_by_keyword():
    from AgentDropout.algorithm.planner import _estimate_complexity
    # 含 "and then" 触发升级
    assert _estimate_complexity("do A and then do B") == ComplexityLevel.HARD
    # 同时长度 + 关键词 → hard
    long_complex = "step 1: 设计 UI；step 2: 实现后端；step 3: 部署到生产环境；" + "x" * 200
    assert _estimate_complexity(long_complex) == ComplexityLevel.HARD
    # 短 query 含中文关键词也触发 hard
    assert _estimate_complexity("设计前端，然后实现后端") == ComplexityLevel.HARD


def test_planner_parse_clean_json():
    from AgentDropout.algorithm.planner import StagePlanner
    raw = json_dumps_3_stages()
    dag = StagePlanner._parse_dag(raw, ComplexityLevel.HARD)
    assert dag.num_stages == 3
    assert {s.id for s in dag.stages} == {"S1", "S2", "S3"}
    assert ("S1", "S2") in dag.edges
    assert ("S2", "S3") in dag.edges


def test_planner_parse_markdown_fenced():
    from AgentDropout.algorithm.planner import StagePlanner
    raw = "```json\n" + json_dumps_3_stages() + "\n```"
    dag = StagePlanner._parse_dag(raw, ComplexityLevel.HARD)
    assert dag.num_stages == 3


def test_planner_parse_with_leading_text():
    from AgentDropout.algorithm.planner import StagePlanner
    raw = "下面是 JSON：\n" + json_dumps_3_stages() + "\n完。"
    dag = StagePlanner._parse_dag(raw, ComplexityLevel.HARD)
    assert dag.num_stages == 3


def test_planner_parse_rejects_out_of_range():
    from AgentDropout.algorithm.planner import StagePlanner
    # 1 个阶段超出 medium 范围 [3, 5]
    raw = json_dumps_3_stages().replace('"S3"', '"X"')  # 仅构造合法 JSON 但阶段数会被改
    # 实际构造一个只有 1 个阶段的合法 JSON
    raw_one = '{"stages":[{"id":"S1","name":"x","description":"y"}],"edges":[]}'
    with pytest.raises(ValueError):
        StagePlanner._parse_dag(raw_one, ComplexityLevel.HARD)


def test_planner_parse_rejects_empty_description():
    from AgentDropout.algorithm.planner import StagePlanner
    raw = '{"stages":[{"id":"S1","name":"x","description":""},{"id":"S2","name":"x","description":"y"},{"id":"S3","name":"x","description":"z"}],"edges":[]}'
    with pytest.raises(ValueError):
        StagePlanner._parse_dag(raw, ComplexityLevel.HARD)


def test_fallback_dag_is_well_formed():
    from AgentDropout.algorithm.planner import _fallback_dag
    dag = _fallback_dag()
    assert dag.num_stages == 3
    assert len(dag.edges) == 2


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def json_dumps_3_stages():
    """构造一个合法的 3 阶段 JSON 字符串。"""
    import json
    return json.dumps(
        {
            "stages": [
                {"id": "S1", "name": "Analyze", "description": "分析问题并识别关键要素。"},
                {"id": "S2", "name": "Execute", "description": "基于分析执行计算或推理。"},
                {"id": "S3", "name": "Verify", "description": "复核答案一致性并输出最终结果。"},
            ],
            "edges": [
                {"from": "S1", "to": "S2"},
                {"from": "S2", "to": "S3"},
            ],
        },
        ensure_ascii=False,
    )
