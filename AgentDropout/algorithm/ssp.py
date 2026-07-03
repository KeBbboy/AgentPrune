"""SSP（Stage Soft Pruning）核心算法。

剪枝规则（与项目方法文档 v4 对齐）：
    * Planner 把 query 拆成 K 个阶段；
    * 每个 agent 由 Embedder 编码为向量，与 K 个阶段描述算 cosine
      相似度矩阵 S ∈ ℝ^{N×K}；
    * 经过 softmax 得到软指派矩阵 α ∈ ℝ^{N×K}；
    * 对每对 agent (i, j)：
        - 若 c_i == c_j（同阶段），**保留边**（硬边）；
        - 否则 w_ij = α_i · α_j，若 w_ij ≥ τ 则保留，否则删边。

提供两个对外入口：
    * :func:`ssp_pseudo_edges` —— 纯函数，输入 N×K 软指派矩阵，返回
      一个 ``(N, N)`` 的 0/1 邻接矩阵，便于独立单测；
    * :func:`apply_ssp` —— 把 SSP 直接作用到一个 :class:`Graph` 实例：
      通过 ``add_successor``/``clear_spatial_connection`` 修改拓扑。

注意：
    * :func:`apply_ssp` 会改 ``spatial_predecessors/successors``，但
      ``Graph.construct_spatial_connection`` 会在每一轮 inference 重新
      根据 ``fixed_spatial_masks`` 重建边，因此 :func:`apply_ssp` 主要
      用于：(a) 静态可视化，(b) 测试阶段验证剪枝规则。
    * 真正集成到 :class:`Graph` 应通过在
      :func:`experiments.run_mmlu.get_kwargs` 中新增 ``'SSP'`` mode，
      用 :func:`ssp_fixed_spatial_masks` 生成 ``fixed_spatial_masks``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from AgentDropout.algorithm.stages import StageDAG
from AgentDropout.embeddings.base import Embedder


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class SSPConfig:
    """SSP 算法的超参数。

    Attributes:
        tau: 跨阶段边保留阈值，w_ij = α_i·α_j ≥ τ 时保留。τ=1.0
            等价于硬切（跨阶段边全删）；τ=0 等价于完全图。
        temperature: softmax 温度，T 越小越接近硬指派，T 越大越平滑。
        self_loop: 是否允许 agent 与自身的边（默认 False）。
    """

    tau: float = 0.3
    temperature: float = 1.0
    self_loop: bool = False


# ---------------------------------------------------------------------------
# 纯函数：算邻接矩阵
# ---------------------------------------------------------------------------


def soft_assignment(
    profile_emb: np.ndarray,
    stage_emb: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """计算 agent 到阶段的软指派矩阵 α ∈ ℝ^{N×K}。

    Args:
        profile_emb: (N, D) 已归一化的 agent profile 向量。
        stage_emb: (K, D) 已归一化的阶段描述向量。
        temperature: softmax 温度。

    Returns:
        np.ndarray: (N, K) 软指派，每行是 agent 在 K 个阶段的概率分布。
    """
    if temperature <= 0:
        raise ValueError(f"temperature 必须 > 0，实际 {temperature}")
    # cosine 相似度矩阵（行已归一化）
    sims = profile_emb @ stage_emb.T  # (N, K)
    # softmax with temperature
    scaled = sims / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)  # 数值稳定
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def ssp_pseudo_edges(
    alpha: np.ndarray,
    config: SSPConfig,
) -> np.ndarray:
    """由软指派矩阵计算 0/1 邻接矩阵。

    Args:
        alpha: (N, K) 软指派矩阵，每行 ∈ Δ^{K-1}。
        config: SSP 配置。

    Returns:
        np.ndarray: (N, N) 0/1 对称矩阵，1 表示保留边，0 表示剪掉。
    """
    if alpha.ndim != 2:
        raise ValueError(f"alpha 必须是 2D，实际形状 {alpha.shape}")

    N, K = alpha.shape
    if N == 0:
        return np.zeros((0, 0), dtype=np.int8)

    # 跨阶段边权重矩阵：W = α α^T
    cross_weight = alpha @ alpha.T  # (N, N)

    # 硬指派（用于"同阶段保留"判定）：用最大分量索引
    hard = np.argmax(alpha, axis=1)  # (N,)

    # 初始化为 0
    adj = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        for j in range(N):
            if i == j and not config.self_loop:
                continue
            if i == j and config.self_loop:
                adj[i, j] = 1
                continue
            if hard[i] == hard[j]:
                adj[i, j] = 1
            elif float(cross_weight[i, j]) >= config.tau:
                adj[i, j] = 1
            # else: 保持 0

    # 保持对称（i→j 与 j→i 应等价）
    adj = np.maximum(adj, adj.T)
    return adj


def ssp_fixed_spatial_masks(
    profile_texts: Sequence[str],
    stage_dag: StageDAG,
    embedder: Embedder,
    config: SSPConfig,
) -> List[List[int]]:
    """算 ``fixed_spatial_masks`` 风格的 N×N 0/1 矩阵。

    这是与 :class:`Graph` 兼容的接口，返回 ``List[List[int]]``，
    对角线为 0。

    Args:
        profile_texts: 每个 agent 的 profile 文本，长度 = N。
        stage_dag: Planner 输出。
        embedder: :class:`Embedder` 实例。
        config: SSP 配置。

    Returns:
        List[List[int]]: N×N 矩阵。
    """
    if not profile_texts:
        raise ValueError("profile_texts 不能为空")
    if stage_dag.num_stages == 0:
        raise ValueError("stage_dag 至少需要 1 个阶段")

    profile_emb = embedder.encode(list(profile_texts))
    stage_emb = embedder.encode(stage_dag.descriptions())
    if profile_emb.size == 0 or stage_emb.size == 0:
        raise RuntimeError("Embedder 返回空向量")

    alpha = soft_assignment(profile_emb, stage_emb, config.temperature)
    adj = ssp_pseudo_edges(alpha, config)

    # 对角线强制为 0（与 Graph 默认 fixed_spatial_masks 一致）
    np.fill_diagonal(adj, 0)

    return adj.astype(int).tolist()


# ---------------------------------------------------------------------------
# 在 Graph 上直接应用
# ---------------------------------------------------------------------------


def apply_ssp(
    graph,
    stage_dag: StageDAG,
    embedder: Embedder,
    config: Optional[SSPConfig] = None,
) -> dict:
    """把 SSP 直接作用到一个已构造的 :class:`Graph`。

    会：
        1. 收集每个 agent 的 ``role`` 作为 profile 文本；
        2. 跑 :func:`ssp_fixed_spatial_masks` 得到邻接矩阵；
        3. 调 ``graph.clear_spatial_connection`` 清掉旧边；
        4. 按邻接矩阵调 ``add_successor`` 重建边。

    Args:
        graph: :class:`AgentDropout.graph.graph.Graph` 实例。
        stage_dag: Planner 输出。
        embedder: :class:`Embedder` 实例。
        config: SSP 配置；默认 ``SSPConfig()``。

    Returns:
        dict: ``{"num_edges": int, "density": float, "stage_assignment": List[int]}``
    """
    if config is None:
        config = SSPConfig()

    node_ids = list(graph.nodes.keys())
    profile_texts = [getattr(graph.nodes[nid], "role", "") or nid for nid in node_ids]

    adj = ssp_fixed_spatial_masks(profile_texts, stage_dag, embedder, config)

    # 清空并重建
    graph.clear_spatial_connection()
    n = len(node_ids)
    num_edges = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if adj[i][j]:
                graph.nodes[node_ids[i]].add_successor(
                    graph.nodes[node_ids[j]], st="spatial"
                )
                num_edges += 1

    # 报告：硬指派分布（用最大分量）
    profile_emb = embedder.encode(profile_texts)
    stage_emb = embedder.encode(stage_dag.descriptions())
    alpha = soft_assignment(profile_emb, stage_emb, config.temperature)
    hard = np.argmax(alpha, axis=1).tolist()

    full = n * (n - 1)
    density = (num_edges / full) if full > 0 else 0.0
    return {
        "num_edges": num_edges,
        "density": density,
        "stage_assignment": hard,
    }
