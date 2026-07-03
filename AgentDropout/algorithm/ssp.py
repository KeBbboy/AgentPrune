"""SSP（Stage Soft Pruning）核心算法。

提供两种软剪枝策略（不再保留硬剪枝 v1）：

1. **Top-k 模式（默认）**：每个 agent 保留 top-k 强边（k=3）。
   解决"少数派被剪断"和"孤立 stage 0 边"问题。
2. **Soft 模式**：返回 α·α 浮点边权矩阵（>1e-3 视为有边）。

公开 API：
    * :func:`soft_assignment` —— 软指派矩阵 α
    * :func:`ssp_soft_weights` —— 浮点边权矩阵（soft 模式）
    * :func:`ssp_topk_masks` —— top-k 邻接矩阵（topk 模式，默认）
    * :func:`ssp_fixed_spatial_masks` —— Graph 兼容的 0/1 mask
    * :func:`apply_ssp` —— 直接作用到 Graph 实例
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from AgentDropout.algorithm.stages import StageDAG
from AgentDropout.embeddings.base import Embedder


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class SSPConfig:
    """SSP v2 算法的超参数。

    Attributes:
        temperature: softmax 温度，T 越小越接近硬指派，T 越大越平滑。
        self_loop: 是否允许 agent 与自身的边。
        mode: 邻接矩阵生成策略，可选 'soft' / 'topk'。
        topk: 仅 mode='topk' 时使用，每个节点保留的 top-k 边数。
        min_edges: 防御性下限，每个节点至少保留 min_edges 条边
            （避免 N 很小 + k 很大时退化）。
    """

    temperature: float = 1.0
    self_loop: bool = False
    mode: str = "topk"  # 'soft' | 'topk'
    topk: int = 3
    min_edges: int = 1


# ---------------------------------------------------------------------------
# 纯函数
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
    sims = profile_emb @ stage_emb.T  # (N, K)
    scaled = sims / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def _ensure_symmetric(mat: np.ndarray) -> np.ndarray:
    """对非对角线取 max 强制对称。"""
    return np.maximum(mat, mat.T)


def ssp_soft_weights(
    alpha: np.ndarray,
    config: SSPConfig,
) -> np.ndarray:
    """浮点边权矩阵（mode='soft'）。

    W_ij = α_i · α_j（连续值，范围 [0, 1]）。
    """
    if alpha.ndim != 2:
        raise ValueError(f"alpha 必须是 2D，实际形状 {alpha.shape}")
    weights = alpha @ alpha.T
    weights = _ensure_symmetric(weights)
    if not config.self_loop:
        np.fill_diagonal(weights, 0.0)
    return weights


def ssp_topk_masks(
    alpha: np.ndarray,
    config: SSPConfig,
) -> np.ndarray:
    """Top-k 邻接矩阵（mode='topk'，默认）。

    每个 agent 保留 top-k 强边（按 w_ij = α_i·α_j 排序）。
    解决"硬剪枝切断少数派"和"孤立 stage 0 边"两个问题。

    行为细节：
        * k = min(config.topk, N-1)
        * 至少保留 config.min_edges 条边
        * 自身边遵循 self_loop
    """
    if alpha.ndim != 2:
        raise ValueError(f"alpha 必须是 2D，实际形状 {alpha.shape}")
    N, K = alpha.shape
    if N == 0:
        return np.zeros((0, 0), dtype=np.int8)
    if N == 1:
        return np.zeros((1, 1), dtype=np.int8)

    cross_weight = alpha @ alpha.T
    if not config.self_loop:
        np.fill_diagonal(cross_weight, -np.inf)

    k = min(config.topk, N - 1)
    k = max(k, config.min_edges)

    if k >= N:
        adj = np.ones((N, N), dtype=np.int8)
        if not config.self_loop:
            np.fill_diagonal(adj, 0)
        return _ensure_symmetric(adj)

    # argpartition O(N log k)
    topk_idx = np.argpartition(-cross_weight, kth=k - 1, axis=1)[:, :k]

    adj = np.zeros((N, N), dtype=np.int8)
    rows = np.repeat(np.arange(N), k)
    cols = topk_idx.flatten()
    adj[rows, cols] = 1
    return _ensure_symmetric(adj)


# ---------------------------------------------------------------------------
# Graph 集成入口
# ---------------------------------------------------------------------------


def ssp_fixed_spatial_masks(
    profile_texts: Sequence[str],
    stage_dag: StageDAG,
    embedder: Embedder,
    config: SSPConfig,
) -> List[List[int]]:
    """算 Graph 兼容的 N×N 0/1 邻接矩阵。

    根据 ``config.mode`` 选 'soft' / 'topk'。
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

    if config.mode == "soft":
        weights = ssp_soft_weights(alpha, config)
        adj = (weights >= 1e-3).astype(np.int8)
    elif config.mode == "topk":
        adj = ssp_topk_masks(alpha, config)
    else:
        raise ValueError(f"未知 mode={config.mode!r}，应为 soft/topk")

    np.fill_diagonal(adj, 0)
    return adj.astype(int).tolist()


def apply_ssp(
    graph,
    stage_dag: StageDAG,
    embedder: Embedder,
    config: Optional[SSPConfig] = None,
) -> dict:
    """把 SSP 直接作用到一个已构造的 Graph。

    Returns:
        dict: 包含 num_edges/density/stage_assignment/sparsity。
    """
    if config is None:
        config = SSPConfig()

    node_ids = list(graph.nodes.keys())
    profile_texts = [getattr(graph.nodes[nid], "role", "") or nid for nid in node_ids]

    adj = ssp_fixed_spatial_masks(profile_texts, stage_dag, embedder, config)

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
        "config_mode": config.mode,
        "sparsity": 1.0 - density,
    }
