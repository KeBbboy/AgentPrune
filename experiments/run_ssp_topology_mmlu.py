"""在 MMLU 第一条样本上跑 SSP 剪枝流程，输出最终通信拓扑。

不调真实 LLM（不进入推理流程），只跑：
    1. 取 MMLU dev 集第一条样本的 question；
    2. 构造 N 个 AnalyzeAgent + 它们的 role；
    3. Planner 用预定义 DAG（绕开 LLM 调用）划分阶段；
    4. MiniLM Embedding 计算 agent role × stage description 相似度；
    5. SSP 按 τ 阈值剪枝，得到最终邻接矩阵。

打印：
    * 原始 query；
    * Planner 输出 DAG；
    * 每个 stage 的 description embedding 与各 agent 的 cosine 相似度；
    * 软指派矩阵 α；
    * 硬指派分组；
    * SSP 邻接矩阵 + 边数统计；
    * 拓扑可视化（每个 stage 内部的 clique + 跨阶段的桥接边）。

用法：
    python -m experiments.run_ssp_topology_mmlu --num_agents 10 --tau 0.3
"""

import argparse
import asyncio
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from AgentDropout.algorithm.planner import StagePlanner, _estimate_complexity
from AgentDropout.algorithm.ssp import (
    SSPConfig,
    apply_ssp,
    soft_assignment,
    ssp_fixed_spatial_masks,
    ssp_pseudo_edges,
)
from AgentDropout.algorithm.stages import ComplexityLevel, Stage, StageDAG
from AgentDropout.embeddings import MiniLMEmbedder
from AgentDropout.prompt.prompt_set_registry import PromptSetRegistry

from dataset.mmlu_dataset import MMLUDataset


# ---------------------------------------------------------------------------
# 预定义 DAG（不调 LLM，保证可复现）
# ---------------------------------------------------------------------------

PREDEFINED_DAGS = {
    # 3 阶段：分析 → 推理 → 验证
    "analyze_reason_verify": StageDAG(
        stages=[
            Stage("S1", "Analyze", "阅读题目，识别关键概念与给定条件，拆解为可执行子问题。"),
            Stage("S2", "Reason", "基于上一步的分析逐步推理或计算，得到候选答案。"),
            Stage("S3", "Verify", "复核推理过程的逻辑一致性与计算正确性，输出最终选项。"),
        ],
        edges=[("S1", "S2"), ("S2", "S3")],
    ),
    # 4 阶段：分析 → 回忆知识 → 推理 → 验证
    # 注意：description 要具体且彼此区分，避免 softmax 把所有 agent
    # 都分到最"宽泛"的那个阶段。
    "analyze_recall_reason_verify": StageDAG(
        stages=[
            Stage("S1", "Analyze",  "Classify the question into one specific academic discipline and outline the question structure."),
            Stage("S2", "Recall",   "Retrieve domain-specific facts, formulas, theorems, or definitions directly relevant to the identified subject."),
            Stage("S3", "Reason",   "Apply logical deduction, perform calculations step by step, and derive the candidate answer from the recalled knowledge."),
            Stage("S4", "Verify",   "Critically inspect the reasoning chain, point out potential issues in other agents' analysis, and confirm the final selected option."),
        ],
        edges=[("S1", "S2"), ("S2", "S3"), ("S3", "S4")],
    ),
    # 5 阶段：分析 → 知识回忆 → 假设生成 → 评估 → 验证
    "full_pipeline": StageDAG(
        stages=[
            Stage("S1", "Analyze",     "Decompose the question into subject domain, constraints, and the precise query."),
            Stage("S2", "Recall",      "List the domain-specific facts and formulas needed."),
            Stage("S3", "Hypothesize", "Generate multiple candidate answers or interpretations."),
            Stage("S4", "Evaluate",    "Compare candidates against evidence and discard weak ones."),
            Stage("S5", "Verify",      "Lock in the final answer after a final consistency check."),
        ],
        edges=[("S1", "S2"), ("S2", "S3"), ("S3", "S4"), ("S4", "S5")],
    ),
}


# ---------------------------------------------------------------------------
# 可视化辅助
# ---------------------------------------------------------------------------


def print_matrix(mat: np.ndarray, row_labels, col_labels, title: str, fmt: str = "{:.3f}"):
    """打印带行列标签的矩阵。"""
    print(f"\n=== {title} ===")
    col_w = max(max(len(str(c)) for c in col_labels), 6)
    row_w = max(max(len(str(r)) for r in row_labels), 4)
    # 表头
    header = " " * (row_w + 2) + " ".join(f"{str(c):>{col_w}}" for c in col_labels)
    print(header)
    for r, row in zip(row_labels, mat):
        cells = " ".join(fmt.format(v).rjust(col_w) for v in row)
        print(f"{str(r):<{row_w}}  {cells}")


def print_adj_with_groups(adj: np.ndarray, hard: list, stage_names, node_ids):
    """把邻接矩阵按硬指派分组打印。"""
    print("\n=== 最终通信拓扑（按阶段分组）===")
    n = len(node_ids)
    # 按阶段分组
    groups: dict[int, list[int]] = {}
    for idx, h in enumerate(hard):
        groups.setdefault(h, []).append(idx)

    # 组内（clique）
    intra_edges = 0
    for k, idxs in sorted(groups.items()):
        sn = stage_names[k]
        ids = [node_ids[i] for i in idxs]
        print(f"\n[Stage {k}] {sn}: agents = {ids}")
        for i_pos, i in enumerate(idxs):
            for j in idxs[i_pos + 1:]:
                if adj[i, j]:
                    print(f"  intra: {node_ids[i]} <-> {node_ids[j]}")
                    intra_edges += 1

    # 跨阶段（最多 N(N-1)/2 - intra）
    cross_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if hard[i] != hard[j] and adj[i, j]:
                cross_edges.append((node_ids[i], node_ids[j]))
    print(f"\n跨阶段边（共 {len(cross_edges)} 条，每对跨阶段 agent 最多 1 条）：")
    for a, b in cross_edges:
        print(f"  cross: {a} <-> {b}")

    print(f"\n组内边数: {intra_edges}  跨阶段边数: {len(cross_edges)}  总边数: {intra_edges + len(cross_edges)}")
    full = n * (n - 1) // 2
    density = (intra_edges + len(cross_edges)) / full if full > 0 else 0.0
    print(f"完全图边数: {full}  剪枝后密度: {density:.2%}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def get_first_mmlu_sample():
    """读取 MMLU dev 集第一条样本。"""
    try:
        ds = MMLUDataset("dev")
    except Exception as exc:
        print(f"[WARN] 无法加载 MMLU dev 集，使用占位 question：{exc}")
        return {
            "question": "Which of the following is a primary function of the mitochondria?",
            "answer": "A",
        }
    if len(ds) == 0:
        return {
            "question": "Which of the following is a primary function of the mitochondria?",
            "answer": "A",
        }
    return ds[0]


def get_agent_roles(num_agents: int, reset_cycle: bool = True) -> tuple[list[str], list[str]]:
    """构造 N 个 AnalyzeAgent 的 (role_name, role_description) 列表。

    重要：直接 import 模块级的 ``roles`` cycle，**不要**用
    ``MMLUPromptSet.get_role()``。后者返回 ``next(roles)`` 的字符串结果，
    但 ``get_role()`` 的实现返回 ``next(roles)`` 是 cycle 的字符串元素，
    若调用方对返回值再 ``iter()`` 会按字符迭代（'Knowlegable Expert'
    → K/n/o/w/l/e/g/...）。我们直接用 cycle 拿到完整 role 名。

    embedding 用 ``ROLE_DESCRIPTION[role]`` 的完整描述文本，因为
    role 名（如 ``"Knowlegable Expert"``）信息量不足，无法区分学科能力。

    Returns:
        (role_names, role_profiles)：role_names 用于显示，
        role_profiles 用于 embedding。
    """
    # 直接拿到模块级的 cycle，避免 iter 字符串的坑
    from AgentDropout.prompt.mmlu_prompt_set import ROLE_DESCRIPTION, roles as role_cycle

    if reset_cycle:
        # 用 itertools.islice 重新生成一个从头开始的迭代器
        import itertools
        role_iter = itertools.islice(itertools.cycle(role_cycle), num_agents)
        role_names = list(role_iter)
    else:
        role_names = [next(role_cycle) for _ in range(num_agents)]

    role_profiles: list[str] = []
    for r in role_names:
        desc = ROLE_DESCRIPTION.get(r, "")
        if desc:
            profile = f"You are a {r}.\n{desc.strip()}"
        else:
            profile = f"You are a {r}."
        role_profiles.append(profile)

    return role_names, role_profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=10, help="AnalyzeAgent 数量")
    parser.add_argument("--tau", type=float, default=0.3, help="SSP 跨阶段边阈值")
    parser.add_argument("--temperature", type=float, default=1.0, help="softmax 温度")
    parser.add_argument(
        "--dag",
        type=str,
        default="analyze_recall_reason_verify",
        choices=list(PREDEFINED_DAGS.keys()),
        help="预定义阶段 DAG",
    )
    parser.add_argument(
        "--force_complexity",
        type=str,
        default="auto",
        choices=["auto", "simple", "hard"],
        help="强制复杂度档位（默认 auto 由 query 长度判断）",
    )
    parser.add_argument(
        "--reset_cycle",
        action="store_true",
        default=True,
        help="从 role cycle 起点开始分配 role（默认开启，保证可复现）",
    )
    parser.add_argument(
        "--no_reset_cycle",
        dest="reset_cycle",
        action="store_false",
        help="使用模块级 cycle 当前状态",
    )
    args = parser.parse_args()

    # 1. 取第一条样本
    sample = get_first_mmlu_sample()
    question = sample.get("question") or sample.get("task") or ""
    print(f"[1] MMLU 第一条样本 question:\n    {question}")
    print(f"    ground truth answer: {sample.get('answer', '?')}")

    # 2. 复杂度档位
    if args.force_complexity == "auto":
        level = _estimate_complexity(question)
    elif args.force_complexity == "simple":
        level = ComplexityLevel.SIMPLE
    else:
        level = ComplexityLevel.HARD
    print(f"\n[2] 复杂度档位: {level.label} (K ∈ [{level.k_min}, {level.k_max}])")

    # 3. Planner：DAG 来源
    dag = PREDEFINED_DAGS[args.dag]
    print(f"\n[3] Planner 输出 StageDAG（预定义，不调 LLM）:")
    for s in dag.stages:
        print(f"    {s.id} [{s.name}] {s.description}")
    print(f"    edges: {dag.edges}")

    # 4. Agent role 列表
    role_names, role_profiles = get_agent_roles(args.num_agents, reset_cycle=args.reset_cycle)
    node_ids = [f"a{i}" for i in range(args.num_agents)]
    print(f"\n[4] {args.num_agents} 个 AnalyzeAgent 的 role:")
    for nid, r in zip(node_ids, role_names):
        print(f"    {nid}: {r}")

    # 5. Embedding 编码：用完整 ROLE_DESCRIPTION 而不是 role 名
    embedder = MiniLMEmbedder()
    role_emb = embedder.encode(role_profiles)
    stage_emb = embedder.encode(dag.descriptions())
    print(f"\n[5] Embedding:")
    print(f"    role_emb shape:    {role_emb.shape}")
    print(f"    stage_emb shape:   {stage_emb.shape}")

    # 6. Cosine 相似度矩阵（直接相乘，因为已经 L2 归一化）
    sim_matrix = role_emb @ stage_emb.T
    print_matrix(
        sim_matrix,
        row_labels=node_ids,
        col_labels=dag.stage_ids(),
        title="agent role × stage description 的 cosine 相似度",
        fmt="{:.3f}",
    )

    # 7. 软指派矩阵 α
    config = SSPConfig(tau=args.tau, temperature=args.temperature)
    alpha = soft_assignment(role_emb, stage_emb, temperature=config.temperature)
    print_matrix(
        alpha,
        row_labels=node_ids,
        col_labels=dag.stage_ids(),
        title=f"软指派矩阵 α (temperature={config.temperature})",
        fmt="{:.3f}",
    )

    # 8. 硬指派
    hard = np.argmax(alpha, axis=1).tolist()
    print(f"\n[6] 硬指派分组:")
    for stage_idx in range(dag.num_stages):
        ids = [node_ids[i] for i, h in enumerate(hard) if h == stage_idx]
        print(f"    {dag.stages[stage_idx].id} [{dag.stages[stage_idx].name}]: {ids}")

    # 9. SSP 邻接矩阵
    adj = ssp_pseudo_edges(alpha, config)
    np.fill_diagonal(adj, 0)

    print_matrix(
        adj,
        row_labels=node_ids,
        col_labels=node_ids,
        title=f"SSP 邻接矩阵 (τ={args.tau})",
        fmt="{:.0f}",
    )

    # 10. 按阶段分组的拓扑可视化
    stage_names = [s.name for s in dag.stages]
    print_adj_with_groups(adj, hard, stage_names, node_ids)


if __name__ == "__main__":
    main()
