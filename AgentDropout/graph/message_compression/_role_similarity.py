"""模块 1 辅助: 用 sentence-transformers (all-MiniLM-L6-v2) 算句子与接收方角色
描述的余弦相似度, 并按分数分布动态确定阈值.

设计要点:
  - 单例 encoder: 模块级懒加载, 第一次调用时下载/读取模型; 同一进程只
    load 一次, 后续调用复用, 避免重复加载耗时.
  - 动态阈值: 给定一组 (sentence, score) 对, 阈值取 ``max(median, mean -
    std)`` 让"高于均值一个标准差"或者"中位数"两者取较大, 始终保留约一
    半句子. 不可用 (NaN / 全 0) 时退化为 0.0.
  - 模型缺失时 (例如没装 sentence-transformers 或没有下载缓存): 退化为
    ``None`` 评分 (全部 0.0), 上层模块 1 应当识别并跳过.
"""
from __future__ import annotations

import math
from threading import Lock
from typing import List, Optional, Sequence, Tuple


_MODEL_NAME = "all-MiniLM-L6-v2"
_encoder = None
_encoder_lock = Lock()


def get_encoder():
    """懒加载单例 sentence-transformers encoder. 第一次调用慢, 后续快."""
    global _encoder
    if _encoder is not None:
        return _encoder
    with _encoder_lock:
        if _encoder is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                _encoder = SentenceTransformer(_MODEL_NAME)
            except Exception:  # pragma: no cover - 仅在缺失依赖时
                _encoder = False
                return False
    return _encoder


def encode(texts: Sequence[str]):
    enc = get_encoder()
    if enc is False:
        return None
    return enc.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)


def dynamic_threshold(scores: Sequence[float]) -> float:
    """按分数分布动态阈值: max(median, mean - std).

    不可用 (空序列 / 全 NaN) 时退化为 0.0. 此函数纯 numpy 友好, 不依赖 encoder.
    """
    vals: List[float] = [float(s) for s in scores if s is not None and not (isinstance(s, float) and math.isnan(s))]
    if not vals:
        return 0.0
    try:
        import numpy as np  # type: ignore
        arr = np.asarray(vals, dtype=float)
        median = float(np.median(arr))
        mean = float(np.mean(arr))
        std = float(np.std(arr))
    except Exception:
        sorted_v = sorted(vals)
        n = len(sorted_v)
        median = sorted_v[n // 2]
        mean = sum(sorted_v) / n
        var = sum((x - mean) ** 2 for x in sorted_v) / n
        std = math.sqrt(var)
    return max(median, mean - std)


# 接收方 role 描述: 用于 query 端. 与 prompt 语义对齐, 让 MiniLM 能算语义相似度.
ROLE_DESCRIPTIONS = {
    "AnalyzeAgent":     "An expert who deeply analyzes problems by breaking them into logical parts, considering evidence, weighing alternatives, and explaining reasoning step by step.",
    "CriticAgent":      "A critical reviewer who finds flaws, challenges assumptions, points out logical gaps, and demands rigorous justification.",
    "MathSolver":       "A precise mathematician who solves equations, derives formulas, performs calculations, and verifies numeric answers.",
    "Coder":            "A software engineer who writes, reviews, and debugs code, understands algorithms, data structures, and programming language semantics.",
    "Psychologist":     "A psychologist who interprets human behavior, motivation, emotion, and cognitive patterns.",
    "Historian":        "A historian who situates events in time, traces causes and consequences, and recalls factual context.",
    "Knowlegable Expert": "A broadly knowledgeable expert providing authoritative factual context across domains.",
    "FinalRefer":       "An aggregator that consults all prior agents' opinions and produces a single consolidated decision.",
    "FinalWriteCode":   "An agent that writes code based on prior analysis.",
    "FinalMajorVote":   "An agent that picks the answer with the most votes among prior opinions.",
    "Solver":           "A general problem solver integrating ideas from multiple expert perspectives.",
}


def role_description(role: str) -> str:
    """返回 ``role`` 的语义描述; 未知 role 时退化为角色名字面, 不抛错."""
    return ROLE_DESCRIPTIONS.get(role, role or "An expert integrating information from prior agents.")


def sentence_scores_for_role(
    sentences: Sequence[str], role: str
) -> Tuple[List[float], Optional[str]]:
    """算 ``sentences`` 中每个句子与 ``role`` 描述的余弦相似度.

    返回: (分数列表, 错误信息或 None). 失败时分数全部 0.0, 错误信息非 None.
    """
    if not sentences:
        return [], None
    desc = role_description(role)
    try:
        embs = encode([desc, *sentences])
    except Exception as exc:  # pragma: no cover
        return [0.0 for _ in sentences], f"encode failed: {exc}"
    if embs is None:
        return [0.0 for _ in sentences], "encoder unavailable"
    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        return [0.0 for _ in sentences], f"numpy unavailable: {exc}"
    q = embs[0]
    cs = embs[1:]
    sims = (q @ cs.T).flatten()
    return [float(x) for x in sims], None