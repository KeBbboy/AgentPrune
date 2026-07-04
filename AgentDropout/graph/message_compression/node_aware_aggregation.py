"""模块 2: 接收方收到模块 1 保留的多条边后, 跨边做相似度冗余检测并聚合.

由开关 ``--node_aware_aggregation`` 触发, 仅当该开关为 True 时本模块才生效.

聚合策略 (写死, 不暴露 CLI 子开关):
  - 对每个前驱 payload 的 reasoning 按句切分, 每条句携带发送方 role 标签
  - 跨 role / 跨 key 把所有句子放进全局池
  - 用 ``all-MiniLM-L6-v2`` 对每条句做 embedding, 句两两之间算余弦相似度
  - cos >= 0.70 视为同义冗余, 仅保留第一条 (最早出现)
  - 排序: (a) 角色在 ``info`` 中首次出现的顺序; (b) 同一角色内按推理
    时序; (c) 平局按归一化键字符序兜底
  - 每条保留句前缀 ``[Role]`` 标签, 让下游 LLM 识别来源 role
  - answers 取众数 (投票语义), 平局时取 input 首个
  - 输出 1 条聚合 payload, key = ``"aggregated::0"``

依赖: ``sentence-transformers`` (与模块 1 共享同一 encoder 单例).
模型不可用时退化为"保留全部句子, 仅按推理时序 + role 标签排列",
仅打 ``warnings.warn``, 不阻塞主链路.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

from AgentDropout.graph.message_compression._role_similarity import (
    encode,
)
from AgentDropout.graph.message_compression._sent_utils import (
    norm_sentence,
    split_sentences,
)
from AgentDropout.graph.message_compression.config import (
    MessageCompressionConfig,
)


# 冗余判定阈值: cos >= REDUNDANCY_THRESHOLD 视为同义冗余, 仅保留第一条.
REDUNDANCY_THRESHOLD = 0.70


def _filter_redundant_by_embedding(
    sentences: List[str],
) -> Tuple[List[int], List[float]]:
    """返回 (保留下标列表, 每一句相对最早保留句的最大 cos 相似度).

    算法: 顺序遍历; 对每个新句, 计算它与所有"已保留句"的 cos; 若最大 cos
    >= ``REDUNDANCY_THRESHOLD`` 则丢弃, 否则保留. 复杂度 O(n^2 * d), 对
    几十~几百句量级足够, 超过千句时需要换 ANN 索引.
    """
    n = len(sentences)
    if n == 0:
        return [], []
    kept_idx: List[int] = []
    max_sims: List[float] = [-1.0] * n
    try:
        embs = encode(sentences)
    except Exception:
        embs = None
    if embs is None:
        # 模型不可用: 全保留, 相似度填 0.0, 由 caller 决定是否走 fallback.
        return list(range(n)), [0.0] * n
    try:
        import numpy as np  # type: ignore
    except Exception:
        return list(range(n)), [0.0] * n
    embs_np = np.asarray(embs)
    for i in range(n):
        sims = embs_np[kept_idx] @ embs_np[i] if kept_idx else np.zeros(0)
        max_sim = float(sims.max()) if sims.size else -1.0
        max_sims[i] = max_sim
        if max_sim >= REDUNDANCY_THRESHOLD:
            continue  # 视为冗余, 跳过
        kept_idx.append(i)
    return kept_idx, max_sims


def _dedup_sentences_global(
    items: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, int], bool]:
    """把所有 payload 的 reasoning 合并并按句子相似度去重, 再按推理时序排序.

    返回: (排序 + 去重后的字符串, 统计信息 {total, kept, redundant}, embedding_ok).

    - ``total``: 输入句子总数
    - ``kept``: 经相似度判定后保留下来的句子数
    - ``redundant``: 被识别为冗余并丢弃的句子数 (cos >= 0.70)
    - ``embedding_ok``: True 表示相似度判定生效; False 表示模型不可用, 已
      退化保留全部句子, 仅按推理时序 + role 标签排列.
    """
    role_order: Dict[str, int] = {}
    collected: List[Dict[str, Any]] = []   # 顺序: [ {sent, role, pos, norm_key} ]
    total = 0
    for item in items:
        role = item.get("role") or "Unknown"
        if role not in role_order:
            role_order[role] = len(role_order)
        r, _a, _ft = _extract_ra(item)
        if not (isinstance(r, str) and r.strip()):
            continue
        for s in split_sentences(r):
            total += 1
            collected.append({
                "sent": s,
                "role": role,
                "pos": total,
                "norm_key": norm_sentence(s),
            })

    if not collected:
        return "", {"total": 0, "kept": 0, "redundant": 0}, True

    sentences = [c["sent"] for c in collected]
    kept_idx, _max_sims = _filter_redundant_by_embedding(sentences)
    embedding_ok = bool(kept_idx) and all(c["pos"] >= 0 for c in collected)

    kept_set = set(kept_idx)
    # 按 (role_order, pos, norm_key) 稳定排序; 只保留被保留下来的句.
    kept_items = [collected[i] for i in sorted(kept_idx, key=lambda i: (
        role_order.get(collected[i]["role"], 999),
        collected[i]["pos"],
        collected[i]["norm_key"],
    ))]

    merged_parts: List[str] = []
    cur_role: str | None = None
    for c in kept_items:
        r = c["role"]
        sent = c["sent"]
        if r != cur_role:
            if merged_parts:
                merged_parts.append("\n")
            merged_parts.append(f"[{r}] ")
            cur_role = r
        else:
            merged_parts.append(" ")
        merged_parts.append(sent)
    merged = "".join(merged_parts).strip()
    stats = {
        "total": total,
        "kept": len(kept_items),
        "redundant": max(0, total - len(kept_items)),
    }
    return merged, stats, embedding_ok


def _aggregate_global(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把所有 payload 的 reasoning 跨边去冗余 + 排序, answers 众数投票, 输出 1 条."""
    answers: List[str] = []
    for it in items:
        _r, a, _ft = _extract_ra(it)
        if isinstance(a, str) and a.strip():
            answers.append(a.strip())

    merged_reasoning, _stats, embedding_ok = _dedup_sentences_global(items)
    if not embedding_ok and merged_reasoning:
        # 模型不可用: 仅打 1 次 warn, 不阻塞.
        import warnings as _w
        _w.warn(
            "node_aware_aggregation 相似度冗余检测不可用; 退化保留全部句子, "
            "仅按推理时序 + role 标签排列. (cos 阈值 %.2f 未生效)" % REDUNDANCY_THRESHOLD,
            RuntimeWarning,
            stacklevel=2,
        )

    chosen_answer = ""
    if answers:
        cnt = Counter(answers)
        max_count = max(cnt.values())
        chosen_answer = next(a for a in answers if cnt[a] == max_count)

    new_full_text = merged_reasoning + ("\n" + chosen_answer if chosen_answer else "")

    roles_seen = sorted({(it.get("role") or "Unknown") for it in items if isinstance(it, dict)})
    role_label = "+".join(roles_seen) if roles_seen else "Unknown"

    return {
        "role": role_label,
        "reasoning": merged_reasoning,
        "answer": chosen_answer or None,
        "full_text": new_full_text,
        "message_type": "structured",
        "aggregated": True,
        "aggregated_count": len(items),
        "aggregation_scope": "global",
        "redundancy_threshold": REDUNDANCY_THRESHOLD,
        "source_roles": roles_seen,
    }


def _extract_ra(it: Dict[str, Any]) -> tuple[Any, Any, Any]:
    """从 payload 抽出 (reasoning, answer, full_text). 同时支持平铺与嵌套形式."""
    if isinstance(it.get("reasoning"), str) or isinstance(it.get("answer"), str):
        return it.get("reasoning"), it.get("answer"), it.get("full_text")
    out_val = it.get("output")
    if isinstance(out_val, dict):
        return out_val.get("reasoning"), out_val.get("answer"), out_val.get("full_text")
    return None, None, None


class UnstructuredPayloadError(RuntimeError):
    """``message_type != "structured"`` 时本模块拒绝聚合, 主动抛错.

    设计准则: 本模块按句去冗余 reasoning, 绝不拼接整段 ``output`` 字符串
    (无法分辨 reasoning 与 answer). 非 structured 路径一旦拼接就会把 answer
    混入 reasoning 上下文, 与设计准则冲突.
    """


def aggregate_node_aware(
    info: Dict[str, Dict[str, Any]],
    cfg: MessageCompressionConfig,
    self_role: str,
) -> Dict[str, Dict[str, Any]]:
    """模块 2 入口: 供 Node.get_*_info 调用.

    - cfg.node_aware_aggregation=False 或 info 为空, 原样返回.
    - 跨 role / 跨 key 把所有 structured payload 放进全局池, 按 embedding
      相似度 (cos >= 0.70) 判冗余, 输出 1 条聚合 payload. 每条 sentence
      前缀 ``[Role]`` 标签.
    - 任何 ``message_type != "structured"`` 的 payload: 运行时默认 noop
      (与关掉该模块等价, 仅打 warnings.warn); 当
      ``cfg.strict_unstructured=True`` 时抛 :class:`UnstructuredPayloadError`.
    """
    if not cfg.node_aware_aggregation or not info:
        return info

    bad: List[tuple[str, Any, str]] = []
    for key, payload in info.items():
        mt = payload.get("message_type")
        if mt != "structured":
            bad.append((key, mt, type(payload.get("output")).__name__))
    if bad:
        if getattr(cfg, "strict_unstructured", False):
            sample = bad[:3]
            raise UnstructuredPayloadError(
                "node_aware_aggregation 仅对 structured payload 做跨边相似度去冗余; "
                f"检测到 {len(bad)} 个非 structured payload (示例: "
                f"{sample}). 请先确保前驱 Node 已走 LLMResponse 路径使 "
                "payload.message_type == 'structured'."
            )
        import warnings as _w
        _w.warn(
            f"node_aware_aggregation 跳过: 检测到 {len(bad)} 个非 structured "
            f"payload (示例: {bad[:3]}). 运行时 noop, 与 cfg.node_aware_aggregation=False 等价.",
            RuntimeWarning,
            stacklevel=2,
        )
        return info

    items = list(info.values())
    return {"aggregated::0": _aggregate_global(items)}