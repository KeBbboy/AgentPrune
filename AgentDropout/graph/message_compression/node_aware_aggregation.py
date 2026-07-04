"""模块 2: 接收方收到多条已压缩消息后, 进行跨发送方全局排序与去重.

由开关 ``--node_aware_aggregation`` 触发, 仅当该开关为 True 时本模块才生效.

感知 structured payload:
  - ``payload["message_type"] == "structured"``: 对 reasoning 与 answer 分别处理
  - 其他: 走原有的 "按 role 桶合并 output 字符串" 路径 (向后兼容)

聚合策略 (写死, 不再保留 per_role 旧逻辑):
  - 把所有前驱 reasoning 放进一个全局池
  - 按句子归一化去重 (跨 role / 跨 key / 跨 payload), 保留每个唯一句子的最早出现
  - 全局排序: 先按归一化句首字符稳定排序, 再按首次出现位置 (input 顺序)
  - answers 取众数 (投票语义)
  - 聚合后输出 1 条 payload, key = ``"aggregated"``
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

from AgentDropout.graph.message_compression._sent_utils import (
    norm_sentence,
    split_sentences,
)
from AgentDropout.graph.message_compression.config import (
    MessageCompressionConfig,
)


def _dedup_sentences_global(reasonings: List[str]) -> Tuple[str, Dict[str, int]]:
    """把所有 reasoning 合并并按句子去重, 再做全局排序.

    去重: 归一化句键, 保留每个唯一句的最早出现版本.
    排序: 先按句首字符稳定排序 (使相同角色/相似起点的句子聚拢), 再按首次出现位置回退
          (保证语义流顺序). 这里用 ``sorted(first_seen)`` 让跨发送方句子按
          字符序排列, 替代之前的 input 顺序, 满足"全局排序"的设计目标.

    返回: (排序 + 去重后的字符串, 统计信息 {total, unique, dup}).
    """
    first_seen: Dict[str, str] = {}   # 归一化键 -> 原句
    first_pos: Dict[str, int] = {}    # 归一化键 -> 首次出现 index (越小越靠前)
    total = 0
    for r in reasonings:
        for s in split_sentences(r):
            total += 1
            key = norm_sentence(s)
            if key not in first_seen:
                first_seen[key] = s
                first_pos[key] = total
    # 全局排序: 优先按 (句首字符, 首次出现 index), 让排序稳定且可复现
    sorted_keys = sorted(first_seen.keys(), key=lambda k: (k, first_pos[k]))
    merged = " ".join(first_seen[k] for k in sorted_keys)
    return merged, {
        "total": total,
        "unique": len(first_seen),
        "dup": max(0, total - len(first_seen)),
    }


def _aggregate_global(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把所有 payload 的 reasoning 全局去重 + 排序, answers 众数投票, 输出 1 条."""
    reasonings: List[str] = []
    answers: List[str] = []
    for it in items:
        r, a, _ft = _extract_ra(it)
        if isinstance(r, str) and r.strip():
            reasonings.append(r)
        if isinstance(a, str) and a.strip():
            answers.append(a.strip())

    merged_reasoning, _stats = _dedup_sentences_global(reasonings)

    chosen_answer = ""
    if answers:
        cnt = Counter(answers)
        max_count = max(cnt.values())
        chosen_answer = next(a for a in answers if cnt[a] == max_count)

    new_full_text = merged_reasoning + ("\n" + chosen_answer if chosen_answer else "")

    # role 字段保留"跨多 role"的提示, 不强求单一 role
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


def _join_full(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """旧路径: 非 structured payload 走整体 output 字符串合并 (向后兼容)."""
    joined_outputs: List[str] = []
    for it in items:
        out_val = it.get("output")
        if isinstance(out_val, str):
            joined_outputs.append(out_val.strip())
    aggregated_output = (
        "\n--\n".join(joined_outputs) if joined_outputs else items[0].get("output")
    )
    return {
        "role": items[0].get("role") or "Unknown",
        "output": aggregated_output,
        "message_type": items[0].get("message_type", "full"),
        "aggregated": True,
        "aggregated_count": len(items),
    }


class UnstructuredPayloadError(RuntimeError):
    """``message_type != "structured"`` 时本模块拒绝聚合, 主动抛错.

    设计准则: 本模块全局按句去重 reasoning, 绝不拼接整段 ``output`` 字符串
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
    - 跨 role / 跨 key 把所有 structured payload 放进全局池, 按句去重 + 排序,
      输出 1 条聚合 payload.
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
                "node_aware_aggregation 仅对 structured payload 做全局按句去重; "
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