"""模块 2: 按 self_role 重新聚合前驱输出 (感知 reasoning/answer 结构)。

由开关 ``--node_aware_aggregation`` 触发, 仅当该开关为 True 时本模块才生效。

感知 structured payload:
  - ``payload["message_type"] == "structured"``: 对 reasoning 与 answer 分别拼接
  - 其他: 走原有的 "按 role 桶合并 output 字符串" 路径 (向后兼容)

聚合策略:
  - 把同一 ``role`` 的前驱 payload 合并到一个桶
  - 不同 role 跨桶保留, 互不混合
  - 聚合后的 key = ``"<role>::<idx>"``, payload 标记 ``aggregated=True``
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from AgentDropout.graph.message_compression.config import (
    MessageCompressionConfig,
)


def _join_structured(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """同 role 桶里的多个 structured payload 合并: reasoning 与 answer 分别拼接.

    同时支持平铺与嵌套形式 (上游选择).
    """
    def _extract_ra(it: Dict[str, Any]) -> tuple[Any, Any]:
        # 平铺
        if isinstance(it.get("reasoning"), str) or isinstance(it.get("answer"), str):
            return it.get("reasoning"), it.get("answer")
        # 嵌套: output = {"reasoning", "answer", "full_text"} dict
        out_val = it.get("output")
        if isinstance(out_val, dict):
            return out_val.get("reasoning"), out_val.get("answer")
        return None, None

    base = items[0]
    joined_reasoning: List[str] = []
    joined_answer: List[str] = []
    for it in items:
        r, a = _extract_ra(it)
        if isinstance(r, str) and r.strip():
            joined_reasoning.append(r.strip())
        if isinstance(a, str) and a.strip():
            joined_answer.append(a.strip())
    base_out = base.get("output")
    base_full = base_out.get("full_text") if isinstance(base_out, dict) else base.get("full_text")
    return {
        "role": base.get("role", "Unknown"),
        "reasoning": "\n--\n".join(joined_reasoning) if joined_reasoning else (base.get("reasoning") if isinstance(base.get("reasoning"), str) else None),
        "answer":    "\n--\n".join(joined_answer)    if joined_answer    else (base.get("answer")    if isinstance(base.get("answer"),    str) else None),
        "full_text": base_full,
        "message_type": "structured",
        "aggregated": True,
        "aggregated_count": len(items),
    }


def _join_full(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """旧路径: 按 output 字符串合并."""
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

    设计准则: 本模块按 role 桶分别拼接 reasoning 与 answer, 绝不拼接整段
    ``output`` 字符串 (无法分辨 reasoning 与 answer). 非 structured 路径
    一旦拼接就会把 answer 混入 reasoning 上下文, 与设计准则冲突.
    """


def aggregate_node_aware(
    info: Dict[str, Dict[str, Any]],
    cfg: MessageCompressionConfig,
    self_role: str,
) -> Dict[str, Dict[str, Any]]:
    """模块 2 入口: 供 Node.get_*_info 调用.

    - cfg.node_aware_aggregation=False 或 info 为空, 原样返回.
    - 只有一个 role 桶, 原样返回 (无跨 role 信息不值得重排).
    - 任何 ``message_type != "structured"`` 的 payload: 运行时默认 noop
      (与关掉该模块等价, 仅打 warnings.warn); 当
      ``cfg.strict_unstructured=True`` 时抛 :class:`UnstructuredPayloadError`.
    """
    if not cfg.node_aware_aggregation or not info:
        return info

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for key, payload in info.items():
        role = payload.get("role") or "Unknown"
        buckets[role].append(payload)

    if len(buckets) <= 1:
        return info

    # 运行时 noop 路径: 上一轮 commit 76f27ea 误以为生产链路稳定提供
    # structured payload, 实际 Node.async_execute 在 commit 之前就已经把
    # LLMResponse 压扁为 full_text 字符串, 强行整段拼接会误伤 answer.
    bad: List[tuple[str, Any, str]] = []
    for key, payload in info.items():
        mt = payload.get("message_type")
        if mt != "structured":
            bad.append((key, mt, type(payload.get("output")).__name__))
    if bad:
        if getattr(cfg, "strict_unstructured", False):
            sample = bad[:3]
            raise UnstructuredPayloadError(
                "node_aware_aggregation 仅按 role 桶分别拼接 structured payload 的 "
                f"reasoning/answer; 检测到 {len(bad)} 个非 structured payload (示例: "
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

    new_info: Dict[str, Dict[str, Any]] = {}
    for role, items in buckets.items():
        first_key = items[0].get("role") or role
        new_info[f"{first_key}::{len(new_info)}"] = _join_structured(items)
    return new_info
