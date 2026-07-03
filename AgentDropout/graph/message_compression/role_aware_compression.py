"""模块 1: 按 self_role 差异化压缩前驱消息 (感知 reasoning/answer 结构)。

由开关 ``--role_aware_compression`` 触发, 仅当该开关为 True 时本模块才生效。

感知 structured payload:
  - ``payload["message_type"] == "structured"``: payload 是 dict
      ``{"reasoning", "answer", "full_text"}`` (来自 ``Node._resolve_message_payload``
      拆 ``LLMResponse`` 的结果). 本模块只压 ``reasoning``, 不动 ``answer``
  - ``payload["message_type"] == "full"`` 或缺省: payload["output"] 为字符串,
      退化为整体字符截断 (兼容历史用法)

压缩策略 (role -> max_chars):
  - ANALYTIC_ROLES  (AnalyzeAgent / CriticAgent / MathSolver / Coder): 4096
  - 其他: 1024
  - 超出阈值时, 仅在 reasoning / output 字符串内做"头/尾"截断, 中间替换为占位
"""
from __future__ import annotations

from typing import Any, Dict

from AgentDropout.graph.message_compression.config import (
    MessageCompressionConfig,
)


# 这两个集合是模块自己的实现选择, 不暴露给 CLI
ANALYTIC_ROLES = {"AnalyzeAgent", "CriticAgent", "MathSolver", "Coder"}
ANALYTIC_MAX_CHARS = 4096
NON_ANALYTIC_MAX_CHARS = 1024


def _truncate(text: str, max_chars: int, tag: str) -> str:
    """超过阈值时, 头/尾各取 max_chars/2, 中间替换为占位文本."""
    if not isinstance(text, str) or len(text) <= max_chars:
        return text
    half = max_chars // 2
    head = text[:half]
    tail = text[-half:]
    return f"{head}\n... [{tag} @ {len(text)}] ...\n{tail}"


def _compress_structured(payload: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    """``message_type == "structured"``: 只压 reasoning, answer 必须原样保留.

    上游 ``Node.get_spatial_info`` 把 ``_resolve_message_payload`` 拆出来的
    ``{"reasoning","answer","full_text"}`` 放在 ``payload["output"]`` 字段里.
    因此 structured 路径同时支持:
      - ``payload["reasoning"]`` 平铺形式 (若上游选择平铺)
      - ``payload["output"]["reasoning"]`` 嵌套形式 (若上游选择嵌套)
    """
    new_payload = dict(payload)

    # 优先取平铺, 否则从嵌套 output dict 拿
    reasoning = new_payload.get("reasoning")
    reasoning_source = "top"
    if not isinstance(reasoning, str):
        nested = new_payload.get("output")
        if isinstance(nested, dict):
            inner = nested.get("reasoning")
            if isinstance(inner, str):
                reasoning = inner
                reasoning_source = "nested"

    if isinstance(reasoning, str) and len(reasoning) > max_chars:
        new_reasoning = _truncate(reasoning, max_chars, "role-aware reasoning-trunc")
        if reasoning_source == "top":
            new_payload["reasoning"] = new_reasoning
        else:
            new_payload["output"] = dict(new_payload["output"])
            new_payload["output"]["reasoning"] = new_reasoning
    # answer 字段在此模块里严格不动; 若还需要整体 full_text, 让上游重新拼
    return new_payload


class UnstructuredPayloadError(RuntimeError):
    """``message_type != "structured"`` 时本模块拒绝压缩, 主动抛错.

    保留该异常类以支持 :class:`MessageCompressionConfig.strict_unstructured`:
    单元测试可通过 ``strict_unstructured=True`` 让模块在非 structured
    payload 上抛错, 验证设计意图. **运行时默认 ``False``**, 避免真实数据
    中 ``async_execute`` 压扁 ``LLMResponse`` 字段后整个 Graph 静默报
    错而无法产生任何候选答案.
    """


def compress_messages_role_aware(
    info: Dict[str, Dict[str, Any]],
    cfg: MessageCompressionConfig,
    self_role: str,
) -> Dict[str, Dict[str, Any]]:
    """模块 1 入口: 供 Node.get_*_info 调用.

    - cfg.role_aware_compression=False 或 info 为空, 原样返回.
    - payload["message_type"] == "structured" 时, 只压 reasoning.
    - 其他情况: 运行时默认 noop (与关掉该模块等价), 仅打 ``warnings.warn``
      提示; 当 ``cfg.strict_unstructured=True`` 时抛 :class:`UnstructuredPayloadError`.
    """
    if not cfg.role_aware_compression or not info:
        return info

    is_analytic = self_role in ANALYTIC_ROLES
    max_chars = ANALYTIC_MAX_CHARS if is_analytic else NON_ANALYTIC_MAX_CHARS

    out: Dict[str, Dict[str, Any]] = {}
    for key, payload in info.items():
        msg_type = payload.get("message_type")
        if msg_type != "structured":
            out_val_repr = type(payload.get("output")).__name__
            if getattr(cfg, "strict_unstructured", False):
                raise UnstructuredPayloadError(
                    "role_aware_compression 仅压缩 structured payload 的 reasoning; "
                    f"检测到非 structured payload (key={key!r}, message_type={msg_type!r}, "
                    f"output type={out_val_repr}). 请先确保前驱 Node 已走 LLMResponse "
                    "路径使 payload.message_type == 'structured'."
                )
            # 运行时 noop: 上一轮 commit 76f27ea 误以为生产链路稳定提供
            # structured payload, 实际 Node.async_execute 在 commit 之前就已经把
            # LLMResponse 压扁为 full_text 字符串, message_type 永远是 "full",
            # 严格报错会让所有 node 100% 抛错, Graph 没法产生任何候选答案.
            import warnings as _w
            _w.warn(
                f"role_aware_compression 跳过非 structured payload "
                f"(key={key!r}, message_type={msg_type!r}, output type={out_val_repr}); "
                "此时模块行为等价于 cfg.role_aware_compression=False.",
                RuntimeWarning,
                stacklevel=2,
            )
            out[key] = payload
            continue
        out[key] = _compress_structured(payload, max_chars)
    return out
