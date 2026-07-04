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
  - 超出阈值时, 按**句子**切分: 头尾各保留字符预算 ``max_chars/2`` 内的完整句子,
    中间用占位符替换 (切在句边界, 保证语义完整).
  - 同时同步压缩 ``full_text`` 字段, 让下游 ``info["output"]["full_text"]`` 也是压缩版.
"""
from __future__ import annotations

from typing import Any, Dict

from AgentDropout.graph.message_compression._sent_utils import head_tail_by_sentences
from AgentDropout.graph.message_compression.config import (
    MessageCompressionConfig,
)


# 这两个集合是模块自己的实现选择, 不暴露给 CLI
ANALYTIC_ROLES = {"AnalyzeAgent", "CriticAgent", "MathSolver", "Coder"}
ANALYTIC_MAX_CHARS = 4096
NON_ANALYTIC_MAX_CHARS = 1024


def _compress_structured(payload: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    """``message_type == "structured"``: 按句子切分压缩 reasoning, answer 原样保留.

    上游 ``Node.get_spatial_info`` 把 ``_resolve_message_payload`` 拆出来的
    ``{"reasoning","answer","full_text"}`` 放在 ``payload["output"]`` 字段里.
    因此 structured 路径同时支持:
      - ``payload["reasoning"]`` 平铺形式 (若上游选择平铺)
      - ``payload["output"]["reasoning"]`` 嵌套形式 (若上游选择嵌套)

    同时同步压缩 ``full_text`` 字段, 让下游走 ``info["output"]["full_text"]`` 时
    也能拿到压缩版, 真正省 token.
    """
    new_payload = dict(payload)

    # 找 reasoning 字段: 平铺优先, 否则从嵌套 output dict 拿
    reasoning = new_payload.get("reasoning")
    reasoning_source = "top"
    if not isinstance(reasoning, str):
        nested = new_payload.get("output")
        if isinstance(nested, dict) and isinstance(nested.get("reasoning"), str):
            reasoning = nested["reasoning"]
            reasoning_source = "nested"

    truncated_reasoning = None
    if isinstance(reasoning, str) and len(reasoning) > max_chars:
        truncated_reasoning = head_tail_by_sentences(
            reasoning, max_chars, "role-aware reasoning-trunc"
        )
        if reasoning_source == "top":
            new_payload["reasoning"] = truncated_reasoning
        else:
            new_payload["output"] = dict(new_payload["output"])
            new_payload["output"]["reasoning"] = truncated_reasoning

    # 同步压缩 full_text (平铺/嵌套都支持)
    if truncated_reasoning is not None:
        full_text = None
        if isinstance(new_payload.get("full_text"), str):
            full_text = new_payload["full_text"]
        elif isinstance(new_payload.get("output"), dict) and isinstance(
            new_payload["output"].get("full_text"), str
        ):
            full_text = new_payload["output"]["full_text"]
        if isinstance(full_text, str) and len(full_text) > max_chars:
            # full_text 通常包含 reasoning + answer. 先用 truncated_reasoning 替换段,
            # 再按句子切兜底.
            new_full_text = (
                full_text.replace(reasoning, truncated_reasoning)
                if isinstance(reasoning, str)
                else full_text
            )
            if len(new_full_text) > max_chars:
                new_full_text = head_tail_by_sentences(
                    new_full_text, max_chars, "role-aware fulltext-trunc"
                )
            new_payload["full_text"] = new_full_text
            if reasoning_source == "nested" and isinstance(new_payload.get("output"), dict):
                new_payload["output"]["full_text"] = new_full_text

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
    - payload["message_type"] == "structured" 时, 按句子切分压缩 reasoning.
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