"""模块 1: 按接收方角色对每条通信边按句做相似度过滤 (role_aware).

由开关 ``--role_aware_compression`` 触发, 仅当该开关为 True 时本模块才生效.

感知 structured payload:
  - ``payload["message_type"] == "structured"``: 对 reasoning 与 answer
    分别处理
  - 其他: 运行时默认 noop (与关掉该模块等价, 仅打 warnings.warn); 当
    ``cfg.strict_unstructured=True`` 时抛 :class:`UnstructuredPayloadError`

压缩策略 (按设计目标):
  - 对每条 structured payload: 先按句子切分 reasoning
  - 用 sentence-transformers ``all-MiniLM-L6-v2`` 算"句子 vs 接收方角色
    描述"的余弦相似度
  - 按分数分布动态设阈值 ``max(median, mean - std)``, 保留相似度高于
    阈值的句子, 其余丢弃
  - 同步裁剪 ``full_text`` (按句保留 + 头尾截断兜底), 让下游
    ``info["output"]["full_text"]`` 真正拿到压缩版

模型不可用时 (缺依赖 / 下载失败): 退化为"全部保留", 仅打 ``warnings.warn``,
不阻塞主链路. 任何 LLM 调用前都不会失败.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List

from AgentDropout.graph.message_compression._role_similarity import (
    dynamic_threshold,
    role_description,
    sentence_scores_for_role,
)
from AgentDropout.graph.message_compression._sent_utils import (
    head_tail_by_sentences,
    split_sentences,
)
from AgentDropout.graph.message_compression.config import (
    MessageCompressionConfig,
)


# 这两个集合是模块自己的实现选择, 不暴露给 CLI.
ANALYTIC_ROLES = {"AnalyzeAgent", "CriticAgent", "MathSolver", "Coder"}
ANALYTIC_MAX_CHARS = 4096
NON_ANALYTIC_MAX_CHARS = 1024


def _filter_sentences_by_relevance(
    sentences: List[str],
    role: str,
) -> tuple[List[str], List[float], float]:
    """对每个句子与接收方角色描述算余弦相似度, 按动态阈值过滤.

    返回: (保留句列表, 每个句子的分数, 阈值).
    """
    if not sentences:
        return [], [], 0.0
    scores, err = sentence_scores_for_role(sentences, role)
    if err is not None:
        warnings.warn(
            f"role_aware_compression 相似度评分不可用 ({err}); 退化保留全部句子.",
            RuntimeWarning,
            stacklevel=2,
        )
        return sentences, scores, 0.0
    thr = dynamic_threshold(scores)
    kept = [s for s, sc in zip(sentences, scores) if sc >= thr]
    return kept, scores, thr


def _truncate_to_budget(text: str, max_chars: int) -> str:
    """对 ``text`` 做按句头尾截断兜底, 总长不超过 ``max_chars``.

    底层 ``head_tail_by_sentences`` 已统一处理"无标点输入走纯字符截断",
    保证长度严格 ≤ max_chars.
    """
    if not isinstance(text, str) or len(text) <= max_chars:
        return text
    return head_tail_by_sentences(text, max_chars, "role-aware residual")


def _compress_structured(
    payload: Dict[str, Any], self_role: str, max_chars: int
) -> Dict[str, Any]:
    """``message_type == "structured"``:

    - reasoning 按句切 → 算相似度 → 动态阈值过滤 → 重组成 ``reasoning``
    - ``full_text`` 按保留句同步替换, 仍超长则按句头尾截断兜底
    - ``answer`` 原样保留
    """
    new_payload = dict(payload)

    # --- 找 reasoning 字段 (平铺 / 嵌套) ---
    reasoning = new_payload.get("reasoning")
    reasoning_source = "top"
    if not isinstance(reasoning, str):
        nested = new_payload.get("output")
        if isinstance(nested, dict) and isinstance(nested.get("reasoning"), str):
            reasoning = nested["reasoning"]
            reasoning_source = "nested"

    if isinstance(reasoning, str) and reasoning.strip():
        sentences = split_sentences(reasoning)
        if not sentences:
            # 无标点, 句子切分失败: 把整段作为 1 个虚拟句参与评分.
            sentences = [reasoning.strip()]
        kept, scores, thr = _filter_sentences_by_relevance(sentences, self_role)
        if not kept:
            # 阈值过严, 全被滤掉: 退化为保留原始 reasoning (避免下游拿到空)
            kept = sentences
        new_reasoning = " ".join(kept)
        # 长度兜底: 即使相似度过滤通过, 总长仍超 max_chars 时也按句截断
        if len(new_reasoning) > max_chars:
            new_reasoning = _truncate_to_budget(new_reasoning, max_chars)
        if reasoning_source == "top":
            new_payload["reasoning"] = new_reasoning
        else:
            new_payload["output"] = dict(new_payload["output"])
            new_payload["output"]["reasoning"] = new_reasoning
    else:
        new_reasoning = None

    # --- 同步处理 full_text ---
    if new_reasoning is not None:
        full_text = None
        if isinstance(new_payload.get("full_text"), str):
            full_text = new_payload["full_text"]
        elif isinstance(new_payload.get("output"), dict) and isinstance(
            new_payload["output"].get("full_text"), str
        ):
            full_text = new_payload["output"]["full_text"]
        if isinstance(full_text, str):
            # 优先: 同步替换 reasoning 段
            if isinstance(reasoning, str) and reasoning:
                new_full_text = full_text.replace(reasoning, new_reasoning or "")
            else:
                new_full_text = full_text
            # 兜底: 还超长就按句头尾截断
            if len(new_full_text) > max_chars:
                new_full_text = _truncate_to_budget(new_full_text, max_chars)
            new_payload["full_text"] = new_full_text
            if reasoning_source == "nested" and isinstance(new_payload.get("output"), dict):
                new_payload["output"]["full_text"] = new_full_text

    return new_payload


class UnstructuredPayloadError(RuntimeError):
    """``message_type != "structured"`` 时本模块拒绝压缩, 主动抛错.

    保留该异常类以支持 :class:`MessageCompressionConfig.strict_unstructured`.
    """


def compress_messages_role_aware(
    info: Dict[str, Dict[str, Any]],
    cfg: MessageCompressionConfig,
    self_role: str,
) -> Dict[str, Dict[str, Any]]:
    """模块 1 入口: 供 Node.get_*_info 调用.

    - cfg.role_aware_compression=False 或 info 为空, 原样返回.
    - payload["message_type"] == "structured" 时, 按句子切分, 算与接收方
      ``self_role`` 描述的余弦相似度, 动态阈值过滤.
    - 其他情况: 运行时默认 noop (与关掉该模块等价), 仅打 ``warnings.warn``
      提示; 当 ``cfg.strict_unstructured=True`` 时抛
      :class:`UnstructuredPayloadError`.
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
            warnings.warn(
                f"role_aware_compression 跳过非 structured payload "
                f"(key={key!r}, message_type={msg_type!r}, output type={out_val_repr}); "
                "此时模块行为等价于 cfg.role_aware_compression=False.",
                RuntimeWarning,
                stacklevel=2,
            )
            out[key] = payload
            continue
        out[key] = _compress_structured(payload, self_role, max_chars)
    return out