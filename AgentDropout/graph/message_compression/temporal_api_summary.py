"""模块 3: 按 answer 一致性聚合多智能体输出 (--temporal_api_summary 开关).

设计:
  1. 从每个 node 的 last_memory['outputs'][-1] 抽 (reasoning, answer)
  2. 答案归一化 (去前缀噪音 + 去标点 + 小写) 作为分组键
  3. 单组 (答案一致): 合并所有 reasoning, 选最长 reasoning 的首句作"理由",
     输出 ``共识 (<roles>): <ans>\\n理由: <first sentence>``
  4. 多组 (答案分裂): 按 group size 降序遍历, 每组取 sample reasoning 的
     首句作"该答案的理由", 输出

        不一致: 多智能体给出了不同答案
        答案 A (来自 role1, role2, 共 N 个): <ans> -- 理由: <first sentence>
        答案 B (来自 role3, 共 M 个):       <ans> -- 理由: <first sentence>

依赖: 仅 ``re / collections / typing``, 不调 LLM.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from AgentDropout.graph.message_compression.config import (
    MessageCompressionConfig,
)


# 复用 llm.gpt_chat 的 split_thinking 风格 (本模块不依赖 gpt_chat 避免循环)
# 必须有 `` 闭标签, 否则会把整段文本都吞进 reasoning
THINK_RE = re.compile(r"<(?:redacted_)?think>(.*?)</think>", re.DOTALL)

# 答案归一化: 用于"答案一致"的判定
#   1. 先去掉前缀噪音 (按从长到短排序, 防止 'the final answer is' 被 'the answer is' 抢先匹配)
_ANSWER_NOISE_PATTERNS = [
    r"the final answer is",
    r"the answer is",
    r"final answer is",
    r"final answer\s*[:：]",
    r"final answer\s+(?=\S)",
    r"answer\s*[:：]",
    r"final answer\b",
    r"answer\b",
    r"答案是",
    r"答案\s*[:：]",
    r"最终答案\s*[:：]?\s*是",
    r"最终答案是",
    r"最终答案\s*[:：]",
]
_ANSWER_NOISE_RE = re.compile(
    r"^(?:" + "|".join(_ANSWER_NOISE_PATTERNS) + r")\s*[:：]?\s*",
    re.IGNORECASE,
)
#   2. 删除标点/空白, 再小写
_ANSWER_NORM_RE = re.compile(r"[\s\.,;:!\?'\"`()\[\]{}_/\\\-]+")

# "理由" 抽取相关常量
_FIRST_SENTENCE_RE = re.compile(r"[。.!?！？]+\s*")  # 中英文句末 (允许标点后无空白)
_MAX_REASON_LEN = 200


def _normalize_answer(ans: str) -> str:
    """把多份答案归一成比较键.

    例如 ``"The answer is 42."`` / ``"final answer 42"`` / ``"42"`` 都归一为 ``"42"``.
    但没有"是"的句子 ``"final answer"`` 单独成串归一为 ``""``.
    """
    if not isinstance(ans, str):
        return ""
    s = ans.strip()
    s = _ANSWER_NOISE_RE.sub("", s)
    s = _ANSWER_NORM_RE.sub("", s)
    return s.lower()


def _first_sentence(text: str, max_len: int = _MAX_REASON_LEN) -> str:
    """抽取 ``text`` 的首句.

    切分优先级:
      1. ``"。"`` / ``"."`` / ``"!"`` / ``"?"`` 等句末标点
      2. 第一个 ``"\\n"`` 换行
      3. 上面都没有, 返回前 ``max_len`` 字符
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    m = _FIRST_SENTENCE_RE.search(text)
    if m:
        head = text[: m.start() + 1].strip()
    else:
        head = (text.splitlines()[0] if text.splitlines() else "").strip()
    if not head:
        return ""
    return head[:max_len]


def _format_summary_consensus(answer_text: str, reason: str, roles: List[str]) -> str:
    return (
        f"共识 ({', '.join(roles)}): {answer_text}\n"
        f"理由: {reason}"
    )


def _consensus_reason_via_text(per_node: List[Tuple[Any, str, str]]) -> str:
    """单答案路径: 用纯字符从所有 reasoning 中抽取第一句作共识理由.

    取**最长** reasoning 中的首句, 因为它最可能包含完整结论.
    """
    candidates = [r.strip() for _, r, _ in per_node if isinstance(r, str) and r.strip()]
    if not candidates:
        return "共识: 各智能体未给出 reasoning 文本."
    richest = max(candidates, key=len)
    head = _first_sentence(richest)
    return head if head else "共识(无明确首句, 推理已合并)."


def _per_answer_reason_via_text(_answer: str, reasoning: str) -> str:
    """不一致路径: 用纯字符从 reasoning 抽取首句作"该答案的理由"."""
    head = _first_sentence(reasoning)
    return head if head else "(无 reasoning 文本可用)"


def _group_by_answer(per_node: List[Tuple[Any, str, str]]) -> Dict[str, List[Any]]:
    """把 (node, reasoning, answer) 按 answer 归一键分组."""
    groups: Dict[str, List[Any]] = defaultdict(list)
    for node, _, ans in per_node:
        groups[_normalize_answer(ans)].append(node)
    return groups


def split_reasoning(text: str) -> tuple[str, str]:
    """启发式拆 ``<redacted_think>...`` 段; 返回 (reasoning, answer).

    没有 think 标签时 reasoning 为空字符串, answer = 原文.
    """
    if not isinstance(text, str) or not text:
        return "", text or ""
    m = THINK_RE.search(text)
    if not m:
        return "", text
    reasoning = m.group(1).strip()
    answer = (text[: m.start()] + text[m.end():]).strip()
    return reasoning, answer


def _extract_reasoning_and_answer(node) -> Tuple[str, str]:
    """从 ``node.last_memory['outputs'][-1]`` (full_text 字符串) 抽出
    (reasoning, answer). 字典形式也支持.
    """
    mem = getattr(node, "last_memory", None) or {}
    outputs = mem.get("outputs") or []
    if not outputs:
        return "", ""
    last = outputs[-1]
    if isinstance(last, str):
        return split_reasoning(last)
    if isinstance(last, dict) and last.get("message_type") == "structured":
        r = last.get("reasoning")
        a = last.get("answer")
        return (
            r if isinstance(r, str) else "",
            a if isinstance(a, str) else "",
        )
    return "", ""


async def _build_consensus_or_disagreement_summary(
    per_node: List[Tuple[Any, str, str]],
) -> Tuple[str, bool, List[str]]:
    """按 answer 分组构造 summary.

    返回: (summary_text, is_consensus, answers_original)
    """
    grouped = _group_by_answer(per_node)
    answers_original: List[str] = []
    for _n, _, a in per_node:
        if isinstance(a, str) and a.strip():
            answers_original.append(a.strip())

    all_roles = sorted({getattr(n, "role", "Node") or "Node" for n, _, _ in per_node})

    # 仅 0 或 1 个 group: 单答案路径
    if len(grouped) <= 1:
        answer_text = "(无答案文本)"
        for _n, _, a in per_node:
            if isinstance(a, str) and a.strip():
                answer_text = a.strip()
                break

        reason = _consensus_reason_via_text(per_node)
        return _format_summary_consensus(answer_text, reason, all_roles), True, answers_original

    # 多个组: 不一致路径
    lines: List[str] = ["不一致: 多智能体给出了不同答案", ""]
    raw_by_node = {n: (r, a) for n, r, a in per_node}
    for _norm_ans, nodes in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        sample_ans = ""
        sample_reasoning = ""
        for n in nodes:
            r, a = raw_by_node.get(n, ("", ""))
            if isinstance(a, str) and a.strip():
                sample_ans = a.strip()
                sample_reasoning = r or ""
                break
        roles = sorted({getattr(n, "role", "Node") or "Node" for n in nodes})
        reason = _per_answer_reason_via_text(sample_ans, sample_reasoning)
        lines.append(
            f"答案 {sample_ans} (来自 {', '.join(roles)}, 共 {len(nodes)} 个智能体): "
            f"-- 理由: {reason}"
        )
    return "\n".join(lines), False, answers_original


async def apply_temporal_api_summary(graph, cfg: MessageCompressionConfig, round_idx: int) -> None:
    """Graph.arun 每轮末尾调用, 仅当 cfg.temporal_api_summary=True 时生效.

    回写到每个 node.last_memory:
      - temporal_summary         压缩文本
      - temporal_summary_round   round_idx
      - temporal_consensus       bool
      - temporal_answers         所有 node 的原始 answer 列表 (字符串)
    """
    if not cfg.temporal_api_summary:
        return

    per_node: List[Tuple[Any, str, str]] = []
    for node in graph.nodes.values():
        r, a = _extract_reasoning_and_answer(node)
        per_node.append((node, r, a))

    if not per_node:
        return

    if not any(a.strip() for _, _, a in per_node):
        return

    summary_text, is_consensus, answers_original = (
        await _build_consensus_or_disagreement_summary(per_node)
    )

    answers_snapshot = list(answers_original)
    for node, _, _ in per_node:
        node.last_memory["temporal_summary"] = summary_text
        node.last_memory["temporal_summary_round"] = round_idx
        node.last_memory["temporal_consensus"] = is_consensus
        node.last_memory["temporal_answers"] = answers_snapshot
