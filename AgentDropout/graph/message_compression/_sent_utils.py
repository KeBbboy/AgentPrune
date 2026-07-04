"""句子切分与归一化工具, 模块 1 / 模块 2 共用.

设计:
  - 切分: 中英文句末标点 ``. ! ? 。 ! ?`` 视为句末, 段内换行也算句末.
  - 归一化: 去首尾空白 + 统一空白 + 强制句末加 ``.`` + 转小写, 用于句子去重.
"""
from __future__ import annotations

import re
from typing import List


# 句末标点 + 后续空白/紧邻字符, 用 ``split`` 保留分隔符
_SENT_END_RE = re.compile(r"(?<=[。.!?！？])\s+|(?<=[。.!?！？])(?=[^\s])")


def split_sentences(text: str) -> List[str]:
    """把 reasoning 切成句子列表. 保留每个句子的原始字符串 (含末尾标点).

    切分规则:
      - 句末标点 ``. ! ? 。 ! ?`` 视为句末
      - 段内换行也算句末
      - 切完后 strip, 丢弃空字符串
    """
    if not isinstance(text, str) or not text.strip():
        return []
    parts: List[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        for piece in _SENT_END_RE.split(line):
            s = piece.strip()
            if s:
                parts.append(s)
    return parts


def norm_sentence(s: str) -> str:
    """句子归一化键: 去首尾空白 + 统一空白 + 句末加 ``.`` + 小写."""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    if s and s[-1] not in "。.!！？?;":
        s += "."
    return s.lower()


def head_tail_by_sentences(text: str, max_chars: int, tag: str) -> str:
    """按句子切分后, 保留头/尾若干句, 中间用占位符替换.

    保留句数规则:
      - 句子总字符 ≤ max_chars: 原样返回
      - 头/尾各保留字符预算 ``max_chars / 2``, 在预算内尽量保留整句,
        直到累计字符超过预算的一半就停.
      - 中间用 ``... [tag @ total_chars] ...`` 替换, 字符总长 ≤ max_chars.
    """
    if not isinstance(text, str) or len(text) <= max_chars:
        return text

    sentences = split_sentences(text)
    if len(sentences) <= 1:
        # 没有切出句子 (例如整段无标点), 退回字符级头/尾截断
        half = max_chars // 2
        return f"{text[:half]}\n... [{tag} @ {len(text)}] ...\n{text[-half:]}"

    half_budget = max_chars // 2
    head: List[str] = []
    head_len = 0
    for s in sentences:
        # +1 给 join 时的分隔符
        if head_len + len(s) + 1 > half_budget:
            break
        head.append(s)
        head_len += len(s) + 1

    tail: List[str] = []
    tail_len = 0
    for s in reversed(sentences):
        if tail_len + len(s) + 1 > half_budget:
            break
        tail.append(s)
        tail_len += len(s) + 1
    tail.reverse()

    placeholder = f"... [{tag} @ {len(text)}, {len(sentences)} sents] ..."
    head_str = " ".join(head)
    tail_str = " ".join(tail)
    result = f"{head_str}\n{placeholder}\n{tail_str}".strip()
    # 兜底: 万一拼出来还超阈值, 再粗裁一下尾巴
    if len(result) > max_chars:
        result = result[:max_chars]
    return result