"""阶段划分 Planner。

输入 query → 输出 :class:`StageDAG`。流程：
    1. 用 query 长度估算复杂度档位（simple/medium/complex）；
    2. 由档位得到 K 范围（simple 2-3、medium 3-5、complex 5-7）；
    3. 调 LLM（通过 :class:`LLMRegistry`）让其在 K 范围内给出 DAG；
    4. JSON 解析失败时按 fallback 默认 DAG 返回（3 阶段：分析-执行-验证）。

注意：这里 **不** 直接复用 project 内的 agent 实例（避免污染上层图），
而是用一个独立的轻量 LLM 调用。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from AgentDropout.algorithm.stages import ComplexityLevel, Stage, StageDAG
from AgentDropout.llm.llm_registry import LLMRegistry
from AgentDropout.llm.format import Message


# ---------------------------------------------------------------------------
# 复杂度启发式
# ---------------------------------------------------------------------------

# 简单 query 的关键词长度上界（字符数）；超过这个长度通常更复杂
_SIMPLE_QUERY_LEN_THRESHOLD = 80
# 多任务/多步骤的强信号关键词（任一出现就升级一档）
_MULTI_TASK_HINTS = (
    " and then ", "之后", "然后", "同时", "finally", "最后", "step 1", "步骤",
    "first", "second", "设计并实现", "前端", "后端", "deploy", "部署",
)


def _estimate_complexity(query: str) -> ComplexityLevel:
    """根据 query 长度与关键词启发式估计复杂度档位。

    规则：
        * 默认 simple；
        * 长度 > 阈值 **或** 含多任务关键词 → HARD。
    """
    q = (query or "").strip().lower()
    if not q:
        return ComplexityLevel.SIMPLE

    if len(q) > _SIMPLE_QUERY_LEN_THRESHOLD or any(
        hint in q for hint in _MULTI_TASK_HINTS
    ):
        return ComplexityLevel.HARD
    return ComplexityLevel.SIMPLE


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = """You are a task planner. Decompose the user query into a Directed
Acyclic Graph (DAG) of stages.

Output strictly as JSON with two keys:
{
  "stages": [
    {"id": "S1", "name": "...", "description": "..."},
    ...
  ],
  "edges": [
    {"from": "S1", "to": "S2"},
    ...
  ]
}

Rules:
1. The number of stages MUST be within the recommended range.
2. Stages must form a DAG (no cycles).
3. Use branching when sub-tasks are independent; use merging when several
   prior stages feed into one.
4. Each description must be 1-2 sentences, specific enough to match agent roles.
5. Do NOT include any text outside the JSON object."""


@dataclass
class StagePlanner:
    """阶段划分 Planner。

    Attributes:
        llm_name: 用于调度的 LLM 标识（会传给 :class:`LLMRegistry`）。
        temperature: 生成温度，默认 0.2 保证稳定性。
        max_tokens: 响应 token 上限。
        max_retries: JSON 解析失败时的重试次数。
    """

    llm_name: str = "MiniMax-M3"
    temperature: float = 0.2
    max_tokens: int = 1000
    max_retries: int = 2

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def plan(self, query: str) -> StageDAG:
        """根据 query 划分阶段。

        Args:
            query: 用户问题。

        Returns:
            StageDAG: 解析后的阶段 DAG；解析失败时返回 fallback DAG。
        """
        complexity = _estimate_complexity(query)
        return self.plan_with_complexity(query, complexity)

    def plan_with_complexity(self, query: str, complexity: ComplexityLevel) -> StageDAG:
        """显式指定复杂度档位的 plan 接口（便于测试与 ablation）。"""
        user_prompt = self._build_user_prompt(query, complexity)
        messages = [
            Message(role="system", content=_PLANNER_SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ]

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self._call_llm(messages)
                dag = self._parse_dag(raw, complexity)
                return dag
            except Exception as exc:  # 解析或网络错误
                last_err = exc
        # 全部失败 → fallback
        return _fallback_dag()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _build_user_prompt(self, query: str, complexity: ComplexityLevel) -> str:
        return (
            f"Recommended number of stages: {complexity.k_min} to {complexity.k_max}\n"
            f"Query: {query}\n"
            "Return JSON only."
        )

    def _call_llm(self, messages: List[Message]) -> str:
        """调用 LLMRegistry 取到的 LLM 实例，返回首条响应文本。"""
        try:
            llm = LLMRegistry.get(self.llm_name)
        except Exception as exc:
            raise RuntimeError(f"Planner 无法获取 LLM {self.llm_name!r}: {exc}") from exc

        try:
            response = llm.gen(
                messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as exc:
            raise RuntimeError(f"Planner LLM 调用失败：{exc}") from exc

        # 不同实现可能返回 str 或 List[str]，统一取首条
        if isinstance(response, list):
            response = response[0] if response else ""
        if response is None:
            return ""
        return str(response)

    @staticmethod
    def _parse_dag(raw: str, complexity: ComplexityLevel) -> StageDAG:
        """解析 LLM 输出为 StageDAG。

        容忍以下情况：
            * 响应被 ```json ... ``` 包裹；
            * 响应前后有非 JSON 文本（取首个 {...}）。
        """
        text = (raw or "").strip()
        if not text:
            raise ValueError("Planner 返回为空")

        # 去掉 markdown code fence
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        # 截取首个 JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Planner 输出不含 JSON object：{text[:200]!r}")
        text = match.group(0)

        try:
            obj: Dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Planner 输出 JSON 解析失败：{exc}") from exc

        raw_stages = obj.get("stages") or []
        raw_edges = obj.get("edges") or []
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("Planner 输出缺少 stages")
        if not isinstance(raw_edges, list):
            raise ValueError("Planner 输出 edges 不是 list")

        # 强制 K 在 complexity 范围内（防止 LLM 跑飞）
        if not (complexity.k_min <= len(raw_stages) <= complexity.k_max):
            raise ValueError(
                f"Planner 输出阶段数 {len(raw_stages)} 超出范围 "
                f"[{complexity.k_min}, {complexity.k_max}]"
            )

        stages: List[Stage] = []
        for i, s in enumerate(raw_stages):
            sid = str(s.get("id") or f"S{i + 1}")
            name = str(s.get("name") or sid).strip()
            desc = str(s.get("description") or "").strip()
            if not desc:
                raise ValueError(f"Planner 输出阶段 {sid} 缺少 description")
            stages.append(Stage(id=sid, name=name, description=desc))

        stage_ids = {s.id for s in stages}
        edges: List = []
        for e in raw_edges:
            if not isinstance(e, dict):
                continue
            src = e.get("from")
            dst = e.get("to")
            if src in stage_ids and dst in stage_ids and src != dst:
                edges.append((src, dst))

        return StageDAG(stages=stages, edges=edges)


def _fallback_dag() -> StageDAG:
    """Planner 彻底失败时的兜底 DAG（线性 3 阶段）。"""
    stages = [
        Stage("S1", "Analyze", "分析问题、识别关键要素并拆解为可执行子目标。"),
        Stage("S2", "Execute", "基于上一步的分析逐项执行，产出中间结果。"),
        Stage("S3", "Verify", "复核执行结果，检查一致性并输出最终答案。"),
    ]
    edges = [("S1", "S2"), ("S2", "S3")]
    return StageDAG(stages=stages, edges=edges)
