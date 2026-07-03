"""SSP 算法的数据结构：Stage、StageDAG、ComplexityLevel。"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


class ComplexityLevel(Enum):
    """query 复杂度档位，决定阶段数 K 的范围。

    两档设计：简单与困难都使用 K ∈ [3, 5]。Planner 在 prompt 里会
    给出该范围，由 LLM 自行决定落在哪个具体值。
    """

    SIMPLE = ("simple", 3, 5)
    HARD = ("hard", 3, 5)

    def __init__(self, label: str, k_min: int, k_max: int) -> None:
        self.label = label
        self.k_min = k_min
        self.k_max = k_max

    @classmethod
    def from_label(cls, label: str) -> "ComplexityLevel":
        for level in cls:
            if level.label == label:
                return level
        raise ValueError(f"未知复杂度档位：{label!r}")


@dataclass
class Stage:
    """任务的一个阶段。

    Attributes:
        id: 阶段编号（字符串，例如 "S1"）。
        name: 阶段名（短，便于显示）。
        description: 阶段描述（1~2 句，用于与 agent role 算相似度）。
    """

    id: str
    name: str
    description: str


@dataclass
class StageDAG:
    """阶段的有向无环图。

    Attributes:
        stages: 阶段列表（保持 Planner 输出的原始顺序）。
        edges: 阶段依赖边列表，每个元素 ``(from_id, to_id)``。
    """

    stages: List[Stage] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    def descriptions(self) -> List[str]:
        """所有阶段描述（顺序与 ``stages`` 一致）。"""
        return [s.description for s in self.stages]

    def stage_ids(self) -> List[str]:
        return [s.id for s in self.stages]
