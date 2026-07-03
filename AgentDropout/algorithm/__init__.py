"""SSP（Stage Soft Pruning）算法包。

在多智能体通信拓扑中按"任务阶段"对边进行结构感知剪枝：
    1. Planner 把 query 拆成若干阶段（带 DAG）；
    2. Embedder 计算 agent profile 与阶段描述的相似度；
    3. SSP 按"组内全留 + 跨阶段软阈值"规则重布线。
"""

from AgentDropout.algorithm.stages import Stage, StageDAG, ComplexityLevel
from AgentDropout.algorithm.planner import StagePlanner
from AgentDropout.algorithm.ssp import SSPConfig, apply_ssp, ssp_pseudo_edges

__all__ = [
    "Stage",
    "StageDAG",
    "ComplexityLevel",
    "StagePlanner",
    "SSPConfig",
    "apply_ssp",
    "ssp_pseudo_edges",
]
