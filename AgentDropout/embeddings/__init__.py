"""文本向量化工具包（Text Embedding Utilities）。

与 :mod:`AgentDropout.llm` 的定位不同：
    * LLM 通过在线 API 生成文本（涉及成本与延迟）。
    * Embedder 通过本地 sentence-vector 模型把文本映射到稠密向量，
      主要用于 **离线** 相似度计算（如 SSP 算法里 agent 角色与
      任务阶段的匹配打分）。

本包提供：
    * :class:`Embedder` —— 抽象接口，定义 ``encode`` 协议。
    * :class:`MiniLMEmbedder` —— 基于 ``sentence-transformers``
      集成 ``all-MiniLM-L6-v2`` 的默认实现。

约定：``encode`` 必须返回 ``np.ndarray``，形状为 ``(N, D)``，
dtype 为 ``float32``，并按行做 L2 归一化（cosine 相似度可由
矩阵点积直接得到）。
"""

from AgentDropout.embeddings.base import Embedder
from AgentDropout.embeddings.minilm_embedder import MiniLMEmbedder

__all__ = ["Embedder", "MiniLMEmbedder"]
