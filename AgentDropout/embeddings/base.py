"""Embedder 抽象接口。

任何文本向量化实现都必须遵循 :meth:`Embedder.encode` 的输入与输出
约定，保证 SSP 等下游算法可以无差别地使用不同实现。
"""

from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class Embedder(ABC):
    """文本向量化抽象基类。

    接口契约：
        * 输入支持单条 :class:`str` 或 :class:`List[str]`；
        * 返回 :class:`numpy.ndarray`，形状 ``(N, D)``，dtype ``float32``；
        * 每行必须 L2 归一化，使得 ``np.dot(a, b) == cos(a, b)``。
    """

    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """把文本编码为向量。

        Args:
            texts: 单条字符串或字符串列表。

        Returns:
            np.ndarray: 形状 ``(N, D)`` 的 float32 矩阵。
        """

    @property
    @abstractmethod
    def dim(self) -> int:
        """返回向量的维度 ``D``。"""
