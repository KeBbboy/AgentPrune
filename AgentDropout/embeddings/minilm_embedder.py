"""基于 sentence-transformers 的本地 Embedder 实现。

默认加载 ``sentence-transformers/all-MiniLM-L6-v2``（约 80MB），
该模型输出 384 维向量且已对 cosine 相似度优化。本项目已经在
``requirements.txt`` 中固定了 ``sentence-transformers>=2.2.2``，
无需新增依赖。

可扩展性：
    * 支持任意长度的输入文本。模型最大序列长度为 512 token，超长
      文本会被自动截断（这是 MiniLM 本身的限制）。
    * 支持 ``device`` 选择，缺省自动判断 GPU/CPU。
    * 支持 ``cache_dir``，便于离线环境预先下载模型。
"""

from typing import List, Union

import numpy as np

from AgentDropout.embeddings.base import Embedder


class MiniLMEmbedder(Embedder):
    """``all-MiniLM-L6-v2`` 包装。

    Examples:
        >>> emb = MiniLMEmbedder()
        >>> v = emb.encode(["hello", "你好"])
        >>> v.shape
        (2, 384)
        >>> float(np.linalg.norm(v[0]))
        1.0
    """

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_DIM = 384

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        cache_dir: str | None = None,
        normalize: bool = True,
    ) -> None:
        """初始化 MiniLMEmbedder。

        Args:
            model_name: sentence-transformers 模型标识符。
            device: ``"cpu"`` / ``"cuda"`` / ``None``（自动选择）。
            cache_dir: 模型缓存目录，传 ``None`` 使用默认。
            normalize: 是否对输出做 L2 归一化（默认 True，保持接口契约）。
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "MiniLMEmbedder 需要 sentence-transformers，请先 "
                "`pip install sentence-transformers>=2.2.2`"
            ) from exc

        self.model_name = model_name
        self.normalize = normalize

        # 延迟到第一次 encode 时再加载模型？这里选择立即加载：
        # 1. 让错误尽早暴露；
        # 2. 避免运行时偶发的网络/CPU spike 影响推理。
        load_kwargs = {}
        if device is not None:
            load_kwargs["device"] = device
        if cache_dir is not None:
            load_kwargs["cache_folder"] = cache_dir
        try:
            self._model = SentenceTransformer(model_name, **load_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"无法加载 sentence-transformers 模型 {model_name!r}：{exc}"
            ) from exc

        # 部分 sentence-transformers 版本不暴露 get_sentence_embedding_dimension
        try:
            self._dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:
            self._dim = self.DEFAULT_DIM

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """把文本编码为向量。

        Args:
            texts: 单条字符串或字符串列表。

        Returns:
            np.ndarray: ``(N, D)`` float32，已 L2 归一化。
        """
        # 输入归一化：单字符串 → [str]
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise TypeError(
                f"encode 仅接受 str 或 List[str]，实际类型：{type(texts).__name__}"
            )
        if len(texts) == 0:
            # 与 sentence-transformers 行为对齐：空输入直接返回 (0, D)
            return np.zeros((0, self._dim), dtype=np.float32)

        try:
            # convert_to_numpy=True 返回 ndarray；normalize_embeddings 与接口契约一致
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise RuntimeError(f"sentence-transformers 编码失败：{exc}") from exc

        # 强制 dtype=float32，避免下游矩阵乘法时类型不一致
        return np.asarray(embeddings, dtype=np.float32)
