"""MiniLMEmbedder 单元测试。

覆盖：
    * dim 属性正确；
    * 单字符串输入；
    * 列表输入；
    * 空列表返回 (0, D)；
    * L2 归一化；
    * 输入类型异常 → TypeError。
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pytest

from AgentDropout.embeddings import MiniLMEmbedder


@pytest.fixture(scope="module")
def embedder():
    """所有测试共用一个 embedder 实例，避免重复加载模型。"""
    return MiniLMEmbedder()


def test_dim(embedder):
    """向量维度应为 384。"""
    assert embedder.dim == 384


def test_encode_single_string(embedder):
    """单字符串输入应返回 (1, D)。"""
    v = embedder.encode("hello")
    assert v.shape == (1, 384)
    assert v.dtype == np.float32


def test_encode_list(embedder):
    """字符串列表输入应返回 (N, D)。"""
    texts = ["hello world", "你好世界", "good morning"]
    v = embedder.encode(texts)
    assert v.shape == (3, 384)
    assert v.dtype == np.float32


def test_encode_empty_list(embedder):
    """空列表应返回 (0, D)，不抛异常。"""
    v = embedder.encode([])
    assert v.shape == (0, 384)
    assert v.dtype == np.float32


def test_l2_normalization(embedder):
    """每行的 L2 范数应接近 1。"""
    v = embedder.encode(["alpha", "beta", "gamma"])
    norms = np.linalg.norm(v, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), f"L2 归一化失败，norms={norms}"


def test_invalid_input_type(embedder):
    """非 str/List[str] 应抛 TypeError。"""
    with pytest.raises(TypeError):
        embedder.encode(123)
    with pytest.raises(TypeError):
        embedder.encode(None)


def test_semantic_similarity(embedder):
    """语义相近的文本 cosine 相似度应高于不相关文本。"""
    v = embedder.encode(["cat", "dog", "quantum physics"])
    sim_cat_dog = float(np.dot(v[0], v[1]))
    sim_cat_physics = float(np.dot(v[0], v[2]))
    assert sim_cat_dog > sim_cat_physics, (
        f"期望 cat-dog 相似度 > cat-physics，实际 {sim_cat_dog} vs {sim_cat_physics}"
    )
