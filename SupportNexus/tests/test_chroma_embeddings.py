import math

from core.chroma_embeddings import StableHashEmbeddingFunction


def test_stable_hash_embeddings_are_deterministic_and_normalized():
    embedding = StableHashEmbeddingFunction()

    first, repeated, different = embedding(["API 鉴权失败 401", "API 鉴权失败 401", "订阅账单重复扣款"])

    assert len(first) == embedding.dimensions == 384
    assert first == repeated
    assert first != different
    assert math.isclose(math.sqrt(sum(value * value for value in first)), 1.0)
