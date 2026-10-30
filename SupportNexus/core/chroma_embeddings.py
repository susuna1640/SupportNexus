"""Deterministic ChromaDB embeddings that never download model weights.

Chroma's default embedding function lazily downloads all-MiniLM-L6-v2 the
first time a collection receives documents or a text query.  The application
must remain usable while offline or behind a restricted network, so its
collections use this lightweight, process-local alternative explicitly.
"""

from __future__ import annotations

import hashlib
import math
import re

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


class StableHashEmbeddingFunction(EmbeddingFunction[Documents]):
    """A normalized character n-gram embedding with no network or model IO.

    The 384 dimensions match the common MiniLM collection shape, allowing an
    existing Chroma collection to remain queryable after the default client
    embedding function is removed.  BM25 remains part of the application's
    hybrid retrieval path for precise terminology and identifiers.
    """

    dimensions = 384

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(text) for text in input]

    @classmethod
    def _embed(cls, text: str) -> list[float]:
        normalized = re.sub(r"\s+", " ", str(text or "").lower()).strip()
        if not normalized:
            normalized = "<empty>"

        vector = [0.0] * cls.dimensions
        features: list[str] = []
        for width in (1, 2, 3):
            features.extend(normalized[index:index + width] for index in range(len(normalized) - width + 1))

        for feature in features or [normalized]:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % cls.dimensions
            vector[index] += 1.0 if digest[4] & 1 else -1.0

        magnitude = math.sqrt(sum(value * value for value in vector))
        return [value / magnitude for value in vector] if magnitude else vector
