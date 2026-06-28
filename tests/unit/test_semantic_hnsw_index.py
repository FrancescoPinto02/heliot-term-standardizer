"""Unit tests for the semantic HNSW vector index."""

import numpy as np

from heliot_terms.domain.enums import TargetType
from heliot_terms.fallback.semantic.hnsw_index import SemanticVectorIndex
from heliot_terms.fallback.semantic.models import SemanticIndexMetadata


def _metadata(vector_id: int, target_id: str, alias: str) -> SemanticIndexMetadata:
    return SemanticIndexMetadata(
        vector_id=vector_id,
        target_id=target_id,
        target_type=TargetType.INGREDIENT,
        alias_raw=alias,
        alias_normalized=alias,
        language="it",
        source="UMLS",
    )


def test_semantic_vector_index_returns_nearest_neighbor() -> None:
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    metadata = [
        _metadata(0, "active:paracetamolo", "paracetamolo"),
        _metadata(1, "active:ibuprofene", "ibuprofene"),
        _metadata(2, "active:amoxicillina", "amoxicillina"),
    ]

    index = SemanticVectorIndex.build(vectors=vectors, metadata=metadata)

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = index.search(query, top_k=2)

    assert results[0].metadata.target_id == "active:paracetamolo"
    assert results[0].similarity > results[1].similarity


def test_semantic_vector_index_save_and_load_roundtrip(tmp_path) -> None:
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    metadata = [
        _metadata(0, "active:paracetamolo", "paracetamolo"),
        _metadata(1, "active:ibuprofene", "ibuprofene"),
    ]

    index = SemanticVectorIndex.build(vectors=vectors, metadata=metadata)
    index.save(tmp_path, model_name="test-model")

    loaded = SemanticVectorIndex.load(tmp_path)

    query = np.array([0.0, 1.0], dtype=np.float32)
    results = loaded.search(query, top_k=1)

    assert results[0].metadata.target_id == "active:ibuprofene"