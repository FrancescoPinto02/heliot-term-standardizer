from __future__ import annotations

import json
from pathlib import Path

import hnswlib
import numpy as np

from heliot_terms.fallback.semantic.models import (
    SemanticIndexMetadata,
    SemanticSearchResult,
)


INDEX_FILENAME = "index.hnsw"
METADATA_FILENAME = "metadata.jsonl"
CONFIG_FILENAME = "index_config.json"

SPACE = "cosine"
M = 32
EF_CONSTRUCTION = 200
EF_SEARCH = 64
TOP_K = 10


class SemanticVectorIndex:
    """Wrapper around an HNSW approximate nearest-neighbor index."""

    def __init__(
        self,
        index: hnswlib.Index,
        metadata_by_vector_id: dict[int, SemanticIndexMetadata],
        dim: int,
    ) -> None:
        self.index = index
        self.metadata_by_vector_id = metadata_by_vector_id
        self.dim = dim
        self.index.set_ef(EF_SEARCH)

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        metadata: list[SemanticIndexMetadata],
    ) -> SemanticVectorIndex:
        """Build an HNSW index from vectors and metadata."""
        if vectors.ndim != 2:
            raise ValueError("Expected a 2D vectors array.")

        if len(vectors) != len(metadata):
            raise ValueError("Vectors and metadata must have the same length.")

        if len(metadata) == 0:
            raise ValueError("Cannot build a semantic index with zero vectors.")

        dim = vectors.shape[1]

        index = hnswlib.Index(space=SPACE, dim=dim)
        index.init_index(
            max_elements=len(metadata),
            ef_construction=EF_CONSTRUCTION,
            M=M,
        )

        labels = np.array([item.vector_id for item in metadata], dtype=np.int64)
        index.add_items(vectors, labels)
        index.set_ef(EF_SEARCH)

        return cls(
            index=index,
            metadata_by_vector_id={item.vector_id: item for item in metadata},
            dim=dim,
        )

    @classmethod
    def load(cls, index_dir: str | Path) -> SemanticVectorIndex:
        """Load an HNSW index and metadata from disk."""
        index_dir = Path(index_dir)

        config_path = index_dir / CONFIG_FILENAME
        metadata_path = index_dir / METADATA_FILENAME
        index_path = index_dir / INDEX_FILENAME

        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        dim = int(config["dim"])

        index = hnswlib.Index(space=config.get("space", SPACE), dim=dim)
        index.load_index(str(index_path))
        index.set_ef(EF_SEARCH)

        metadata_by_vector_id: dict[int, SemanticIndexMetadata] = {}

        with metadata_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    payload = json.loads(line)
                    metadata = SemanticIndexMetadata.model_validate(payload)
                except Exception as exc:
                    raise ValueError(
                        f"Invalid metadata record at {metadata_path}:{line_number}"
                    ) from exc

                metadata_by_vector_id[metadata.vector_id] = metadata

        return cls(
            index=index,
            metadata_by_vector_id=metadata_by_vector_id,
            dim=dim,
        )

    def save(
        self,
        index_dir: str | Path,
        model_name: str,
    ) -> None:
        """Save index, metadata, and configuration to disk."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        self.index.save_index(str(index_dir / INDEX_FILENAME))

        with (index_dir / METADATA_FILENAME).open("w", encoding="utf-8") as file:
            for vector_id in sorted(self.metadata_by_vector_id):
                metadata = self.metadata_by_vector_id[vector_id]
                file.write(metadata.model_dump_json(exclude_none=True))
                file.write("\n")

        config = {
            "space": SPACE,
            "dim": self.dim,
            "model_name": model_name,
            "num_vectors": len(self.metadata_by_vector_id),
        }

        with (index_dir / CONFIG_FILENAME).open("w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)

    def search(
        self,
        vector: np.ndarray,
        top_k: int = TOP_K,
    ) -> list[SemanticSearchResult]:
        """Search nearest neighbors for one vector."""
        if vector.ndim == 1:
            query = vector.reshape(1, -1)
        elif vector.ndim == 2 and vector.shape[0] == 1:
            query = vector
        else:
            raise ValueError("Expected one query vector.")

        k = min(top_k, len(self.metadata_by_vector_id))

        labels, distances = self.index.knn_query(query, k=k)

        results: list[SemanticSearchResult] = []

        for label, distance in zip(labels[0], distances[0], strict=True):
            vector_id = int(label)
            metadata = self.metadata_by_vector_id.get(vector_id)

            if metadata is None:
                continue

            similarity = 1.0 - float(distance)

            results.append(
                SemanticSearchResult(
                    metadata=metadata,
                    similarity=similarity,
                    distance=float(distance),
                )
            )

        return results