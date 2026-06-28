from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "UMCU/SapBERT-UMLS-2020AB-all-lang-from-XLMR-ST_bf16"
DEFAULT_BATCH_SIZE = 64


class SemanticEncoder:
    """Encode biomedical entity strings into dense vectors."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into normalized float32 vectors."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return vectors.astype(np.float32)