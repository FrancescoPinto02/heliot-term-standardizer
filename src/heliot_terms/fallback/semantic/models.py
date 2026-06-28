from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from heliot_terms.domain.enums import TargetType


@dataclass(frozen=True)
class SemanticTextCandidate:
    """Text span extracted by the NER candidate extractor."""

    text: str
    start: int
    end: int
    label: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SemanticIndexMetadata(BaseModel):
    """Metadata associated with one vector in the semantic index."""

    model_config = ConfigDict(extra="forbid")

    vector_id: int
    target_id: str
    target_type: TargetType

    alias_raw: str
    alias_normalized: str

    language: str | None = None
    source: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class SemanticSearchResult:
    """Nearest-neighbor result returned by the semantic vector index."""

    metadata: SemanticIndexMetadata
    similarity: float
    distance: float


@dataclass(frozen=True)
class SemanticAcceptanceDecision:
    """Acceptance decision for semantic nearest-neighbor results."""

    accepted_result: SemanticSearchResult | None
    reason: str
    top_results: list[SemanticSearchResult] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        """Return True if a semantic result was accepted."""
        return self.accepted_result is not None