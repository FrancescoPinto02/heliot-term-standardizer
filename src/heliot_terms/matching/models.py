"""Models returned by terminology matchers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from heliot_terms.domain.enums import AliasCategory, TargetType


class MatchCandidate(BaseModel):
    """Candidate match found by a deterministic or fallback matcher."""

    model_config = ConfigDict(extra="forbid")

    surface: str = Field(..., description="Matched text as it appears in the searched text.")
    normalized_surface: str = Field(..., description="Normalized matched alias.")

    start: int = Field(..., ge=0, description="Start offset in the normalized text.")
    end: int = Field(..., ge=0, description="End offset in the normalized text, exclusive.")

    target_id: str
    target_type: TargetType

    alias_raw: str
    alias_category: AliasCategory

    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    priority: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Alias priority used by overlap resolution.",
    )
    method: str = "exact"

    safe_for_exact_match: bool = True
    requires_context: bool = False

    metadata: dict[str, Any] = Field(default_factory=dict)