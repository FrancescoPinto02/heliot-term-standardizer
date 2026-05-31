from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from heliot_terms.domain.enums import AliasCategory, TargetType


class ResolvedMatch(BaseModel):
    """Final non-overlapping match selected by the resolver."""

    model_config = ConfigDict(extra="forbid")

    surface: str
    normalized_surface: str

    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)

    target_id: str
    target_type: TargetType

    alias_raw: str
    alias_category: AliasCategory

    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    priority: int = Field(default=50, ge=0, le=100)
    method: str

    safe_for_exact_match: bool = True
    requires_context: bool = False

    metadata: dict[str, Any] = Field(default_factory=dict)