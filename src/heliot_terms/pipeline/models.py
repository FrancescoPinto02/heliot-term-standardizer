from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from heliot_terms.domain.enums import EntityType, TargetType


OutputLanguage = Literal["it", "en"]
NoteOutputPolicy = Literal["annotate", "replace", "structured_only"]


class StandardizedConcept(BaseModel):
    """Canonical concept used in a standardized mention."""

    model_config = ConfigDict(extra="forbid")

    concept_id: str
    entity_type: EntityType
    canonical_it: str
    canonical_en: str | None = None
    role: str | None = Field(
        default=None,
        description="Role in the specific context, e.g. active_ingredient or excipient.",
    )

    def display_name(self, output_language: OutputLanguage) -> str:
        """Return the canonical name in the requested output language."""
        if output_language == "en" and self.canonical_en:
            return self.canonical_en
        return self.canonical_it


class StandardizedMention(BaseModel):
    """A resolved and enriched mention found in the clinical note."""

    model_config = ConfigDict(extra="forbid")

    surface: str
    normalized_surface: str
    start: int
    end: int
    normalized_start: int
    normalized_end: int

    target_id: str
    target_type: TargetType

    matched_alias: str
    method: str
    confidence: float

    concepts: list[StandardizedConcept] = Field(default_factory=list)

    annotation_text: str | None = Field(
        default=None,
        description="Text appended in annotate mode, without brackets.",
    )
    replacement_text: str | None = Field(
        default=None,
        description="Text used in replace mode.",
    )

    status: str = "resolved"
    metadata: dict[str, Any] = Field(default_factory=dict)


class StandardizationResult(BaseModel):
    """Final output returned by the terminology standardization pipeline."""

    model_config = ConfigDict(extra="forbid")

    original_text: str
    normalized_text: str
    standardized_text: str

    output_language: OutputLanguage
    note_output_policy: NoteOutputPolicy

    matches: list[StandardizedMention] = Field(default_factory=list)
    ambiguous: list[StandardizedMention] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)