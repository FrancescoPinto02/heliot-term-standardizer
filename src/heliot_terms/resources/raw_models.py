from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RawModel(BaseModel):
    """Base model for raw input records."""

    model_config = ConfigDict(extra="forbid")


class RawSynonymEntry(RawModel):
    """Synonym entry loaded from synonyms.json."""

    ingredient_it: str
    ingredient_en: str | None = None
    entity_type: str = Field(..., description="Expected values: active, inactive, or active/inactive.")

    synonyms_it: list[str] = Field(default_factory=list)
    synonyms_en: list[str] = Field(default_factory=list)

    @field_validator("ingredient_it", "entity_type")
    @classmethod
    def _required_string(cls, value: str) -> str:
        """Ensure required string fields are not empty."""
        value = str(value or "").strip()
        if not value:
            raise ValueError("Required field cannot be empty.")
        return value

    @field_validator("ingredient_en")
    @classmethod
    def _optional_string(cls, value: str | None) -> str | None:
        """Normalize optional string fields."""
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    @field_validator("synonyms_it", "synonyms_en", mode="before")
    @classmethod
    def _ensure_list(cls, value: object) -> list[str]:
        """Convert null values to empty lists and trim synonym strings."""
        if value is None:
            return []

        if not isinstance(value, list):
            raise ValueError("Synonyms must be provided as a list.")

        cleaned: list[str] = []
        for item in value:
            item_str = str(item or "").strip()
            if item_str:
                cleaned.append(item_str)

        return cleaned


class RawDrugRecord(RawModel):
    """Drug product record loaded from drugs.csv."""

    drug_code: str
    drug_name: str
    atc: str | None = None
    drug_form: str | None = None

    compositions: list[str] = Field(default_factory=list)
    excipients: list[str] = Field(default_factory=list)

    @field_validator("drug_code", "drug_name")
    @classmethod
    def _required_string(cls, value: str) -> str:
        """Ensure required drug fields are not empty."""
        value = str(value or "").strip()
        if not value:
            raise ValueError("Required field cannot be empty.")
        return value

    @field_validator("atc", "drug_form")
    @classmethod
    def _optional_string(cls, value: str | None) -> str | None:
        """Normalize optional string fields."""
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    @field_validator("compositions", "excipients", mode="before")
    @classmethod
    def _ensure_list(cls, value: object) -> list[str]:
        """Ensure composition and excipient values are represented as lists."""
        if value is None:
            return []

        if not isinstance(value, list):
            raise ValueError("Composition and excipients must be provided as lists.")

        cleaned: list[str] = []
        for item in value:
            item_str = str(item or "").strip()
            if item_str:
                cleaned.append(item_str)

        return cleaned