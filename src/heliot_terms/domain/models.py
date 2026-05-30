"""Domain models for the HELIOT terminology standardizer.

These models define the internal contract of the project. They are used both
during knowledge-base construction and later by the matching pipeline.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from heliot_terms.domain.enums import (
    AliasLanguage,
    AliasSource,
    BrandStatus,
    EntityType,
    IssueSeverity,
    TargetType,
    AliasCategory,
)

from heliot_terms.domain.ids import entity_type_prefix


class DomainModel(BaseModel):
    """Base class for all domain models.

    Extra fields are forbidden to avoid silently accepting malformed records
    while building or loading the knowledge base.
    """

    model_config = ConfigDict(extra="forbid")


class IngredientConcept(DomainModel):
    """Canonical representation of an active ingredient or excipient."""

    concept_id: str = Field(
        ...,
        description="Stable identifier, e.g. 'active:paracetamolo'.",
    )
    entity_type: EntityType
    canonical_it: str
    canonical_en: str | None = None
    sources: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("concept_id", "canonical_it")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        """Reject empty identifiers and canonical names."""
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be empty.")
        return value

    @model_validator(mode="after")
    def _validate_concept_prefix(self) -> IngredientConcept:
        """Ensure that the concept ID prefix matches the entity type."""
        expected_prefix = f"{entity_type_prefix(self.entity_type)}:"
        if not self.concept_id.startswith(expected_prefix):
            raise ValueError(
                f"Concept ID '{self.concept_id}' must start with '{expected_prefix}'."
            )
        return self


class Alias(DomainModel):
    """Textual form that can refer to a concept, drug product, or drug brand."""

    alias_raw: str = Field(..., description="Alias as found in the source.")
    alias_normalized: str = Field(..., description="Alias after text normalization.")
    target_id: str = Field(..., description="ID of the target entity.")
    target_type: TargetType

    language: AliasLanguage = AliasLanguage.UNKNOWN
    source: AliasSource = AliasSource.DERIVED
    alias_category: AliasCategory = AliasCategory.CLINICAL

    priority: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Higher priority aliases are preferred during conflict resolution.",
    )
    safe_for_exact_match: bool = True
    requires_context: bool = False

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("alias_raw", "alias_normalized", "target_id")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        """Reject empty alias fields."""
        value = value.strip()
        if not value:
            raise ValueError("Alias fields cannot be empty.")
        return value


class DrugProduct(DomainModel):
    """Specific pharmaceutical product/formulation from the drug database."""

    product_id: str = Field(
        ...,
        description="Stable product identifier, e.g. 'drug:aic:012345678'.",
    )
    drug_code: str
    drug_name: str

    brand_name: str | None = None
    normalized_brand_name: str | None = None

    drug_form: str | None = None
    atc: str | None = None

    active_ingredient_ids: list[str] = Field(default_factory=list)
    excipient_ids: list[str] = Field(default_factory=list)

    source: str = "AIFA"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("product_id", "drug_code", "drug_name")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        """Reject empty product identifiers and names."""
        value = value.strip()
        if not value:
            raise ValueError("Product fields cannot be empty.")
        return value

    @model_validator(mode="after")
    def _validate_product_id(self) -> DrugProduct:
        """Ensure product IDs use the expected namespace."""
        if not self.product_id.startswith("drug:aic:"):
            raise ValueError("Product ID must start with 'drug:aic:'.")
        return self


class DrugBrand(DomainModel):
    """Commercial drug name aggregated across multiple products.

    A brand is intentionally modeled separately from ingredients because a brand
    mention is not clinically equivalent to a direct active ingredient mention.
    """

    brand_id: str = Field(..., description="Stable brand identifier, e.g. 'brand:tachipirina'.")
    brand_name: str
    normalized_brand_name: str

    product_ids: list[str] = Field(default_factory=list)

    active_ingredient_ids: list[str] = Field(
        default_factory=list,
        description="Flattened set of active ingredients observed across products.",
    )
    active_ingredient_signatures: list[list[str]] = Field(
        default_factory=list,
        description="Distinct active ingredient sets observed for this brand.",
    )

    brand_status: BrandStatus
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("brand_id", "brand_name", "normalized_brand_name")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        """Reject empty brand identifiers and names."""
        value = value.strip()
        if not value:
            raise ValueError("Brand fields cannot be empty.")
        return value

    @model_validator(mode="after")
    def _validate_brand_id(self) -> DrugBrand:
        """Ensure brand IDs use the expected namespace."""
        if not self.brand_id.startswith("brand:"):
            raise ValueError("Brand ID must start with 'brand:'.")
        return self


class BuildIssue(DomainModel):
    """Issue detected while building the knowledge base."""

    severity: IssueSeverity
    code: str
    message: str
    item_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseBuildReport(DomainModel):
    """Summary of a knowledge-base build execution."""

    num_concepts: int = 0
    num_aliases: int = 0
    num_drug_products: int = 0
    num_drug_brands: int = 0

    num_ambiguous_aliases: int = 0
    num_unsafe_short_aliases: int = 0
    num_unresolved_product_ingredients: int = 0
    num_ambiguous_brands: int = 0

    issues: list[BuildIssue] = Field(default_factory=list)