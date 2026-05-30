"""Unit tests for domain models."""

import pytest
from pydantic import ValidationError

from heliot_terms.domain.enums import (
    AliasLanguage,
    AliasSource,
    BrandStatus,
    EntityType,
    TargetType,
)
from heliot_terms.domain.models import Alias, DrugBrand, DrugProduct, IngredientConcept


def test_ingredient_concept_accepts_valid_active_concept() -> None:
    concept = IngredientConcept(
        concept_id="active:paracetamolo",
        entity_type=EntityType.ACTIVE,
        canonical_it="paracetamolo",
        canonical_en="Paracetamol",
        sources=["AIFA", "UMLS"],
    )

    assert concept.concept_id == "active:paracetamolo"
    assert concept.canonical_en == "Paracetamol"


def test_ingredient_concept_rejects_wrong_prefix() -> None:
    with pytest.raises(ValidationError):
        IngredientConcept(
            concept_id="inactive:paracetamolo",
            entity_type=EntityType.ACTIVE,
            canonical_it="paracetamolo",
            canonical_en="Paracetamol",
        )


def test_alias_accepts_short_contextual_alias() -> None:
    alias = Alias(
        alias_raw="PEG",
        alias_normalized="peg",
        target_id="inactive:macrogol",
        target_type=TargetType.INGREDIENT,
        language=AliasLanguage.EN,
        source=AliasSource.UMLS,
        safe_for_exact_match=False,
        requires_context=True,
    )

    assert alias.requires_context is True


def test_drug_product_accepts_valid_product() -> None:
    product = DrugProduct(
        product_id="drug:aic:012345678",
        drug_code="012345678",
        drug_name="TACHIPIRINA 500MG COMPRESSE",
        brand_name="TACHIPIRINA",
        normalized_brand_name="tachipirina",
        active_ingredient_ids=["active:paracetamolo"],
    )

    assert product.product_id == "drug:aic:012345678"


def test_drug_brand_accepts_valid_brand() -> None:
    brand = DrugBrand(
        brand_id="brand:tachipirina",
        brand_name="Tachipirina",
        normalized_brand_name="tachipirina",
        product_ids=["drug:aic:012345678"],
        active_ingredient_ids=["active:paracetamolo"],
        active_ingredient_signatures=[["active:paracetamolo"]],
        brand_status=BrandStatus.SINGLE_ACTIVE_SIGNATURE,
    )

    assert brand.brand_status == BrandStatus.SINGLE_ACTIVE_SIGNATURE