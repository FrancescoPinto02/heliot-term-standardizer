"""Unit tests for the standardization pipeline."""

from heliot_terms.domain.enums import (
    AliasCategory,
    AliasLanguage,
    AliasSource,
    BrandStatus,
    EntityType,
    TargetType,
)
from heliot_terms.domain.models import Alias, DrugBrand, DrugProduct, IngredientConcept
from heliot_terms.matching.aho_corasick_matcher import AhoCorasickMatcher
from heliot_terms.pipeline.standardization_pipeline import (
    StandardizationPipeline,
    StandardizationPipelineConfig,
)
from heliot_terms.resources.knowledge_base_repository import KnowledgeBaseRepository
from heliot_terms.resolution.overlap_resolver import OverlapResolver


def _build_test_pipeline(note_output_policy: str = "annotate") -> StandardizationPipeline:
    aliases = [
        Alias(
            alias_raw="acetaminofene",
            alias_normalized="acetaminofene",
            target_id="active:paracetamolo",
            target_type=TargetType.INGREDIENT,
            language=AliasLanguage.IT,
            source=AliasSource.UMLS,
            alias_category=AliasCategory.CLINICAL,
            priority=90,
            safe_for_exact_match=True,
        ),
        Alias(
            alias_raw="Tachipirina",
            alias_normalized="tachipirina",
            target_id="brand:tachipirina",
            target_type=TargetType.DRUG_BRAND,
            language=AliasLanguage.IT,
            source=AliasSource.AIFA,
            alias_category=AliasCategory.CLINICAL,
            priority=85,
            safe_for_exact_match=True,
        ),
        Alias(
            alias_raw="AMPLIZER",
            alias_normalized="amplizer",
            target_id="brand:amplizer",
            target_type=TargetType.DRUG_BRAND,
            language=AliasLanguage.IT,
            source=AliasSource.AIFA,
            alias_category=AliasCategory.CLINICAL,
            priority=85,
            safe_for_exact_match=True,
        ),
    ]

    repository = KnowledgeBaseRepository(
        aliases=aliases,
        concepts_by_id={
            "active:paracetamolo": IngredientConcept(
                concept_id="active:paracetamolo",
                entity_type=EntityType.ACTIVE,
                canonical_it="paracetamolo",
                canonical_en="Paracetamol",
                sources=["UMLS"],
            ),
            "active:ampicillina_triidrata": IngredientConcept(
                concept_id="active:ampicillina_triidrata",
                entity_type=EntityType.ACTIVE,
                canonical_it="ampicillina triidrata",
                canonical_en="Ampicillin trihydrate",
                sources=["UMLS"],
            ),
        },
        products_by_id={},
        brands_by_id={
            "brand:tachipirina": DrugBrand(
                brand_id="brand:tachipirina",
                brand_name="Tachipirina",
                normalized_brand_name="tachipirina",
                product_ids=["drug:aic:1"],
                active_ingredient_ids=["active:paracetamolo"],
                active_ingredient_signatures=[["active:paracetamolo"]],
                brand_status=BrandStatus.SINGLE_ACTIVE_SIGNATURE,
            ),
            "brand:amplizer": DrugBrand(
                brand_id="brand:amplizer",
                brand_name="AMPLIZER",
                normalized_brand_name="amplizer",
                product_ids=["drug:aic:2", "drug:aic:3"],
                active_ingredient_ids=["active:ampicillina_triidrata"],
                active_ingredient_signatures=[
                    ["active:ampicillina_triidrata"],
                    ["active:ampicillina_triidrata", "active:ampicillina_anidra"],
                ],
                brand_status=BrandStatus.AMBIGUOUS_BRAND,
            ),
        },
    )

    return StandardizationPipeline(
        repository=repository,
        matcher=AhoCorasickMatcher.from_aliases(aliases),
        resolver=OverlapResolver(),
        config=StandardizationPipelineConfig(
            output_language="en",
            note_output_policy=note_output_policy,
        ),
    )


def test_pipeline_annotates_ingredient_match() -> None:
    pipeline = _build_test_pipeline(note_output_policy="annotate")

    result = pipeline.standardize("Allergia ad acetaminofene.")

    assert result.standardized_text == "Allergia ad acetaminofene [Paracetamol]."
    assert len(result.matches) == 1
    assert result.matches[0].target_id == "active:paracetamolo"


def test_pipeline_replaces_ingredient_match() -> None:
    pipeline = _build_test_pipeline(note_output_policy="replace")

    result = pipeline.standardize("Allergia ad acetaminofene.")

    assert result.standardized_text == "Allergia ad Paracetamol."


def test_pipeline_structured_only_keeps_normalized_text() -> None:
    pipeline = _build_test_pipeline(note_output_policy="structured_only")

    result = pipeline.standardize("Allergia ad acetaminofene.")

    assert result.standardized_text == "Allergia ad acetaminofene."
    assert len(result.matches) == 1


def test_pipeline_expands_non_ambiguous_brand() -> None:
    pipeline = _build_test_pipeline(note_output_policy="annotate")

    result = pipeline.standardize("Paziente allergico alla Tachipirina.")

    assert result.standardized_text == "Paziente allergico alla Tachipirina [Paracetamol]."
    assert len(result.matches) == 1
    assert result.matches[0].target_id == "brand:tachipirina"


def test_pipeline_does_not_annotate_ambiguous_brand() -> None:
    pipeline = _build_test_pipeline(note_output_policy="annotate")

    result = pipeline.standardize("Allergia ad AMPLIZER.")

    assert result.standardized_text == "Allergia ad AMPLIZER."
    assert len(result.matches) == 0
    assert len(result.ambiguous) == 1
    assert result.ambiguous[0].target_id == "brand:amplizer"