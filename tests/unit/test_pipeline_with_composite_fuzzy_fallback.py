from heliot_terms.domain.enums import (
    AliasCategory,
    AliasLanguage,
    AliasSource,
    EntityType,
    TargetType,
)
from heliot_terms.domain.models import Alias, IngredientConcept
from heliot_terms.fallback.fuzzy.composite_matcher import (
    CompositeFuzzyMatcher,
    FuzzyFallbackConfig,
)
from heliot_terms.matching.aho_corasick_matcher import AhoCorasickMatcher
from heliot_terms.pipeline.standardization_pipeline import (
    StandardizationPipeline,
    StandardizationPipelineConfig,
)
from heliot_terms.resources.knowledge_base_repository import KnowledgeBaseRepository
from heliot_terms.resolution.overlap_resolver import OverlapResolver


def _alias(
    alias_raw: str,
    alias_normalized: str,
    target_id: str,
    canonical_en: str,
) -> tuple[Alias, IngredientConcept]:
    alias = Alias(
        alias_raw=alias_raw,
        alias_normalized=alias_normalized,
        target_id=target_id,
        target_type=TargetType.INGREDIENT,
        language=AliasLanguage.IT,
        source=AliasSource.UMLS,
        alias_category=AliasCategory.CLINICAL,
        priority=90,
        safe_for_exact_match=True,
        requires_context=False,
        metadata={"policy_reason": "clinical_alias"},
    )

    concept = IngredientConcept(
        concept_id=target_id,
        entity_type=EntityType.ACTIVE,
        canonical_it=alias_raw,
        canonical_en=canonical_en,
        sources=["UMLS"],
    )

    return alias, concept


def test_pipeline_with_composite_fuzzy_handles_symspell_typo() -> None:
    alias, concept = _alias(
        alias_raw="acetaminofene",
        alias_normalized="acetaminofene",
        target_id="active:paracetamolo",
        canonical_en="Paracetamol",
    )

    pipeline = _pipeline(
        aliases=[alias],
        concepts=[concept],
        strategies=("symspell", "ngram_rapidfuzz"),
    )

    result = pipeline.standardize("Allergia ad acetaminofne.")

    assert result.standardized_text == "Allergia ad acetaminofne [Paracetamol]."
    assert len(result.matches) == 1
    assert result.matches[0].target_id == "active:paracetamolo"


def test_pipeline_with_composite_fuzzy_handles_reordered_tokens() -> None:
    alias, concept = _alias(
        alias_raw="lisinopril didrato",
        alias_normalized="lisinopril didrato",
        target_id="active:lisinopril_diidrato",
        canonical_en="lisinopril dihydrate",
    )

    pipeline = _pipeline(
        aliases=[alias],
        concepts=[concept],
        strategies=("symspell", "ngram_rapidfuzz"),
    )

    result = pipeline.standardize(
        "Il paziente non tollera didrato lisinopril."
    )

    assert result.standardized_text == (
        "Il paziente non tollera didrato lisinopril [lisinopril dihydrate]."
    )
    assert len(result.matches) == 1
    assert result.matches[0].target_id == "active:lisinopril_diidrato"


def _pipeline(
    aliases: list[Alias],
    concepts: list[IngredientConcept],
    strategies: tuple[str, ...],
) -> StandardizationPipeline:
    repository = KnowledgeBaseRepository(
        aliases=aliases,
        concepts_by_id={concept.concept_id: concept for concept in concepts},
        products_by_id={},
        brands_by_id={},
    )

    exact_matcher = AhoCorasickMatcher.from_aliases(aliases)

    fuzzy_matcher = CompositeFuzzyMatcher.from_aliases(
        aliases=aliases,
        config=FuzzyFallbackConfig(strategies=strategies),
    )

    return StandardizationPipeline(
        repository=repository,
        matcher=exact_matcher,
        resolver=OverlapResolver(),
        config=StandardizationPipelineConfig(
            output_language="en",
            note_output_policy="annotate",
        ),
        fallback_matchers=[fuzzy_matcher],
    )