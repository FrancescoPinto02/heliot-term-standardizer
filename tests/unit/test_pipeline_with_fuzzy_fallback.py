"""Unit tests for pipeline integration with fuzzy fallback."""

from heliot_terms.domain.enums import (
    AliasCategory,
    AliasLanguage,
    AliasSource,
    EntityType,
    TargetType,
)
from heliot_terms.domain.models import Alias, IngredientConcept
from heliot_terms.fallback.acceptance import FuzzyAcceptanceConfig
from heliot_terms.fallback.candidate_extractor import CandidateExtractorConfig
from heliot_terms.fallback.symspell_fuzzy_matcher import (
    SymSpellFuzzyMatcher,
    SymSpellFuzzyMatcherConfig,
)
from heliot_terms.matching.aho_corasick_matcher import AhoCorasickMatcher
from heliot_terms.pipeline.standardization_pipeline import (
    StandardizationPipeline,
    StandardizationPipelineConfig,
)
from heliot_terms.resources.knowledge_base_repository import KnowledgeBaseRepository
from heliot_terms.resolution.overlap_resolver import OverlapResolver


def test_pipeline_uses_fuzzy_fallback_on_unmatched_residue() -> None:
    alias = Alias(
        alias_raw="acetaminofene",
        alias_normalized="acetaminofene",
        target_id="active:paracetamolo",
        target_type=TargetType.INGREDIENT,
        language=AliasLanguage.IT,
        source=AliasSource.UMLS,
        alias_category=AliasCategory.CLINICAL,
        priority=90,
        safe_for_exact_match=True,
        requires_context=False,
        metadata={"policy_reason": "clinical_alias"},
    )

    repository = KnowledgeBaseRepository(
        aliases=[alias],
        concepts_by_id={
            "active:paracetamolo": IngredientConcept(
                concept_id="active:paracetamolo",
                entity_type=EntityType.ACTIVE,
                canonical_it="paracetamolo",
                canonical_en="Paracetamol",
                sources=["UMLS"],
            )
        },
        products_by_id={},
        brands_by_id={},
    )

    exact_matcher = AhoCorasickMatcher.from_aliases([alias])

    fuzzy_matcher = SymSpellFuzzyMatcher.from_aliases(
        [alias],
        SymSpellFuzzyMatcherConfig(
            max_dictionary_edit_distance=2,
            prefix_length=7,
            max_lookup_edit_distance=2,
            candidate_extractor=CandidateExtractorConfig(
                max_ngram_tokens=4,
                min_token_chars=3,
                min_candidate_chars=6,
            ),
            acceptance=FuzzyAcceptanceConfig(
                min_score_short=0.90,
                min_score_medium=0.85,
                min_score_long=0.80,
                ambiguity_margin=0.05,
            ),
        ),
    )

    pipeline = StandardizationPipeline(
        repository=repository,
        matcher=exact_matcher,
        resolver=OverlapResolver(),
        config=StandardizationPipelineConfig(
            output_language="en",
            note_output_policy="annotate",
        ),
        fallback_matchers=[fuzzy_matcher],
    )

    result = pipeline.standardize(
        "Il paziente ha presentato una reazione allergica ad acetaminofne."
    )

    assert result.standardized_text == (
        "Il paziente ha presentato una reazione allergica ad "
        "acetaminofne [Paracetamol]."
    )
    assert len(result.matches) == 1
    assert result.matches[0].method == "symspell_fuzzy"
    assert result.metadata["num_fallback_resolved_matches"] == 1