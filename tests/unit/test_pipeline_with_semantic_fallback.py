"""Unit tests for pipeline integration with semantic fallback."""

from dataclasses import dataclass

from heliot_terms.domain.enums import EntityType, TargetType
from heliot_terms.domain.models import IngredientConcept
from heliot_terms.matching.aho_corasick_matcher import AhoCorasickMatcher
from heliot_terms.matching.models import MatchCandidate
from heliot_terms.pipeline.standardization_pipeline import (
    StandardizationPipeline,
    StandardizationPipelineConfig,
)
from heliot_terms.resources.knowledge_base_repository import KnowledgeBaseRepository
from heliot_terms.resolution.overlap_resolver import OverlapResolver


@dataclass
class FakeSemanticFallbackMatcher:
    """Fake semantic fallback returning one ingredient match."""

    def match(
        self,
        text: str,
        protected_spans: list[tuple[int, int]] | None = None,
    ) -> list[MatchCandidate]:
        start = text.index("paracetamolo")
        end = start + len("paracetamolo")

        protected_spans = protected_spans or []
        if any(start < protected_end and protected_start < end for protected_start, protected_end in protected_spans):
            return []

        return [
            MatchCandidate(
                surface="paracetamolo",
                normalized_surface="paracetamolo",
                start=start,
                end=end,
                target_id="active:paracetamolo",
                target_type=TargetType.INGREDIENT,
                alias_raw="paracetamolo",
                alias_category="clinical",
                confidence=0.95,
                priority=55,
                method="semantic_embedding",
                safe_for_exact_match=True,
                requires_context=False,
            )
        ]


def test_pipeline_uses_semantic_fallback_after_exact_and_fuzzy() -> None:
    repository = KnowledgeBaseRepository(
        aliases=[],
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

    pipeline = StandardizationPipeline(
        repository=repository,
        matcher=AhoCorasickMatcher.from_aliases([]),
        resolver=OverlapResolver(),
        config=StandardizationPipelineConfig(
            output_language="en",
            note_output_policy="annotate",
        ),
        fallback_matchers=[],
        semantic_fallback_matchers=[FakeSemanticFallbackMatcher()],
    )

    result = pipeline.standardize("Allergia a paracetamolo.")

    assert result.standardized_text == "Allergia a paracetamolo [Paracetamol]."
    assert len(result.matches) == 1
    assert result.matches[0].method == "semantic_embedding"
    assert result.metadata["num_semantic_resolved_matches"] == 1