from heliot_terms.domain.enums import TargetType
from heliot_terms.fallback.semantic.acceptance import SemanticAcceptancePolicy
from heliot_terms.fallback.semantic.models import SemanticIndexMetadata, SemanticSearchResult


def _result(target_id: str, similarity: float) -> SemanticSearchResult:
    metadata = SemanticIndexMetadata(
        vector_id=1,
        target_id=target_id,
        target_type=TargetType.INGREDIENT,
        alias_raw="paracetamolo",
        alias_normalized="paracetamolo",
        language="it",
        source="UMLS",
    )

    return SemanticSearchResult(
        metadata=metadata,
        similarity=similarity,
        distance=1.0 - similarity,
    )


def test_semantic_acceptance_accepts_clear_best_result() -> None:
    policy = SemanticAcceptancePolicy()

    decision = policy.decide(
        [
            _result("active:paracetamolo", 0.95),
            _result("active:ibuprofene", 0.85),
        ]
    )

    assert decision.accepted is True
    assert decision.accepted_result is not None
    assert decision.accepted_result.metadata.target_id == "active:paracetamolo"


def test_semantic_acceptance_rejects_below_threshold() -> None:
    policy = SemanticAcceptancePolicy()

    decision = policy.decide(
        [
            _result("active:paracetamolo", 0.70),
        ]
    )

    assert decision.accepted is False
    assert decision.reason == "below_semantic_similarity_threshold"


def test_semantic_acceptance_rejects_close_competing_target() -> None:
    policy = SemanticAcceptancePolicy()

    decision = policy.decide(
        [
            _result("active:paracetamolo", 0.93),
            _result("active:ibuprofene", 0.91),
        ]
    )

    assert decision.accepted is False
    assert decision.reason == "ambiguous_semantic_match"


def test_semantic_acceptance_does_not_treat_same_target_as_ambiguous() -> None:
    policy = SemanticAcceptancePolicy()

    decision = policy.decide(
        [
            _result("active:paracetamolo", 0.93),
            _result("active:paracetamolo", 0.92),
        ]
    )

    assert decision.accepted is True
    assert decision.accepted_result is not None
    assert decision.accepted_result.metadata.target_id == "active:paracetamolo"