"""Unit tests for fuzzy acceptance policy."""

from heliot_terms.domain.enums import AliasCategory, AliasLanguage, AliasSource, TargetType
from heliot_terms.domain.models import Alias
from heliot_terms.fallback.fuzzy.acceptance import FuzzyAcceptanceConfig, FuzzyAcceptancePolicy
from heliot_terms.fallback.fuzzy.models import FuzzyScoredSuggestion, FuzzyTextCandidate


def _alias(
    target_id: str,
    alias_raw: str = "acetaminofene",
    alias_normalized: str = "acetaminofene",
    priority: int = 90,
) -> Alias:
    return Alias(
        alias_raw=alias_raw,
        alias_normalized=alias_normalized,
        target_id=target_id,
        target_type=TargetType.INGREDIENT,
        language=AliasLanguage.IT,
        source=AliasSource.UMLS,
        alias_category=AliasCategory.CLINICAL,
        priority=priority,
        safe_for_exact_match=True,
        requires_context=False,
        metadata={"policy_reason": "clinical_alias"},
    )


def _suggestion(
    target_id: str,
    score: float,
    edit_distance: int = 1,
    candidate_text: str = "acetaminofne",
) -> FuzzyScoredSuggestion:
    candidate = FuzzyTextCandidate(
        text=candidate_text,
        start=0,
        end=len(candidate_text),
        token_count=1,
    )

    alias = _alias(target_id=target_id)

    return FuzzyScoredSuggestion(
        candidate=candidate,
        alias=alias,
        suggested_alias_normalized=alias.alias_normalized,
        dictionary_key=alias.alias_normalized,
        edit_distance=edit_distance,
        score=score,
    )


def test_acceptance_policy_accepts_clear_best_suggestion() -> None:
    policy = FuzzyAcceptancePolicy(
        FuzzyAcceptanceConfig(
            min_score_short=0.95,
            min_score_medium=0.90,
            min_score_long=0.86,
            ambiguity_margin=0.05,
        )
    )

    decision = policy.decide(
        [
            _suggestion("active:paracetamolo", score=0.92),
            _suggestion("active:other", score=0.80),
        ]
    )

    assert decision.accepted is True
    assert decision.accepted_suggestion is not None
    assert decision.accepted_suggestion.alias.target_id == "active:paracetamolo"


def test_acceptance_policy_rejects_below_threshold() -> None:
    policy = FuzzyAcceptancePolicy(
        FuzzyAcceptanceConfig(
            min_score_short=0.95,
            min_score_medium=0.90,
            min_score_long=0.86,
        )
    )

    decision = policy.decide(
        [
            _suggestion("active:paracetamolo", score=0.70),
        ]
    )

    assert decision.accepted is False
    assert decision.reason == "below_length_dependent_threshold"


def test_acceptance_policy_rejects_ambiguous_close_targets() -> None:
    policy = FuzzyAcceptancePolicy(
        FuzzyAcceptanceConfig(
            ambiguity_margin=0.05,
            min_score_short=0.80,
            min_score_medium=0.80,
            min_score_long=0.80,
        )
    )

    decision = policy.decide(
        [
            _suggestion("active:first", score=0.92),
            _suggestion("active:second", score=0.90),
        ]
    )

    assert decision.accepted is False
    assert decision.reason == "ambiguous_fuzzy_suggestion"