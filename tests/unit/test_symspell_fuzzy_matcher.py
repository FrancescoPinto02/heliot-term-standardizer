"""Unit tests for the SymSpell fuzzy fallback matcher."""

from heliot_terms.domain.enums import AliasCategory, AliasLanguage, AliasSource, TargetType
from heliot_terms.domain.models import Alias
from heliot_terms.fallback.acceptance import FuzzyAcceptanceConfig
from heliot_terms.fallback.candidate_extractor import CandidateExtractorConfig
from heliot_terms.fallback.symspell_fuzzy_matcher import (
    SymSpellFuzzyMatcher,
    SymSpellFuzzyMatcherConfig,
)


def _alias(
    alias_raw: str,
    alias_normalized: str,
    target_id: str,
    target_type: TargetType = TargetType.INGREDIENT,
    alias_category: AliasCategory = AliasCategory.CLINICAL,
    safe_for_exact_match: bool = True,
    policy_reason: str = "clinical_alias",
) -> Alias:
    return Alias(
        alias_raw=alias_raw,
        alias_normalized=alias_normalized,
        target_id=target_id,
        target_type=target_type,
        language=AliasLanguage.IT,
        source=AliasSource.UMLS,
        alias_category=alias_category,
        priority=90,
        safe_for_exact_match=safe_for_exact_match,
        requires_context=False,
        metadata={"policy_reason": policy_reason},
    )


def _matcher(aliases: list[Alias]) -> SymSpellFuzzyMatcher:
    return SymSpellFuzzyMatcher.from_aliases(
        aliases,
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


def test_symspell_fuzzy_matcher_recovers_typo_for_ingredient_alias() -> None:
    matcher = _matcher(
        [
            _alias(
                alias_raw="acetaminofene",
                alias_normalized="acetaminofene",
                target_id="active:paracetamolo",
            )
        ]
    )

    matches = matcher.match(
        text="reazione allergica ad acetaminofne",
        protected_spans=[],
    )

    assert len(matches) == 1
    assert matches[0].target_id == "active:paracetamolo"
    assert matches[0].method == "symspell_fuzzy"
    assert matches[0].metadata["suggested_alias_normalized"] == "acetaminofene"


def test_symspell_fuzzy_matcher_respects_protected_spans() -> None:
    matcher = _matcher(
        [
            _alias(
                alias_raw="acetaminofene",
                alias_normalized="acetaminofene",
                target_id="active:paracetamolo",
            )
        ]
    )

    text = "reazione allergica ad acetaminofne"
    start = text.index("acetaminofne")
    end = start + len("acetaminofne")

    matches = matcher.match(
        text=text,
        protected_spans=[(start, end)],
    )

    assert matches == []


def test_symspell_fuzzy_matcher_does_not_index_brand_aliases() -> None:
    matcher = _matcher(
        [
            _alias(
                alias_raw="Tachipirina",
                alias_normalized="tachipirina",
                target_id="brand:tachipirina",
                target_type=TargetType.DRUG_BRAND,
            )
        ]
    )

    matches = matcher.match(
        text="allergia a tachipirna",
        protected_spans=[],
    )

    assert matches == []


def test_symspell_fuzzy_matcher_does_not_index_unsafe_aliases() -> None:
    matcher = _matcher(
        [
            _alias(
                alias_raw="APAP",
                alias_normalized="apap",
                target_id="active:paracetamolo",
                alias_category=AliasCategory.UNSAFE,
                safe_for_exact_match=False,
                policy_reason="short_acronym_alias",
            )
        ]
    )

    matches = matcher.match(
        text="allergia ad apap",
        protected_spans=[],
    )

    assert matches == []