from heliot_terms.domain.enums import AliasCategory, AliasLanguage, AliasSource, TargetType
from heliot_terms.domain.models import Alias
from heliot_terms.fallback.fuzzy.ngram_rapidfuzz_matcher import NgramRapidFuzzMatcher


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


def test_ngram_rapidfuzz_matches_reordered_tokens() -> None:
    matcher = NgramRapidFuzzMatcher.from_aliases(
        [
            _alias(
                alias_raw="lisinopril didrato",
                alias_normalized="lisinopril didrato",
                target_id="active:lisinopril_diidrato",
            )
        ]
    )

    matches = matcher.match(
        text="il paziente non tollera didrato lisinopril",
        protected_spans=[],
    )

    assert len(matches) >= 1
    best = max(matches, key=lambda match: match.confidence)

    assert best.target_id == "active:lisinopril_diidrato"
    assert best.method == "ngram_rapidfuzz"


def test_ngram_rapidfuzz_trims_trailing_stopword_candidate() -> None:
    matcher = NgramRapidFuzzMatcher.from_aliases(
        [
            _alias(
                alias_raw="lisinopril didrato",
                alias_normalized="lisinopril didrato",
                target_id="active:lisinopril_diidrato",
            )
        ]
    )

    matches = matcher.match(
        text="il paziente non tollera lisinopril didrato e macrogol",
        protected_spans=[],
    )

    assert matches
    best = max(matches, key=lambda match: match.confidence)

    assert best.surface == "lisinopril didrato"
    assert best.surface != "lisinopril didrato e"


def test_ngram_rapidfuzz_does_not_index_brand_aliases() -> None:
    matcher = NgramRapidFuzzMatcher.from_aliases(
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


def test_ngram_rapidfuzz_does_not_index_unsafe_aliases() -> None:
    matcher = NgramRapidFuzzMatcher.from_aliases(
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