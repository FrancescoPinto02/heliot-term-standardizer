from heliot_terms.domain.enums import AliasCategory, AliasLanguage, AliasSource, TargetType
from heliot_terms.domain.models import Alias
from heliot_terms.fallback.fuzzy.composite_matcher import (
    CompositeFuzzyMatcher,
    FuzzyFallbackConfig,
)


def _alias(
    alias_raw: str,
    alias_normalized: str,
    target_id: str,
) -> Alias:
    return Alias(
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


def test_composite_fuzzy_matcher_supports_symspell_only() -> None:
    matcher = CompositeFuzzyMatcher.from_aliases(
        aliases=[
            _alias(
                alias_raw="acetaminofene",
                alias_normalized="acetaminofene",
                target_id="active:paracetamolo",
            )
        ],
        config=FuzzyFallbackConfig(strategies=("symspell",)),
    )

    matches = matcher.match(
        text="reazione allergica ad acetaminofne",
        protected_spans=[],
    )

    assert len(matches) == 1
    assert matches[0].method == "symspell_fuzzy"


def test_composite_fuzzy_matcher_supports_ngram_rapidfuzz_only() -> None:
    matcher = CompositeFuzzyMatcher.from_aliases(
        aliases=[
            _alias(
                alias_raw="lisinopril didrato",
                alias_normalized="lisinopril didrato",
                target_id="active:lisinopril_diidrato",
            )
        ],
        config=FuzzyFallbackConfig(strategies=("ngram_rapidfuzz",)),
    )

    matches = matcher.match(
        text="il paziente non tollera didrato lisinopril",
        protected_spans=[],
    )

    assert len(matches) >= 1
    assert any(match.method == "ngram_rapidfuzz" for match in matches)


def test_composite_fuzzy_matcher_rejects_unknown_strategy() -> None:
    try:
        CompositeFuzzyMatcher.from_aliases(
            aliases=[],
            config=FuzzyFallbackConfig(strategies=("unknown_strategy",)),
        )
    except ValueError as exc:
        assert "Unsupported fuzzy strategies" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported fuzzy strategy.")