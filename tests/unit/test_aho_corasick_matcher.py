from heliot_terms.domain.enums import AliasCategory, AliasLanguage, AliasSource, TargetType
from heliot_terms.domain.models import Alias
from heliot_terms.matching.aho_corasick_matcher import AhoCorasickMatcher


def test_aho_corasick_matcher_finds_safe_alias() -> None:
    aliases = [
        Alias(
            alias_raw="acetaminofene",
            alias_normalized="acetaminofene",
            target_id="active:paracetamolo",
            target_type=TargetType.INGREDIENT,
            language=AliasLanguage.IT,
            source=AliasSource.UMLS,
            alias_category=AliasCategory.CLINICAL,
            safe_for_exact_match=True,
            requires_context=False,
        )
    ]

    matcher = AhoCorasickMatcher.from_aliases(aliases)
    matches = matcher.match("paziente allergico ad acetaminofene")

    assert len(matches) == 1
    assert matches[0].target_id == "active:paracetamolo"


def test_aho_corasick_matcher_ignores_unsafe_alias_by_default() -> None:
    aliases = [
        Alias(
            alias_raw="APAP",
            alias_normalized="apap",
            target_id="active:paracetamolo",
            target_type=TargetType.INGREDIENT,
            language=AliasLanguage.EN,
            source=AliasSource.UMLS,
            alias_category=AliasCategory.UNSAFE,
            safe_for_exact_match=False,
            requires_context=True,
        ),
        Alias(
            alias_raw="acetaminofene",
            alias_normalized="acetaminofene",
            target_id="active:paracetamolo",
            target_type=TargetType.INGREDIENT,
            language=AliasLanguage.IT,
            source=AliasSource.UMLS,
            alias_category=AliasCategory.CLINICAL,
            safe_for_exact_match=True,
            requires_context=False,
        ),
    ]

    matcher = AhoCorasickMatcher.from_aliases(aliases)
    matches = matcher.match("paziente allergico ad apap")

    assert matches == []


def test_aho_corasick_matcher_respects_boundaries() -> None:
    aliases = [
        Alias(
            alias_raw="sodio",
            alias_normalized="sodio",
            target_id="active_inactive:sodio",
            target_type=TargetType.INGREDIENT,
            language=AliasLanguage.IT,
            source=AliasSource.UMLS,
            alias_category=AliasCategory.CLINICAL,
            safe_for_exact_match=True,
            requires_context=False,
        )
    ]

    matcher = AhoCorasickMatcher.from_aliases(aliases)

    assert matcher.match("allergia a sodio") != []
    assert matcher.match("allergia a disodio") == []