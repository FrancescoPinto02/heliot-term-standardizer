"""Unit tests for overlap resolution."""

from heliot_terms.domain.enums import AliasCategory, TargetType
from heliot_terms.matching.models import MatchCandidate
from heliot_terms.resolution.overlap_resolver import OverlapResolver


def _candidate(
    surface: str,
    start: int,
    end: int,
    target_id: str,
    target_type: TargetType = TargetType.INGREDIENT,
    priority: int = 50,
) -> MatchCandidate:
    return MatchCandidate(
        surface=surface,
        normalized_surface=surface,
        start=start,
        end=end,
        target_id=target_id,
        target_type=target_type,
        alias_raw=surface,
        alias_category=AliasCategory.CLINICAL,
        priority=priority,
        method="test",
    )


def test_resolver_prefers_longer_overlapping_match() -> None:
    resolver = OverlapResolver()

    candidates = [
        _candidate("macrogol", 10, 18, "inactive:macrogol"),
        _candidate("macrogol 3350", 10, 23, "inactive:macrogol_3350"),
    ]

    resolved = resolver.resolve(candidates)

    assert len(resolved) == 1
    assert resolved[0].target_id == "inactive:macrogol_3350"


def test_resolver_keeps_non_overlapping_matches() -> None:
    resolver = OverlapResolver()

    candidates = [
        _candidate("paracetamolo", 10, 22, "active:paracetamolo"),
        _candidate("lattosio", 40, 48, "inactive:lattosio"),
    ]

    resolved = resolver.resolve(candidates)

    assert len(resolved) == 2


def test_resolver_uses_target_type_priority_when_length_is_equal() -> None:
    resolver = OverlapResolver()

    candidates = [
        _candidate(
            surface="tachipirina",
            start=10,
            end=21,
            target_id="active:some_wrong_ingredient",
            target_type=TargetType.INGREDIENT,
            priority=90,
        ),
        _candidate(
            surface="tachipirina",
            start=10,
            end=21,
            target_id="brand:tachipirina",
            target_type=TargetType.DRUG_BRAND,
            priority=80,
        ),
    ]

    resolved = resolver.resolve(candidates)

    assert len(resolved) == 1
    assert resolved[0].target_id == "brand:tachipirina"


def test_resolver_uses_alias_priority_when_length_and_type_are_equal() -> None:
    resolver = OverlapResolver()

    candidates = [
        _candidate("termine", 10, 17, "active:low_priority", priority=20),
        _candidate("termine", 10, 17, "active:high_priority", priority=90),
    ]

    resolved = resolver.resolve(candidates)

    assert len(resolved) == 1
    assert resolved[0].target_id == "active:high_priority"