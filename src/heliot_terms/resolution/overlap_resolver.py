"""Overlap resolution for terminology matches.

The deterministic matcher can return overlapping candidates. This module
selects a non-overlapping set of matches using configurable priorities.
"""

from __future__ import annotations

from dataclasses import dataclass

from heliot_terms.domain.enums import TargetType
from heliot_terms.matching.models import MatchCandidate
from heliot_terms.resolution.models import ResolvedMatch


_DEFAULT_TARGET_TYPE_PRIORITY = {
    TargetType.DRUG_PRODUCT: 3,
    TargetType.DRUG_BRAND: 2,
    TargetType.INGREDIENT: 1,
}


@dataclass(frozen=True)
class OverlapResolverConfig:
    """Configuration for overlap resolution."""

    prefer_longest_match: bool = True
    target_type_priority: dict[TargetType, int] | None = None


class OverlapResolver:
    """Resolve overlapping match candidates.

    The resolver prefers longer and more specific matches. This prevents cases
    like 'macrogol' being selected when 'macrogol 3350' is also present.
    """

    def __init__(self, config: OverlapResolverConfig | None = None) -> None:
        self.config = config or OverlapResolverConfig()
        self.target_type_priority = (
            self.config.target_type_priority or _DEFAULT_TARGET_TYPE_PRIORITY
        )

    def resolve(self, candidates: list[MatchCandidate]) -> list[ResolvedMatch]:
        """Return a non-overlapping list of resolved matches."""
        if not candidates:
            return []

        sorted_candidates = sorted(candidates, key=self._ranking_key, reverse=True)

        selected: list[MatchCandidate] = []

        for candidate in sorted_candidates:
            if self._overlaps_any(candidate, selected):
                continue
            selected.append(candidate)

        selected.sort(key=lambda match: (match.start, match.end))

        return [self._to_resolved_match(candidate) for candidate in selected]

    def _ranking_key(self, candidate: MatchCandidate) -> tuple[int, int, float, int, int]:
        """Return sorting key used to rank candidates.

        Order:
        1. longer match
        2. target type priority
        3. confidence
        4. alias priority
        5. earlier start position
        """
        length_score = candidate.end - candidate.start if self.config.prefer_longest_match else 0
        type_score = self.target_type_priority.get(candidate.target_type, 0)

        # For reverse sorting, earlier matches should rank slightly higher.
        earlier_position_score = -candidate.start

        return (
            length_score,
            type_score,
            candidate.confidence,
            candidate.priority,
            earlier_position_score,
        )

    def _overlaps_any(
        self,
        candidate: MatchCandidate,
        selected: list[MatchCandidate],
    ) -> bool:
        """Return True if candidate overlaps any already selected match."""
        return any(self._overlaps(candidate, other) for other in selected)

    def _overlaps(self, left: MatchCandidate, right: MatchCandidate) -> bool:
        """Return True if two half-open spans overlap."""
        return left.start < right.end and right.start < left.end

    def _to_resolved_match(self, candidate: MatchCandidate) -> ResolvedMatch:
        """Convert a matcher candidate into a resolved match."""
        return ResolvedMatch(
            surface=candidate.surface,
            normalized_surface=candidate.normalized_surface,
            start=candidate.start,
            end=candidate.end,
            target_id=candidate.target_id,
            target_type=candidate.target_type,
            alias_raw=candidate.alias_raw,
            alias_category=candidate.alias_category,
            confidence=candidate.confidence,
            priority=candidate.priority,
            method=candidate.method,
            safe_for_exact_match=candidate.safe_for_exact_match,
            requires_context=candidate.requires_context,
            metadata=candidate.metadata,
        )