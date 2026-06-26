"""Acceptance policy for fuzzy suggestions.

The fuzzy fallback should be conservative. This module decides whether the best
suggestion is strong enough and sufficiently separated from alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass

from heliot_terms.fallback.fuzzy.models import (
    FuzzyAcceptanceDecision,
    FuzzyScoredSuggestion,
)


@dataclass(frozen=True)
class FuzzyAcceptanceConfig:
    """Configuration for fuzzy suggestion acceptance."""

    max_suggestions: int = 5
    ambiguity_margin: float = 0.05

    short_max_chars: int = 8
    medium_max_chars: int = 14

    min_score_short: float = 0.95
    min_score_medium: float = 0.90
    min_score_long: float = 0.86


class FuzzyAcceptancePolicy:
    """Accept or reject fuzzy suggestions using conservative rules."""

    def __init__(self, config: FuzzyAcceptanceConfig | None = None) -> None:
        self.config = config or FuzzyAcceptanceConfig()

    def decide(
        self,
        suggestions: list[FuzzyScoredSuggestion],
    ) -> FuzzyAcceptanceDecision:
        """Return an acceptance decision for a list of scored suggestions."""
        if not suggestions:
            return FuzzyAcceptanceDecision(
                accepted_suggestion=None,
                reason="no_suggestions",
            )

        ranked = sorted(
            suggestions,
            key=lambda item: (
                item.score,
                -item.edit_distance,
                item.alias.priority,
            ),
            reverse=True,
        )

        top_suggestions = ranked[: self.config.max_suggestions]
        best = top_suggestions[0]

        threshold = self._threshold_for_candidate(best)
        if best.score < threshold:
            return FuzzyAcceptanceDecision(
                accepted_suggestion=None,
                reason="below_length_dependent_threshold",
                top_suggestions=top_suggestions,
            )

        nearest_competing_target = self._nearest_competing_target(
            best=best,
            suggestions=top_suggestions[1:],
        )

        if nearest_competing_target is not None:
            margin = best.score - nearest_competing_target.score
            if margin < self.config.ambiguity_margin:
                return FuzzyAcceptanceDecision(
                    accepted_suggestion=None,
                    reason="ambiguous_fuzzy_suggestion",
                    top_suggestions=top_suggestions,
                )

        return FuzzyAcceptanceDecision(
            accepted_suggestion=best,
            reason="accepted",
            top_suggestions=top_suggestions,
        )

    def _threshold_for_candidate(self, suggestion: FuzzyScoredSuggestion) -> float:
        """Return the minimum score required for the candidate length."""
        candidate_length = len(suggestion.candidate.text.replace(" ", ""))

        if candidate_length <= self.config.short_max_chars:
            return self.config.min_score_short

        if candidate_length <= self.config.medium_max_chars:
            return self.config.min_score_medium

        return self.config.min_score_long

    def _nearest_competing_target(
        self,
        best: FuzzyScoredSuggestion,
        suggestions: list[FuzzyScoredSuggestion],
    ) -> FuzzyScoredSuggestion | None:
        """Return the strongest suggestion pointing to a different target."""
        for suggestion in suggestions:
            if suggestion.alias.target_id != best.alias.target_id:
                return suggestion

        return None