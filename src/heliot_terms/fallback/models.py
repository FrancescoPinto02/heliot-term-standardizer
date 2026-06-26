from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from heliot_terms.domain.models import Alias


@dataclass(frozen=True)
class FuzzyTextCandidate:
    """Candidate text span extracted from unconsumed note fragments."""

    text: str
    start: int
    end: int
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FuzzyScoredSuggestion:
    """SymSpell suggestion enriched with alias metadata and score."""

    candidate: FuzzyTextCandidate
    alias: Alias
    suggested_alias_normalized: str
    dictionary_key: str
    edit_distance: int
    score: float


@dataclass(frozen=True)
class FuzzyAcceptanceDecision:
    """Decision returned by the fuzzy acceptance policy."""

    accepted_suggestion: FuzzyScoredSuggestion | None
    reason: str
    top_suggestions: list[FuzzyScoredSuggestion] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        """Return True if a suggestion was accepted."""
        return self.accepted_suggestion is not None