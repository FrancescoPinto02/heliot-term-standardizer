from __future__ import annotations

from abc import ABC, abstractmethod

from heliot_terms.matching.models import MatchCandidate


class BaseFallbackMatcher(ABC):
    """Common interface for fallback matchers.

    Fallback matchers receive spans already consumed by exact matching, so they
    can avoid searching inside high-confidence deterministic matches.
    """

    @abstractmethod
    def match(
        self,
        text: str,
        protected_spans: list[tuple[int, int]] | None = None,
    ) -> list[MatchCandidate]:
        """Return fallback match candidates."""
        raise NotImplementedError