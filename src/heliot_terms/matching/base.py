"""Base interfaces for deterministic matchers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from heliot_terms.matching.models import MatchCandidate


class BaseMatcher(ABC):
    """Common interface for all exact matchers.

    Implementations can use Aho-Corasick, FlashText, regular expressions, or any
    other strategy. The rest of the pipeline should depend only on this class.
    """

    @abstractmethod
    def match(self, text: str) -> list[MatchCandidate]:
        """Return match candidates found in the provided normalized text."""
        raise NotImplementedError