from __future__ import annotations

from dataclasses import dataclass

from heliot_terms.domain.models import Alias
from heliot_terms.fallback.base import BaseFallbackMatcher
from heliot_terms.fallback.fuzzy.ngram_rapidfuzz_matcher import NgramRapidFuzzMatcher
from heliot_terms.fallback.fuzzy.symspell_matcher import SymSpellFuzzyMatcher
from heliot_terms.matching.models import MatchCandidate


SUPPORTED_FUZZY_STRATEGIES = frozenset(
    {
        "symspell",
        "ngram_rapidfuzz",
    }
)


@dataclass(frozen=True)
class FuzzyFallbackConfig:
    """Runtime configuration for the composite fuzzy fallback."""

    strategies: tuple[str, ...] = ("symspell",)


class CompositeFuzzyMatcher(BaseFallbackMatcher):
    """Run one or more fuzzy strategies on the same exact-match residuals."""

    def __init__(self, matchers: list[BaseFallbackMatcher]) -> None:
        self.matchers = matchers

    @classmethod
    def from_aliases(
        cls,
        aliases: list[Alias],
        config: FuzzyFallbackConfig | None = None,
    ) -> CompositeFuzzyMatcher:
        """Build the composite matcher from aliases and selected strategies."""
        config = config or FuzzyFallbackConfig()

        normalized_strategies = tuple(
            strategy.strip().lower()
            for strategy in config.strategies
            if strategy.strip()
        )

        unknown_strategies = set(normalized_strategies) - SUPPORTED_FUZZY_STRATEGIES
        if unknown_strategies:
            raise ValueError(
                f"Unsupported fuzzy strategies: {sorted(unknown_strategies)}"
            )

        matchers: list[BaseFallbackMatcher] = []

        if "symspell" in normalized_strategies:
            matchers.append(SymSpellFuzzyMatcher.from_aliases(aliases))

        if "ngram_rapidfuzz" in normalized_strategies:
            matchers.append(NgramRapidFuzzMatcher.from_aliases(aliases))

        return cls(matchers=matchers)

    def match(
        self,
        text: str,
        protected_spans: list[tuple[int, int]] | None = None,
    ) -> list[MatchCandidate]:
        """Return candidates from all enabled fuzzy strategies.

        Strategies are siblings: each receives the same protected spans from the
        exact matcher. They do not consume each other's output.
        """
        candidates: list[MatchCandidate] = []

        for matcher in self.matchers:
            candidates.extend(
                matcher.match(
                    text=text,
                    protected_spans=protected_spans or [],
                )
            )

        return candidates