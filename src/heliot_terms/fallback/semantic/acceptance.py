from __future__ import annotations

from heliot_terms.fallback.semantic.models import (
    SemanticAcceptanceDecision,
    SemanticSearchResult,
)


MIN_COSINE_SIMILARITY = 0.88
AMBIGUITY_MARGIN = 0.04
MAX_TOP_RESULTS_TO_KEEP = 10


class SemanticAcceptancePolicy:
    """Accept semantic matches only when they are confident and unambiguous."""

    def decide(
        self,
        results: list[SemanticSearchResult],
    ) -> SemanticAcceptanceDecision:
        """Return an acceptance decision for semantic nearest-neighbor results."""
        if not results:
            return SemanticAcceptanceDecision(
                accepted_result=None,
                reason="no_semantic_neighbors",
            )

        ranked = sorted(results, key=lambda item: item.similarity, reverse=True)
        top_results = ranked[:MAX_TOP_RESULTS_TO_KEEP]
        best = top_results[0]

        if best.similarity < MIN_COSINE_SIMILARITY:
            return SemanticAcceptanceDecision(
                accepted_result=None,
                reason="below_semantic_similarity_threshold",
                top_results=top_results,
            )

        nearest_competing_target = self._nearest_competing_target(
            best=best,
            results=top_results[1:],
        )

        if nearest_competing_target is not None:
            margin = best.similarity - nearest_competing_target.similarity
            if margin < AMBIGUITY_MARGIN:
                return SemanticAcceptanceDecision(
                    accepted_result=None,
                    reason="ambiguous_semantic_match",
                    top_results=top_results,
                )

        return SemanticAcceptanceDecision(
            accepted_result=best,
            reason="accepted",
            top_results=top_results,
        )

    def _nearest_competing_target(
        self,
        best: SemanticSearchResult,
        results: list[SemanticSearchResult],
    ) -> SemanticSearchResult | None:
        """Return the nearest neighbor pointing to a different target."""
        for result in results:
            if result.metadata.target_id != best.metadata.target_id:
                return result

        return None