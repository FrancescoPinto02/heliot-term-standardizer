from __future__ import annotations

from pathlib import Path

from heliot_terms.domain.enums import AliasCategory, TargetType
from heliot_terms.fallback.base import BaseFallbackMatcher
from heliot_terms.fallback.semantic.acceptance import SemanticAcceptancePolicy
from heliot_terms.fallback.semantic.encoder import SemanticEncoder
from heliot_terms.fallback.semantic.hnsw_index import TOP_K, SemanticVectorIndex
from heliot_terms.fallback.semantic.models import (
    SemanticAcceptanceDecision,
    SemanticTextCandidate,
)
from heliot_terms.fallback.semantic.ner_candidate_extractor import (
    NerSemanticCandidateExtractor,
)
from heliot_terms.matching.models import MatchCandidate
from heliot_terms.runtime.cache import LruCache

SEMANTIC_MATCH_PRIORITY = 55
SEMANTIC_DECISION_CACHE_SIZE = 10_000

class SemanticEmbeddingMatcher(BaseFallbackMatcher):
    """NER + embedding nearest-neighbor fallback matcher."""

    def __init__(
        self,
        encoder: SemanticEncoder,
        index: SemanticVectorIndex,
        candidate_extractor: NerSemanticCandidateExtractor | None = None,
        acceptance_policy: SemanticAcceptancePolicy | None = None,
    ) -> None:
        self.encoder = encoder
        self.index = index
        self.candidate_extractor = candidate_extractor or NerSemanticCandidateExtractor()
        self.acceptance_policy = acceptance_policy or SemanticAcceptancePolicy()
        self._decision_cache: LruCache[str, SemanticAcceptanceDecision] = LruCache(
            max_size=SEMANTIC_DECISION_CACHE_SIZE
        )

    @classmethod
    def from_index_dir(
        cls,
        index_dir: str | Path,
    ) -> SemanticEmbeddingMatcher:
        """Load a semantic matcher from a built index directory."""
        encoder = SemanticEncoder()
        index = SemanticVectorIndex.load(index_dir)

        return cls(
            encoder=encoder,
            index=index,
        )

    def match(
        self,
        text: str,
        protected_spans: list[tuple[int, int]] | None = None,
    ) -> list[MatchCandidate]:
        """Return accepted semantic embedding fallback matches."""
        candidates = self.candidate_extractor.extract(
            text=text,
            protected_spans=protected_spans or [],
        )

        if not candidates:
            return []

        decisions_by_index: dict[int, SemanticAcceptanceDecision] = {}
        uncached_candidates: list[tuple[int, SemanticTextCandidate]] = []

        for index, candidate in enumerate(candidates):
            cache_key = _semantic_cache_key(candidate.text)
            cached_decision = self._decision_cache.get(cache_key)

            if cached_decision is not None:
                decisions_by_index[index] = cached_decision
            else:
                uncached_candidates.append((index, candidate))

        if uncached_candidates:
            vectors = self.encoder.encode(
                [candidate.text for _, candidate in uncached_candidates]
            )

            for (index, candidate), vector in zip(
                uncached_candidates,
                vectors,
                strict=True,
            ):
                search_results = self.index.search(vector, top_k=TOP_K)
                decision = self.acceptance_policy.decide(search_results)

                self._decision_cache.set(
                    _semantic_cache_key(candidate.text),
                    decision,
                )

                decisions_by_index[index] = decision

        matches: list[MatchCandidate] = []

        for index, candidate in enumerate(candidates):
            decision = decisions_by_index[index]

            if not decision.accepted or decision.accepted_result is None:
                continue

            matches.append(
                self._to_match_candidate(
                    candidate=candidate,
                    decision=decision,
                )
            )

        return matches

    def _to_match_candidate(
        self,
        candidate: SemanticTextCandidate,
        decision: SemanticAcceptanceDecision,
    ) -> MatchCandidate:
        """Convert an accepted semantic result into a MatchCandidate."""
        accepted = decision.accepted_result

        if accepted is None:
            raise ValueError("Cannot convert a rejected semantic decision.")

        metadata = accepted.metadata

        return MatchCandidate(
            surface=candidate.text,
            normalized_surface=candidate.text,
            start=candidate.start,
            end=candidate.end,
            target_id=metadata.target_id,
            target_type=TargetType(metadata.target_type),
            alias_raw=metadata.alias_raw,
            alias_category=AliasCategory.CLINICAL,
            confidence=accepted.similarity,
            priority=SEMANTIC_MATCH_PRIORITY,
            method="semantic_embedding",
            safe_for_exact_match=True,
            requires_context=False,
            metadata={
                **metadata.metadata,
                "semantic_reason": decision.reason,
                "semantic_similarity": accepted.similarity,
                "semantic_alias_raw": metadata.alias_raw,
                "semantic_alias_normalized": metadata.alias_normalized,
                "ner_label": candidate.label,
                "ner_score": candidate.score,
                "top_neighbors": [
                    {
                        "target_id": result.metadata.target_id,
                        "alias_raw": result.metadata.alias_raw,
                        "alias_normalized": result.metadata.alias_normalized,
                        "similarity": result.similarity,
                    }
                    for result in decision.top_results
                ],
            },
        )


def _semantic_cache_key(text: str) -> str:
    """Return a stable cache key for a semantic candidate."""
    return " ".join(text.lower().strip().split())