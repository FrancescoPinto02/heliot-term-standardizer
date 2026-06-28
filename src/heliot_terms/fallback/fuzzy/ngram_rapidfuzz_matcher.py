from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace

from rapidfuzz import fuzz

from heliot_terms.domain.enums import AliasCategory, TargetType
from heliot_terms.domain.models import Alias
from heliot_terms.fallback.fuzzy.acceptance import FuzzyAcceptanceConfig, FuzzyAcceptancePolicy
from heliot_terms.fallback.base import BaseFallbackMatcher
from heliot_terms.fallback.fuzzy.candidate_extractor import (
    DEFAULT_STOPWORDS,
    CandidateExtractorConfig,
    ResidualCandidateExtractor,
)
from heliot_terms.fallback.fuzzy.models import (
    FuzzyAcceptanceDecision,
    FuzzyScoredSuggestion,
    FuzzyTextCandidate,
)
from heliot_terms.matching.models import MatchCandidate
from heliot_terms.runtime.cache import LruCache

# Candidate extraction.
MAX_NGRAM_TOKENS = 4
MIN_TOKEN_CHARS = 3
MIN_CANDIDATE_CHARS = 6

# Blocking.
CHAR_NGRAM_SIZE = 3
MIN_SHARED_NGRAMS = 2
MIN_NGRAM_OVERLAP_RATIO = 0.35
MAX_BLOCK_CANDIDATES = 100
MIN_INFORMATIVE_TOKEN_DICE = 0.75
SCORING_CACHE_SIZE = 10_000

# Acceptance.
MAX_SUGGESTIONS = 5
AMBIGUITY_MARGIN = 0.05
SHORT_MAX_CHARS = 8
MEDIUM_MAX_CHARS = 14
MIN_SCORE_SHORT = 0.96
MIN_SCORE_MEDIUM = 0.91
MIN_SCORE_LONG = 0.88

# Alias indexing.
ALLOWED_POLICY_REASONS = frozenset({"clinical_alias"})
TARGET_TYPES = frozenset({TargetType.INGREDIENT})


@dataclass(frozen=True)
class IndexedRapidFuzzAlias:
    """Alias payload stored in the n-gram blocking index."""

    alias_id: int
    alias: Alias
    normalized_text: str
    char_ngrams: frozenset[str]


class NgramRapidFuzzMatcher(BaseFallbackMatcher):
    """Fallback matcher using character n-gram blocking and RapidFuzz scoring."""

    def __init__(
        self,
        indexed_aliases: list[IndexedRapidFuzzAlias],
        ngram_to_alias_ids: dict[str, set[int]],
        candidate_extractor: ResidualCandidateExtractor | None = None,
        acceptance_policy: FuzzyAcceptancePolicy | None = None,
    ) -> None:
        self.indexed_aliases = indexed_aliases
        self.ngram_to_alias_ids = ngram_to_alias_ids
        self.candidate_extractor = candidate_extractor or ResidualCandidateExtractor(
            CandidateExtractorConfig(
                max_ngram_tokens=MAX_NGRAM_TOKENS,
                min_token_chars=MIN_TOKEN_CHARS,
                min_candidate_chars=MIN_CANDIDATE_CHARS,
            )
        )
        self.acceptance_policy = acceptance_policy or FuzzyAcceptancePolicy(
            FuzzyAcceptanceConfig(
                max_suggestions=MAX_SUGGESTIONS,
                ambiguity_margin=AMBIGUITY_MARGIN,
                short_max_chars=SHORT_MAX_CHARS,
                medium_max_chars=MEDIUM_MAX_CHARS,
                min_score_short=MIN_SCORE_SHORT,
                min_score_medium=MIN_SCORE_MEDIUM,
                min_score_long=MIN_SCORE_LONG,
            )
        )
        self._decision_cache: LruCache[str, FuzzyAcceptanceDecision] = LruCache(
            max_size=SCORING_CACHE_SIZE
        )

    @classmethod
    def from_aliases(cls, aliases: list[Alias]) -> NgramRapidFuzzMatcher:
        """Build the matcher index from processed aliases."""
        indexed_aliases: list[IndexedRapidFuzzAlias] = []
        ngram_to_alias_ids: dict[str, set[int]] = defaultdict(set)

        for alias in aliases:
            if not _should_index_alias(alias):
                continue

            char_ngrams = _char_ngrams(alias.alias_normalized)

            if not char_ngrams:
                continue

            alias_id = len(indexed_aliases)

            indexed_alias = IndexedRapidFuzzAlias(
                alias_id=alias_id,
                alias=alias,
                normalized_text=alias.alias_normalized,
                char_ngrams=frozenset(char_ngrams),
            )

            indexed_aliases.append(indexed_alias)

            for ngram in char_ngrams:
                ngram_to_alias_ids[ngram].add(alias_id)

        return cls(
            indexed_aliases=indexed_aliases,
            ngram_to_alias_ids=dict(ngram_to_alias_ids),
        )

    def match(
        self,
        text: str,
        protected_spans: list[tuple[int, int]] | None = None,
    ) -> list[MatchCandidate]:
        """Return accepted RapidFuzz fallback match candidates."""
        text_candidates = self.candidate_extractor.extract(
            text=text,
            protected_spans=protected_spans or [],
        )

        matches: list[MatchCandidate] = []

        for text_candidate in text_candidates:
            decision = self._cached_decision_for_candidate(text_candidate)

            if not decision.accepted or decision.accepted_suggestion is None:
                continue

            matches.append(
                self._to_match_candidate(
                    accepted=decision.accepted_suggestion,
                    decision_reason=decision.reason,
                    top_suggestions=decision.top_suggestions,
                )
            )

        return matches

    def _cached_decision_for_candidate(
        self,
        candidate: FuzzyTextCandidate,
    ) -> FuzzyAcceptanceDecision:
        """Return cached or newly computed RapidFuzz decision for a candidate.

        Cached decisions are rebuilt with the current candidate span so offsets
        remain correct across different notes.
        """
        cache_key = _candidate_cache_key(candidate.text)

        cached_decision = self._decision_cache.get(cache_key)
        if cached_decision is not None:
            return self._decision_with_candidate(
                decision=cached_decision,
                candidate=candidate,
            )

        scored_suggestions = self._score_candidate(candidate)
        decision = self.acceptance_policy.decide(scored_suggestions)

        self._decision_cache.set(cache_key, decision)

        return decision

    def _decision_with_candidate(
        self,
        decision: FuzzyAcceptanceDecision,
        candidate: FuzzyTextCandidate,
    ) -> FuzzyAcceptanceDecision:
        """Return a cached decision rebound to the current candidate span."""
        accepted_suggestion = (
            replace(decision.accepted_suggestion, candidate=candidate)
            if decision.accepted_suggestion is not None
            else None
        )

        top_suggestions = [
            replace(suggestion, candidate=candidate)
            for suggestion in decision.top_suggestions
        ]

        return FuzzyAcceptanceDecision(
            accepted_suggestion=accepted_suggestion,
            reason=decision.reason,
            top_suggestions=top_suggestions,
        )

    def _score_candidate(
        self,
        candidate: FuzzyTextCandidate,
    ) -> list[FuzzyScoredSuggestion]:
        """Return scored alias suggestions for one extracted candidate."""
        candidate_ngrams = _char_ngrams(candidate.text)

        if not candidate_ngrams:
            return []

        blocked_aliases = self._blocked_aliases(candidate_ngrams)

        scored: list[FuzzyScoredSuggestion] = []

        for indexed_alias, shared_count, overlap_ratio in blocked_aliases:
            token_sort_score = (
                fuzz.token_sort_ratio(
                    candidate.text,
                    indexed_alias.normalized_text,
                )
                / 100.0
            )
            token_set_score = (
                fuzz.token_set_ratio(
                    candidate.text,
                    indexed_alias.normalized_text,
                )
                / 100.0
            )

            informative_token_dice = _informative_token_dice(
                candidate_text=candidate.text,
                alias_text=indexed_alias.normalized_text,
            )

            if informative_token_dice < MIN_INFORMATIVE_TOKEN_DICE:
                continue

            raw_score = max(token_sort_score, token_set_score)
            score = raw_score * informative_token_dice

            pseudo_edit_distance = _pseudo_edit_distance(
                candidate_text=candidate.text,
                alias_text=indexed_alias.normalized_text,
                score=score,
            )

            scored.append(
                FuzzyScoredSuggestion(
                    candidate=candidate,
                    alias=indexed_alias.alias,
                    suggested_alias_normalized=indexed_alias.normalized_text,
                    dictionary_key=indexed_alias.normalized_text,
                    edit_distance=pseudo_edit_distance,
                    score=score,
                    metadata={
                        "rapidfuzz_token_sort_score": token_sort_score,
                        "rapidfuzz_token_set_score": token_set_score,
                        "raw_rapidfuzz_score": raw_score,
                        "informative_token_dice": informative_token_dice,
                        "shared_ngrams": shared_count,
                        "ngram_overlap_ratio": overlap_ratio,
                    },
                )
            )

        return scored

    def _blocked_aliases(
        self,
        candidate_ngrams: set[str],
    ) -> list[tuple[IndexedRapidFuzzAlias, int, float]]:
        """Return aliases passing the n-gram blocking filter."""
        alias_hit_counts: Counter[int] = Counter()

        for ngram in candidate_ngrams:
            for alias_id in self.ngram_to_alias_ids.get(ngram, set()):
                alias_hit_counts[alias_id] += 1

        blocked: list[tuple[IndexedRapidFuzzAlias, int, float]] = []

        for alias_id, shared_count in alias_hit_counts.items():
            indexed_alias = self.indexed_aliases[alias_id]

            if shared_count < MIN_SHARED_NGRAMS:
                continue

            overlap_denominator = max(
                min(len(candidate_ngrams), len(indexed_alias.char_ngrams)),
                1,
            )
            overlap_ratio = shared_count / overlap_denominator

            if overlap_ratio < MIN_NGRAM_OVERLAP_RATIO:
                continue

            blocked.append((indexed_alias, shared_count, overlap_ratio))

        blocked.sort(
            key=lambda item: (
                item[1],
                item[2],
                item[0].alias.priority,
            ),
            reverse=True,
        )

        return blocked[:MAX_BLOCK_CANDIDATES]

    def _to_match_candidate(
        self,
        accepted: FuzzyScoredSuggestion,
        decision_reason: str,
        top_suggestions: list[FuzzyScoredSuggestion],
    ) -> MatchCandidate:
        """Convert an accepted RapidFuzz suggestion into a MatchCandidate."""
        alias = accepted.alias
        candidate = accepted.candidate

        return MatchCandidate(
            surface=candidate.text,
            normalized_surface=candidate.text,
            start=candidate.start,
            end=candidate.end,
            target_id=alias.target_id,
            target_type=alias.target_type,
            alias_raw=alias.alias_raw,
            alias_category=alias.alias_category,
            confidence=accepted.score,
            priority=alias.priority,
            method="ngram_rapidfuzz",
            safe_for_exact_match=alias.safe_for_exact_match,
            requires_context=alias.requires_context,
            metadata={
                **alias.metadata,
                **accepted.metadata,
                "fuzzy_reason": decision_reason,
                "fuzzy_score": accepted.score,
                "suggested_alias_normalized": accepted.suggested_alias_normalized,
                "candidate_token_count": candidate.token_count,
                "top_suggestions": [
                    {
                        "target_id": item.alias.target_id,
                        "alias_raw": item.alias.alias_raw,
                        "alias_normalized": item.alias.alias_normalized,
                        "score": item.score,
                        "rapidfuzz_token_sort_score": item.metadata.get(
                            "rapidfuzz_token_sort_score"
                        ),
                        "rapidfuzz_token_set_score": item.metadata.get(
                            "rapidfuzz_token_set_score"
                        ),
                        "ngram_overlap_ratio": item.metadata.get(
                            "ngram_overlap_ratio"
                        ),
                    }
                    for item in top_suggestions
                ],
            },
        )


def _should_index_alias(alias: Alias) -> bool:
    """Return True if an alias should be indexed for RapidFuzz fallback."""
    if alias.target_type not in TARGET_TYPES:
        return False

    if alias.alias_category != AliasCategory.CLINICAL:
        return False

    if not alias.safe_for_exact_match:
        return False

    if alias.metadata.get("policy_reason") not in ALLOWED_POLICY_REASONS:
        return False

    if not alias.alias_normalized:
        return False

    return True


def _char_ngrams(text: str) -> set[str]:
    """Return character n-grams used for blocking.

    Non-alphanumeric characters are removed so that token order and spacing have
    less influence on candidate retrieval.
    """
    compact = re.sub(r"[^a-z0-9]+", "", text.lower())

    if not compact:
        return set()

    if len(compact) <= CHAR_NGRAM_SIZE:
        return {compact}

    return {
        compact[index : index + CHAR_NGRAM_SIZE]
        for index in range(len(compact) - CHAR_NGRAM_SIZE + 1)
    }


def _pseudo_edit_distance(
    candidate_text: str,
    alias_text: str,
    score: float,
) -> int:
    """Create an edit-distance-like value for shared acceptance sorting."""
    candidate_length = len(candidate_text.replace(" ", ""))
    alias_length = len(alias_text.replace(" ", ""))
    denominator = max(candidate_length, alias_length, 1)

    return max(0, round((1.0 - score) * denominator))


def _informative_token_dice(candidate_text: str, alias_text: str) -> float:
    """Return token Dice similarity after removing stopwords.

    This complements RapidFuzz token_set_ratio. token_set_ratio is useful for
    reordered or partially missing tokens, but it can be too permissive when the
    candidate includes unrelated extra tokens.
    """
    candidate_tokens = _informative_tokens(candidate_text)
    alias_tokens = _informative_tokens(alias_text)

    if not candidate_tokens or not alias_tokens:
        return 1.0

    shared = len(candidate_tokens & alias_tokens)
    denominator = len(candidate_tokens) + len(alias_tokens)

    if denominator == 0:
        return 1.0

    return (2 * shared) / denominator


def _informative_tokens(text: str) -> set[str]:
    """Return informative normalized tokens for token-level validation."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())

    return {
        token
        for token in tokens
        if token not in DEFAULT_STOPWORDS
        and len(token) >= 2
    }


def _candidate_cache_key(text: str) -> str:
    """Return a stable cache key for a normalized RapidFuzz candidate."""
    return " ".join(text.lower().strip().split())