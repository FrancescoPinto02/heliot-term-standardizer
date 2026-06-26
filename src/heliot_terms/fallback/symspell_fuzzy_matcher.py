"""SymSpell-based fuzzy fallback matcher."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from symspellpy import SymSpell, Verbosity

from heliot_terms.domain.enums import AliasCategory, TargetType
from heliot_terms.domain.models import Alias
from heliot_terms.fallback.acceptance import (
    FuzzyAcceptanceConfig,
    FuzzyAcceptancePolicy,
)
from heliot_terms.fallback.base import BaseFallbackMatcher
from heliot_terms.fallback.candidate_extractor import (
    CandidateExtractorConfig,
    ResidualCandidateExtractor,
)
from heliot_terms.fallback.models import FuzzyScoredSuggestion, FuzzyTextCandidate
from heliot_terms.matching.models import MatchCandidate


@dataclass(frozen=True)
class SymSpellFuzzyMatcherConfig:
    """Configuration for the SymSpell fuzzy matcher."""

    max_dictionary_edit_distance: int = 2
    prefix_length: int = 7
    max_lookup_edit_distance: int = 2

    target_types: tuple[TargetType, ...] = (TargetType.INGREDIENT,)
    allowed_policy_reasons: tuple[str, ...] = ("clinical_alias",)

    candidate_extractor: CandidateExtractorConfig = field(
        default_factory=CandidateExtractorConfig
    )
    acceptance: FuzzyAcceptanceConfig = field(default_factory=FuzzyAcceptanceConfig)


class SymSpellFuzzyMatcher(BaseFallbackMatcher):
    """Fuzzy matcher based on SymSpell.

    This matcher is intentionally conservative:
    - it indexes only selected aliases;
    - it searches only text spans not consumed by exact matching;
    - it returns a match only if the acceptance policy approves it.
    """

    def __init__(
        self,
        sym_spell: SymSpell,
        aliases_by_dictionary_key: dict[str, list[Alias]],
        config: SymSpellFuzzyMatcherConfig | None = None,
        candidate_extractor: ResidualCandidateExtractor | None = None,
        acceptance_policy: FuzzyAcceptancePolicy | None = None,
    ) -> None:
        self.sym_spell = sym_spell
        self.aliases_by_dictionary_key = aliases_by_dictionary_key
        self.config = config or SymSpellFuzzyMatcherConfig()
        self.candidate_extractor = candidate_extractor or ResidualCandidateExtractor(
            self.config.candidate_extractor
        )
        self.acceptance_policy = acceptance_policy or FuzzyAcceptancePolicy(
            self.config.acceptance
        )

    @classmethod
    def from_aliases(
        cls,
        aliases: list[Alias],
        config: SymSpellFuzzyMatcherConfig | None = None,
    ) -> SymSpellFuzzyMatcher:
        """Build a SymSpell fuzzy matcher from processed aliases."""
        config = config or SymSpellFuzzyMatcherConfig()

        sym_spell = SymSpell(
            max_dictionary_edit_distance=config.max_dictionary_edit_distance,
            prefix_length=config.prefix_length,
        )

        aliases_by_dictionary_key: dict[str, list[Alias]] = defaultdict(list)
        dictionary_counts: dict[str, int] = {}

        for alias in aliases:
            if not _should_index_alias(alias, config):
                continue

            dictionary_key = _to_dictionary_key(alias.alias_normalized)
            aliases_by_dictionary_key[dictionary_key].append(alias)
            dictionary_counts[dictionary_key] = max(
                dictionary_counts.get(dictionary_key, 1),
                max(alias.priority, 1),
            )

        for dictionary_key, count in dictionary_counts.items():
            sym_spell.create_dictionary_entry(dictionary_key, count)

        return cls(
            sym_spell=sym_spell,
            aliases_by_dictionary_key=dict(aliases_by_dictionary_key),
            config=config,
        )

    def match(
        self,
        text: str,
        protected_spans: list[tuple[int, int]] | None = None,
    ) -> list[MatchCandidate]:
        """Return accepted fuzzy match candidates."""
        text_candidates = self.candidate_extractor.extract(
            text=text,
            protected_spans=protected_spans or [],
        )

        matches: list[MatchCandidate] = []

        for text_candidate in text_candidates:
            scored_suggestions = self._lookup_candidate(text_candidate)
            decision = self.acceptance_policy.decide(scored_suggestions)

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

    def _lookup_candidate(
        self,
        candidate: FuzzyTextCandidate,
    ) -> list[FuzzyScoredSuggestion]:
        """Lookup and score SymSpell suggestions for one text candidate."""
        candidate_key = _to_dictionary_key(candidate.text)

        suggestions = self.sym_spell.lookup(
            candidate_key,
            Verbosity.ALL,
            max_edit_distance=self.config.max_lookup_edit_distance,
            include_unknown=False,
        )

        scored: list[FuzzyScoredSuggestion] = []

        for suggestion in suggestions:
            aliases = self.aliases_by_dictionary_key.get(suggestion.term, [])

            for alias in aliases:
                scored.append(
                    FuzzyScoredSuggestion(
                        candidate=candidate,
                        alias=alias,
                        suggested_alias_normalized=alias.alias_normalized,
                        dictionary_key=suggestion.term,
                        edit_distance=suggestion.distance,
                        score=_score_from_edit_distance(
                            candidate_text=candidate.text,
                            alias_text=alias.alias_normalized,
                            edit_distance=suggestion.distance,
                        ),
                    )
                )

        return scored

    def _to_match_candidate(
        self,
        accepted: FuzzyScoredSuggestion,
        decision_reason: str,
        top_suggestions: list[FuzzyScoredSuggestion],
    ) -> MatchCandidate:
        """Convert an accepted fuzzy suggestion into a MatchCandidate."""
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
            method="symspell_fuzzy",
            safe_for_exact_match=alias.safe_for_exact_match,
            requires_context=alias.requires_context,
            metadata={
                **alias.metadata,
                "fuzzy_reason": decision_reason,
                "fuzzy_score": accepted.score,
                "edit_distance": accepted.edit_distance,
                "suggested_alias_normalized": accepted.suggested_alias_normalized,
                "dictionary_key": accepted.dictionary_key,
                "candidate_token_count": candidate.token_count,
                "top_suggestions": [
                    {
                        "target_id": item.alias.target_id,
                        "alias_raw": item.alias.alias_raw,
                        "alias_normalized": item.alias.alias_normalized,
                        "score": item.score,
                        "edit_distance": item.edit_distance,
                    }
                    for item in top_suggestions
                ],
            },
        )


def _should_index_alias(
    alias: Alias,
    config: SymSpellFuzzyMatcherConfig,
) -> bool:
    """Return True if an alias should be indexed for fuzzy matching."""
    if alias.target_type not in config.target_types:
        return False

    if alias.alias_category != AliasCategory.CLINICAL:
        return False

    if not alias.safe_for_exact_match:
        return False

    policy_reason = alias.metadata.get("policy_reason")
    if policy_reason not in config.allowed_policy_reasons:
        return False

    if not alias.alias_normalized:
        return False

    return True


def _to_dictionary_key(text: str) -> str:
    """Convert normalized text into a SymSpell dictionary key.

    Spaces are converted to underscores so multi-word aliases can be indexed and
    looked up as a single fuzzy term. Leading and trailing punctuation is removed
    to avoid penalizing candidates at sentence boundaries.
    """
    cleaned = " ".join(text.strip().strip(".,;:!?()[]{}").split())
    return cleaned.replace(" ", "_")


def _score_from_edit_distance(
    candidate_text: str,
    alias_text: str,
    edit_distance: int,
) -> float:
    """Convert edit distance into a normalized similarity score."""
    candidate_length = len(candidate_text.replace(" ", ""))
    alias_length = len(alias_text.replace(" ", ""))
    denominator = max(candidate_length, alias_length, 1)

    return max(0.0, 1.0 - (edit_distance / denominator))