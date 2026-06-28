"""Aho-Corasick exact matcher implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ahocorasick

from heliot_terms.domain.enums import TargetType
from heliot_terms.domain.models import Alias
from heliot_terms.matching.base import BaseMatcher
from heliot_terms.matching.models import MatchCandidate


@dataclass(frozen=True)
class IndexedAlias:
    """Alias metadata stored inside the automaton."""

    alias_raw: str
    alias_normalized: str
    target_id: str
    target_type: TargetType
    alias_category: str
    safe_for_exact_match: bool
    requires_context: bool
    priority: int
    metadata: dict[str, Any]


class AhoCorasickMatcher(BaseMatcher):
    """Exact matcher backed by a pyahocorasick automaton."""

    def __init__(
        self, automaton: ahocorasick.Automaton, is_empty: bool = False
    ) -> None:
        self._automaton = automaton
        self._is_empty = is_empty

    @classmethod
    def from_aliases(
        cls,
        aliases: list[Alias],
        *,
        include_unsafe: bool = False,
    ) -> AhoCorasickMatcher:
        """Build an Aho-Corasick matcher from processed aliases.

        Parameters
        ----------
        aliases:
            Processed aliases loaded from ``aliases.jsonl``.
        include_unsafe:
            If False, only aliases marked as ``safe_for_exact_match`` are
            inserted into the automaton.
        """
        automaton = ahocorasick.Automaton()

        for alias in aliases:
            if not include_unsafe and not alias.safe_for_exact_match:
                continue

            if not alias.alias_normalized:
                continue

            indexed_alias = IndexedAlias(
                alias_raw=alias.alias_raw,
                alias_normalized=alias.alias_normalized,
                target_id=alias.target_id,
                target_type=alias.target_type,
                alias_category=alias.alias_category.value,
                safe_for_exact_match=alias.safe_for_exact_match,
                requires_context=alias.requires_context,
                priority=alias.priority,
                metadata=alias.metadata,
            )

            # Multiple aliases can normalize to the same string. pyahocorasick
            # stores one value per key, so we keep a list of alias payloads.
            if alias.alias_normalized in automaton:
                existing_values = automaton.get(alias.alias_normalized)
                existing_values.append(indexed_alias)
                automaton.add_word(alias.alias_normalized, existing_values)
            else:
                automaton.add_word(alias.alias_normalized, [indexed_alias])

        if len(automaton) == 0:
            return cls(automaton=automaton, is_empty=True)

        automaton.make_automaton()
        return cls(automaton=automaton, is_empty=False)

    def match(self, text: str) -> list[MatchCandidate]:
        """Return exact match candidates found in normalized text."""
        if self._is_empty:
            return []

        matches: list[MatchCandidate] = []

        for end_index, indexed_aliases in self._automaton.iter(text):
            for indexed_alias in indexed_aliases:
                start = end_index - len(indexed_alias.alias_normalized) + 1
                end = end_index + 1

                if not _has_token_boundaries(text, start, end):
                    continue

                surface = text[start:end]

                matches.append(
                    MatchCandidate(
                        surface=surface,
                        normalized_surface=indexed_alias.alias_normalized,
                        start=start,
                        end=end,
                        target_id=indexed_alias.target_id,
                        target_type=indexed_alias.target_type,
                        alias_raw=indexed_alias.alias_raw,
                        alias_category=indexed_alias.alias_category,
                        confidence=1.0,
                        priority=indexed_alias.priority,
                        method="aho_corasick_exact",
                        safe_for_exact_match=indexed_alias.safe_for_exact_match,
                        requires_context=indexed_alias.requires_context,
                        metadata={
                            **indexed_alias.metadata,
                            "priority": indexed_alias.priority,
                        },
                    )
                )

        return matches


def _has_token_boundaries(text: str, start: int, end: int) -> bool:
    """Check conservative token boundaries around a match.

    Without this check, an alias could match inside a longer token.
    We allow boundaries at whitespace and most punctuation characters.
    """
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "

    return _is_boundary(before) and _is_boundary(after)


def _is_boundary(char: str) -> bool:
    """Return True if a character can delimit a terminology match."""
    return not (char.isalnum() or char in {"_", "-"})