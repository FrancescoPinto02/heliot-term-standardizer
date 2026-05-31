"""Factory for deterministic matcher implementations."""

from __future__ import annotations

from heliot_terms.domain.models import Alias
from heliot_terms.matching.aho_corasick_matcher import AhoCorasickMatcher
from heliot_terms.matching.base import BaseMatcher


def build_matcher(
    matcher_type: str,
    aliases: list[Alias],
    *,
    include_unsafe: bool = False,
) -> BaseMatcher:
    """Build a matcher implementation from configuration."""
    normalized_type = matcher_type.strip().lower()

    if normalized_type in {"aho_corasick", "ahocorasick", "aho-corasick"}:
        return AhoCorasickMatcher.from_aliases(
            aliases=aliases,
            include_unsafe=include_unsafe,
        )

    raise ValueError(f"Unsupported matcher type: {matcher_type}")