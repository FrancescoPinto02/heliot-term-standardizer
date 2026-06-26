"""Candidate extraction for fuzzy matching.

The extractor works only on text fragments not already consumed by exact
matching. It generates filtered token n-grams that can be sent to a fuzzy lookup
engine such as SymSpell.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from heliot_terms.fallback.models import FuzzyTextCandidate


_TOKEN_RE = re.compile(
    r"[a-z0-9]+(?:[+\-'][a-z0-9]+)*",
    flags=re.IGNORECASE,
)

_DEFAULT_STOPWORDS = frozenset(
    {
        "a",
        "ad",
        "al",
        "allo",
        "alla",
        "ai",
        "agli",
        "alle",
        "con",
        "da",
        "dal",
        "dalla",
        "del",
        "dello",
        "della",
        "dei",
        "degli",
        "delle",
        "di",
        "e",
        "ed",
        "il",
        "lo",
        "la",
        "i",
        "gli",
        "le",
        "in",
        "nel",
        "nella",
        "nei",
        "negli",
        "nelle",
        "o",
        "od",
        "per",
        "su",
        "sul",
        "sulla",
        "tra",
        "fra",
        "un",
        "uno",
        "una",
        "paziente",
        "pz",
        "allergia",
        "allergico",
        "allergica",
        "intolleranza",
        "intollerante",
        "reazione",
        "rash",
        "dopo",
        "assunzione",
        "terapia",
        "farmaco",
        "farmaci",
        "compresse",
        "compressa",
        "capsule",
        "capsula",
        "mg",
        "ml",
        "g",
    }
)


@dataclass(frozen=True)
class CandidateExtractorConfig:
    """Configuration for fuzzy candidate extraction."""

    max_ngram_tokens: int = 4
    min_token_chars: int = 3
    min_candidate_chars: int = 6
    stopwords: frozenset[str] = field(default_factory=lambda: _DEFAULT_STOPWORDS)


class ResidualCandidateExtractor:
    """Extract fuzzy candidates from spans not consumed by exact matching."""

    def __init__(self, config: CandidateExtractorConfig | None = None) -> None:
        self.config = config or CandidateExtractorConfig()

    def extract(
        self,
        text: str,
        protected_spans: list[tuple[int, int]] | None = None,
    ) -> list[FuzzyTextCandidate]:
        """Extract candidate n-grams from unprotected text regions."""
        if not text:
            return []

        protected_spans = protected_spans or []
        residual_segments = self._residual_segments(text, protected_spans)

        candidates: list[FuzzyTextCandidate] = []

        for segment_start, segment_end in residual_segments:
            tokens = self._tokenize(text, segment_start, segment_end)

            for token_index in range(len(tokens)):
                max_end = min(
                    len(tokens),
                    token_index + self.config.max_ngram_tokens,
                )

                for ngram_end_index in range(token_index + 1, max_end + 1):
                    ngram_tokens = tokens[token_index:ngram_end_index]
                    candidate = self._candidate_from_tokens(text, ngram_tokens)

                    if candidate and self._is_interesting_candidate(candidate):
                        candidates.append(candidate)

        return self._deduplicate(candidates)

    def _residual_segments(
        self,
        text: str,
        protected_spans: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Return text spans not covered by protected spans."""
        if not protected_spans:
            return [(0, len(text))]

        normalized_spans = sorted(
            (max(0, start), min(len(text), end))
            for start, end in protected_spans
            if start < end
        )

        segments: list[tuple[int, int]] = []
        cursor = 0

        for start, end in normalized_spans:
            if cursor < start:
                segments.append((cursor, start))
            cursor = max(cursor, end)

        if cursor < len(text):
            segments.append((cursor, len(text)))

        return segments

    def _tokenize(
        self,
        text: str,
        segment_start: int,
        segment_end: int,
    ) -> list[tuple[str, int, int]]:
        """Tokenize a segment while preserving global offsets."""
        segment = text[segment_start:segment_end]
        tokens: list[tuple[str, int, int]] = []

        for match in _TOKEN_RE.finditer(segment):
            token = match.group(0)
            start = segment_start + match.start()
            end = segment_start + match.end()
            tokens.append((token, start, end))

        return tokens

    def _candidate_from_tokens(
        self,
        text: str,
        tokens: list[tuple[str, int, int]],
    ) -> FuzzyTextCandidate | None:
        """Create a candidate from a token window."""
        if not tokens:
            return None

        start = tokens[0][1]
        end = tokens[-1][2]
        candidate_text = text[start:end].strip()

        if not candidate_text:
            return None

        return FuzzyTextCandidate(
            text=candidate_text,
            start=start,
            end=end,
            token_count=len(tokens),
            metadata={
                "tokens": [token for token, _, _ in tokens],
            },
        )

    def _is_interesting_candidate(self, candidate: FuzzyTextCandidate) -> bool:
        """Return True if a candidate is worth sending to fuzzy lookup."""
        compact = candidate.text.replace(" ", "")

        if len(compact) < self.config.min_candidate_chars:
            return False

        tokens = candidate.metadata.get("tokens", [])

        if not tokens:
            return False

        if all(token in self.config.stopwords for token in tokens):
            return False

        informative_tokens = [
            token
            for token in tokens
            if token not in self.config.stopwords
            and len(token) >= self.config.min_token_chars
            and not token.isnumeric()
        ]

        if not informative_tokens:
            return False

        if candidate.text.isnumeric():
            return False

        return True

    def _deduplicate(
        self,
        candidates: list[FuzzyTextCandidate],
    ) -> list[FuzzyTextCandidate]:
        """Remove duplicate candidate spans while preserving order."""
        seen: set[tuple[int, int, str]] = set()
        deduplicated: list[FuzzyTextCandidate] = []

        for candidate in candidates:
            key = (candidate.start, candidate.end, candidate.text)
            if key in seen:
                continue

            seen.add(key)
            deduplicated.append(candidate)

        return deduplicated