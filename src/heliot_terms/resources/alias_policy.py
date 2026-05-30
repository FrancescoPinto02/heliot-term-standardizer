from __future__ import annotations

import re
from dataclasses import dataclass

from heliot_terms.domain.enums import AliasCategory


_CHEMICAL_COMPLEXITY_RE = re.compile(
    r"(\d+[a-z]?[-,])|(\([^)]+\))|(\[[^\]]+\])|([αβγδ])|(\b\d+h\b)",
    flags=re.IGNORECASE,
)

_STEREO_CHEMISTRY_RE = re.compile(
    r"(\(\s*[±+\-]\s*\))|(\([0-9rs*,\s]+\))|(\b[rs]-)|(\b[dl]-)",
    flags=re.IGNORECASE,
)

_CHEMICAL_PUNCTUATION_RE = re.compile(
    r"[-(),\[\];]",
)

_CLINICAL_CODE_RE = re.compile(
    r"\b(fd\s*&?\s*c|d\s*&?\s*c|e\s*\d{3,4})\b",
    flags=re.IGNORECASE,
)

_FORMULA_LIKE_RE = re.compile(r"\b[a-z]{1,3}\d+[a-z0-9()]*\b", flags=re.IGNORECASE)

_ACRONYM_LIKE_RE = re.compile(r"^[A-Z]{2,6}$")

@dataclass(frozen=True)
class AliasPolicyResult:
    """Decision returned by the alias policy."""

    category: AliasCategory
    safe_for_exact_match: bool
    requires_context: bool
    reason: str


class AliasPolicy:
    """Classify aliases for exact matching.

    The policy is intentionally conservative because this module will be used in
    a clinical decision-support context. Ambiguous or overly technical aliases
    are preserved in the KB but excluded from the v0 exact matcher.
    """

    def __init__(
        self,
        min_safe_alias_length: int = 5,
        include_technical_aliases_in_exact_index: bool = False,
    ) -> None:
        self.min_safe_alias_length = min_safe_alias_length
        self.include_technical_aliases_in_exact_index = include_technical_aliases_in_exact_index

    def classify(self, alias_raw: str, alias_normalized: str) -> AliasPolicyResult:
        """Classify an alias and decide whether it is safe for exact matching."""
        raw = alias_raw.strip()
        normalized = alias_normalized.strip()

        if not normalized:
            return AliasPolicyResult(
                category=AliasCategory.UNSAFE,
                safe_for_exact_match=False,
                requires_context=False,
                reason="empty_alias",
            )

        compact = normalized.replace(" ", "")

        if self._looks_like_short_acronym(raw):
            return AliasPolicyResult(
                category=AliasCategory.UNSAFE,
                safe_for_exact_match=False,
                requires_context=True,
                reason="short_acronym_alias",
            )

        if self._looks_like_clinical_code(raw, normalized):
            return AliasPolicyResult(
                category=AliasCategory.CLINICAL,
                safe_for_exact_match=True,
                requires_context=False,
                reason="clinical_code_alias",
            )

        if len(compact) < self.min_safe_alias_length:
            return AliasPolicyResult(
                category=AliasCategory.UNSAFE,
                safe_for_exact_match=False,
                requires_context=True,
                reason="short_alias",
            )

        if self._looks_like_technical_chemical_name(raw, normalized):
            return AliasPolicyResult(
                category=AliasCategory.TECHNICAL,
                safe_for_exact_match=self.include_technical_aliases_in_exact_index,
                requires_context=False,
                reason="technical_chemical_alias",
            )

        return AliasPolicyResult(
            category=AliasCategory.CLINICAL,
            safe_for_exact_match=True,
            requires_context=False,
            reason="clinical_alias",
        )


    def _looks_like_technical_chemical_name(self, raw: str, normalized: str) -> bool:
        """Return True for complex chemical-like aliases.

        The goal is not to detect every chemical name perfectly. We only need to
        exclude aliases that are unlikely to be written in ordinary clinical notes
        and would unnecessarily pollute the exact matcher index.
        """
        if len(normalized) < 18:
            return False

        punctuation_score = len(_CHEMICAL_PUNCTUATION_RE.findall(raw))
        has_stereo_marker = bool(_STEREO_CHEMISTRY_RE.search(raw))
        has_complex_pattern = bool(_CHEMICAL_COMPLEXITY_RE.search(raw))
        has_formula_like_token = bool(_FORMULA_LIKE_RE.search(raw))

        if has_stereo_marker and punctuation_score >= 1:
            return True

        if punctuation_score >= 3 and (has_complex_pattern or has_formula_like_token):
            return True

        if len(normalized) >= 60 and punctuation_score >= 2:
            return True

        return False

    def _looks_like_clinical_code(self, raw: str, normalized: str) -> bool:
        """Return True for clinical colorant/additive codes.

        Examples such as 'F D & C #3' or 'E 171' can plausibly appear in drug
        excipient documentation and should not be treated as technical chemical
        names just because they contain symbols or digits.
        """
        return bool(
            _CLINICAL_CODE_RE.search(raw) or _CLINICAL_CODE_RE.search(normalized)
        )

    def _looks_like_short_acronym(self, raw: str) -> bool:
        """Return True for short acronym-like aliases such as PEG, PVP, SLS, APAP.

        These aliases can be clinically relevant, but they are risky as automatic
        exact matches without contextual validation.
        """
        compact_raw = re.sub(r"[^A-Za-z]", "", raw.strip())
        return bool(_ACRONYM_LIKE_RE.fullmatch(compact_raw))