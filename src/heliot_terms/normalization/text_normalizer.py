from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from unidecode import unidecode


_DASH_CHARS = {
    "\u2010",  # hyphen
    "\u2011",  # non-breaking hyphen
    "\u2012",  # figure dash
    "\u2013",  # en dash
    "\u2014",  # em dash
    "\u2212",  # minus sign
}

_APOSTROPHE_CHARS = {
    "\u2018",
    "\u2019",
    "\u201b",
    "\u2032",
    "`",
    "´",
}

_WHITESPACE_RE = re.compile(r"\s+")
_E_NUMBER_RE = re.compile(r"\be\s+(\d{3,4})\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class TextNormalizerConfig:
    """Configuration for text normalization."""

    lowercase: bool = True
    strip_accents: bool = True
    normalize_unicode: bool = True
    normalize_e_numbers: bool = True
    keep_ampersand: bool = False
    keep_hash: bool = False


class TextNormalizer:
    """Normalize aliases, drug names, and clinical notes.

    The goal is to make equivalent textual variants comparable. For example,
    ``"E 171"``, ``"e171"``, and ``"(E171)"`` should become compatible forms.
    """

    def __init__(self, config: TextNormalizerConfig | None = None) -> None:
        self.config = config or TextNormalizerConfig()

    def normalize(self, text: str) -> str:
        """Return a normalized representation of the input text.

        Parameters
        ----------
        text:
            Raw text to normalize.

        Returns
        -------
        str
            Normalized text suitable for dictionary lookup or exact matching.
        """
        if text is None:
            return ""

        normalized = str(text).strip()
        if not normalized:
            return ""

        if self.config.normalize_unicode:
            normalized = unicodedata.normalize("NFKC", normalized)

        normalized = self._normalize_dashes(normalized)
        normalized = self._normalize_apostrophes(normalized)

        if self.config.strip_accents:
            normalized = unidecode(normalized)

        if self.config.lowercase:
            normalized = normalized.lower()

        if self.config.normalize_e_numbers:
            normalized = self._normalize_e_numbers(normalized)

        normalized = self._normalize_punctuation(normalized)
        normalized = self._collapse_whitespace(normalized)

        return normalized

    def normalize_for_id(self, text: str) -> str:
        """Normalize text for use inside stable IDs.

        Example
        -------
        ``"Lattosio monoidrato"`` becomes ``"lattosio_monoidrato"``.
        """
        normalized = self.normalize(text)
        normalized = normalized.replace(" ", "_")
        normalized = re.sub(r"[^a-z0-9_]+", "", normalized)
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        return normalized

    def _normalize_dashes(self, text: str) -> str:
        """Replace Unicode dash variants with a plain hyphen."""
        for char in _DASH_CHARS:
            text = text.replace(char, "-")
        return text

    def _normalize_apostrophes(self, text: str) -> str:
        """Replace apostrophe variants with a standard apostrophe."""
        for char in _APOSTROPHE_CHARS:
            text = text.replace(char, "'")
        return text

    def _normalize_e_numbers(self, text: str) -> str:
        """Normalize additive codes such as 'E 171' to 'e171'."""
        return _E_NUMBER_RE.sub(r"e\1", text)

    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation while preserving meaningful token boundaries.

        We intentionally turn most punctuation into spaces instead of deleting it.
        This avoids joining unrelated fragments and keeps matching predictable.
        """
        if not self.config.keep_ampersand:
            text = text.replace("&", " ")

        if not self.config.keep_hash:
            text = text.replace("#", " ")

        # Parentheses usually add formulation details but should not block matches.
        text = text.replace("(", " ").replace(")", " ")

        # Separators commonly found in drug names and chemical aliases.
        text = text.replace("/", " ")
        text = text.replace("\\", " ")
        text = text.replace(",", " ")
        text = text.replace(";", " ")
        text = text.replace(":", " ")

        # Keep plus because combinations like "40+5MG" can be meaningful in product names.
        # For clinical alias matching, plus-separated terms still remain visible as tokens.
        text = re.sub(r"[^\w\s+\-'.]", " ", text)

        return text

    def _collapse_whitespace(self, text: str) -> str:
        """Collapse repeated whitespace and trim the result."""
        return _WHITESPACE_RE.sub(" ", text).strip()