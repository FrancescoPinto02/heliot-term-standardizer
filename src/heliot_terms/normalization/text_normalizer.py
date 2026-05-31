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


@dataclass(frozen=True)
class NormalizedText:
    """Normalized text with mapping back to original character offsets.

    ``normalized_to_original[i]`` contains the character index in the original
    text that generated ``text[i]``.
    """

    text: str
    normalized_to_original: list[int]

    def original_span(self, normalized_start: int, normalized_end: int) -> tuple[int, int]:
        """Convert a half-open normalized span into an original-text span."""
        if normalized_start < 0 or normalized_end > len(self.text) or normalized_start >= normalized_end:
            raise ValueError("Invalid normalized span.")

        original_start = self.normalized_to_original[normalized_start]
        original_end = self.normalized_to_original[normalized_end - 1] + 1

        return original_start, original_end

class TextNormalizer:
    """Normalize aliases, drug names, and clinical notes.

    The goal is to make equivalent textual variants comparable. For example,
    ``"E 171"``, ``"e171"``, and ``"(E171)"`` should become compatible forms.
    """

    def __init__(self, config: TextNormalizerConfig | None = None) -> None:
        self.config = config or TextNormalizerConfig()

    def normalize(self, text: str) -> str:
        """Return a normalized representation of the input text.

        """
        return self.normalize_with_mapping(text).text

    def normalize_with_mapping(self, text: str) -> NormalizedText:
        """Normalize text while preserving a mapping to original offsets.

        This method is used by the pipeline. The matcher still receives normalized
        text, but final annotations/replacements are applied to the original note.
        """
        if text is None:
            return NormalizedText(text="", normalized_to_original=[])

        original = str(text)
        chars: list[str] = []
        mapping: list[int] = []

        for original_index, char in enumerate(original):
            normalized_chars = self._normalize_char(char)

            for normalized_char in normalized_chars:
                chars.append(normalized_char)
                mapping.append(original_index)

        normalized, mapping = self._normalize_e_numbers_with_mapping(chars, mapping)
        normalized, mapping = self._collapse_whitespace_with_mapping(
            normalized, mapping
        )

        return NormalizedText(text=normalized, normalized_to_original=mapping)
    

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

    def _normalize_char(self, char: str) -> str:
        """Normalize a single character while preserving local mapping."""
        if self.config.normalize_unicode:
            char = unicodedata.normalize("NFKC", char)

        output_chars: list[str] = []

        for expanded_char in char:
            if expanded_char in _DASH_CHARS:
                expanded_char = "-"

            if expanded_char in _APOSTROPHE_CHARS:
                expanded_char = "'"

            if self.config.strip_accents:
                expanded = unidecode(expanded_char)
            else:
                expanded = expanded_char

            if self.config.lowercase:
                expanded = expanded.lower()

            for final_char in expanded:
                output_chars.append(self._normalize_punctuation_char(final_char))

        return "".join(output_chars)

    def _normalize_punctuation_char(self, char: str) -> str:
        """Normalize punctuation at character level."""
        if char == "&" and not self.config.keep_ampersand:
            return " "

        if char == "#" and not self.config.keep_hash:
            return " "

        if char in {"(", ")", "/", "\\", ",", ";", ":", "*"}:
            return " "

        if re.match(r"[^\w\s+\-'.]", char):
            return " "

        return char

    def _normalize_e_numbers_with_mapping(
        self,
        chars: list[str],
        mapping: list[int],
    ) -> tuple[list[str], list[int]]:
        """Normalize E-number patterns while preserving offset mapping."""
        if not self.config.normalize_e_numbers:
            return chars, mapping

        text = "".join(chars)
        output_chars: list[str] = []
        output_mapping: list[int] = []

        cursor = 0

        for match in _E_NUMBER_RE.finditer(text):
            start, end = match.span()
            digits_start, digits_end = match.span(1)

            output_chars.extend(chars[cursor:start])
            output_mapping.extend(mapping[cursor:start])

            output_chars.append("e")
            output_mapping.append(mapping[start])

            output_chars.extend(chars[digits_start:digits_end])
            output_mapping.extend(mapping[digits_start:digits_end])

            cursor = end

        output_chars.extend(chars[cursor:])
        output_mapping.extend(mapping[cursor:])

        return output_chars, output_mapping

    def _collapse_whitespace_with_mapping(
        self,
        chars: list[str],
        mapping: list[int],
    ) -> tuple[str, list[int]]:
        """Collapse whitespace and keep one original offset per emitted character."""
        output_chars: list[str] = []
        output_mapping: list[int] = []

        previous_was_space = False

        for char, original_index in zip(chars, mapping, strict=True):
            if char.isspace():
                if not previous_was_space:
                    output_chars.append(" ")
                    output_mapping.append(original_index)
                    previous_was_space = True
                continue

            output_chars.append(char)
            output_mapping.append(original_index)
            previous_was_space = False

        while output_chars and output_chars[0] == " ":
            output_chars.pop(0)
            output_mapping.pop(0)

        while output_chars and output_chars[-1] == " ":
            output_chars.pop()
            output_mapping.pop()

        return "".join(output_chars), output_mapping