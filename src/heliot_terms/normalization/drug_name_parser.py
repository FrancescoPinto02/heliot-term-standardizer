from __future__ import annotations

from dataclasses import dataclass

from heliot_terms.normalization.text_normalizer import TextNormalizer


@dataclass(frozen=True)
class ParsedDrugName:
    """Parsed representation of a raw drug name."""

    raw_name: str
    full_name: str
    base_name: str
    normalized_full_name: str
    normalized_base_name: str


class DrugNameParser:
    """Parse compact drug names into full and base names.

    The parser is intentionally conservative. In v0, it only uses the asterisk
    separator when available and does not generate shorter brand variants such
    as 'OLMESARTAN' from 'OLMESARTAN AM TEV'.
    """

    def __init__(self, normalizer: TextNormalizer | None = None) -> None:
        self.normalizer = normalizer or TextNormalizer()

    def parse(self, raw_name: str) -> ParsedDrugName:
        """Parse a raw drug name.

        Parameters
        ----------
        raw_name:
            Drug name as found in the source CSV.

        Returns
        -------
        ParsedDrugName
            Full name and base name, both in raw and normalized form.
        """
        full_name = str(raw_name or "").strip()
        base_name = self.extract_base_name(full_name)

        return ParsedDrugName(
            raw_name=raw_name,
            full_name=full_name,
            base_name=base_name,
            normalized_full_name=self.normalizer.normalize(full_name),
            normalized_base_name=self.normalizer.normalize(base_name),
        )

    def extract_base_name(self, full_name: str) -> str:
        """Extract the base commercial name from a compact drug name.

        The main rule is:

        ``NAME*PACKAGING DOSAGE`` → ``NAME``

        If the asterisk is missing, the full name is returned unchanged.
        """
        full_name = str(full_name or "").strip()
        if not full_name:
            return ""

        if "*" in full_name:
            return full_name.split("*", maxsplit=1)[0].strip()

        return full_name