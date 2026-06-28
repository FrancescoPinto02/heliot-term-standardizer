from __future__ import annotations

from transformers import pipeline

from heliot_terms.fallback.semantic.models import SemanticTextCandidate


DEFAULT_NER_MODEL = "expertai/PharmaER.IT-full_xlm-roberta-base"
ALLOWED_NER_LABELS = frozenset({"DRUG"})

MIN_CANDIDATE_CHARS = 3
DEVICE = -1


class NerSemanticCandidateExtractor:
    """Extract candidate drug spans using a pharmaceutical Italian NER model."""

    def __init__(
        self,
        model_name: str = DEFAULT_NER_MODEL,
        allowed_labels: frozenset[str] = ALLOWED_NER_LABELS,
        device: int = DEVICE,
    ) -> None:
        self.model_name = model_name
        self.allowed_labels = allowed_labels
        self.ner = pipeline(
            task="token-classification",
            model=model_name,
            tokenizer=model_name,
            aggregation_strategy="simple",
            device=device,
        )

    def extract(
        self,
        text: str,
        protected_spans: list[tuple[int, int]] | None = None,
    ) -> list[SemanticTextCandidate]:
        """Extract semantic candidates outside protected spans."""
        if not text:
            return []

        protected_spans = protected_spans or []
        raw_entities = self.ner(text)

        candidates: list[SemanticTextCandidate] = []

        for entity in raw_entities:
            label = self._entity_label(entity)

            if label not in self.allowed_labels:
                continue

            start = int(entity["start"])
            end = int(entity["end"])

            start, end = self._trim_span(text, start, end)

            if start >= end:
                continue

            if self._overlaps_protected_span(start, end, protected_spans):
                continue

            surface = text[start:end].strip()

            if len(surface.replace(" ", "")) < MIN_CANDIDATE_CHARS:
                continue

            candidates.append(
                SemanticTextCandidate(
                    text=surface,
                    start=start,
                    end=end,
                    label=label,
                    score=float(entity["score"]) if "score" in entity else None,
                    metadata={
                        "ner_model": self.model_name,
                        "raw_label": self._raw_entity_label(entity),
                    },
                )
            )

        return self._deduplicate(candidates)

    def _entity_label(self, entity: dict) -> str:
        """Return a normalized NER label."""
        label = self._raw_entity_label(entity)
        label = label.replace("B-", "").replace("I-", "")
        return label.upper()

    def _raw_entity_label(self, entity: dict) -> str:
        """Return the raw label from a transformers NER entity."""
        return str(entity.get("entity_group") or entity.get("entity") or "")

    def _trim_span(self, text: str, start: int, end: int) -> tuple[int, int]:
        """Trim punctuation and whitespace at span boundaries."""

        start = max(0, min(start, len(text)))
        end = max(0, min(end, len(text)))

        while start < end and text[start].isspace():
            start += 1

        while start < end and text[end - 1].isspace():
            end -= 1

        while start < end and text[start] in ".,;:!?()[]{}":
            start += 1

        while start < end and text[end - 1] in ".,;:!?()[]{}":
            end -= 1

        return start, end

    def _overlaps_protected_span(
        self,
        start: int,
        end: int,
        protected_spans: list[tuple[int, int]],
    ) -> bool:
        """Return True if the candidate overlaps a protected span."""
        return any(start < protected_end and protected_start < end for protected_start, protected_end in protected_spans)

    def _deduplicate(
        self,
        candidates: list[SemanticTextCandidate],
    ) -> list[SemanticTextCandidate]:
        """Deduplicate candidates by span and text."""
        seen: set[tuple[int, int, str]] = set()
        deduplicated: list[SemanticTextCandidate] = []

        for candidate in candidates:
            key = (candidate.start, candidate.end, candidate.text)
            if key in seen:
                continue

            seen.add(key)
            deduplicated.append(candidate)

        return deduplicated