from __future__ import annotations

from dataclasses import dataclass

from heliot_terms.domain.enums import BrandStatus, EntityType, TargetType
from heliot_terms.domain.models import DrugBrand, DrugProduct, IngredientConcept
from heliot_terms.matching.base import BaseMatcher
from heliot_terms.normalization.text_normalizer import TextNormalizer, NormalizedText
from heliot_terms.pipeline.models import (
    NoteOutputPolicy,
    OutputLanguage,
    StandardizationResult,
    StandardizedConcept,
    StandardizedMention,
)
from heliot_terms.resolution.models import ResolvedMatch
from heliot_terms.resources.knowledge_base_repository import KnowledgeBaseRepository
from heliot_terms.resolution.overlap_resolver import OverlapResolver
from heliot_terms.fallback.base import BaseFallbackMatcher


@dataclass(frozen=True)
class StandardizationPipelineConfig:
    """Configuration for the standardization pipeline."""

    output_language: OutputLanguage = "en"
    note_output_policy: NoteOutputPolicy = "annotate"

    ambiguous_brand_policy: str = "structured_only"
    include_excipients_for_product_mentions: bool = False


class StandardizationPipeline:
    """Run exact terminology extraction and standardization on clinical notes."""

    def __init__(
        self,
        repository: KnowledgeBaseRepository,
        matcher: BaseMatcher,
        resolver: OverlapResolver,
        normalizer: TextNormalizer | None = None,
        config: StandardizationPipelineConfig | None = None,
        fallback_matchers: list[BaseFallbackMatcher] | None = None,
    ) -> None:
        self.repository = repository
        self.matcher = matcher
        self.resolver = resolver
        self.normalizer = normalizer or TextNormalizer()
        self.config = config or StandardizationPipelineConfig()
        self.fallback_matchers = fallback_matchers or []

    def standardize(self, text: str) -> StandardizationResult:
        """Extract and standardize terminology mentions from a clinical note."""
        normalized = self.normalizer.normalize_with_mapping(text)

        exact_candidates = self.matcher.match(normalized.text)
        exact_resolved_matches = self.resolver.resolve(exact_candidates)

        protected_spans = [(match.start, match.end) for match in exact_resolved_matches]

        fallback_candidates = self._run_fallback_matchers(
            text=normalized.text,
            protected_spans=protected_spans,
        )
        fallback_resolved_matches = self.resolver.resolve(fallback_candidates)

        resolved_matches = self._merge_exact_and_fallback_matches(
            exact_matches=exact_resolved_matches,
            fallback_matches=fallback_resolved_matches,
        )

        mentions: list[StandardizedMention] = []
        ambiguous: list[StandardizedMention] = []

        for resolved_match in resolved_matches:
            mention = self._enrich_match(resolved_match, normalized, text)

            if mention.status == "ambiguous":
                ambiguous.append(mention)
            else:
                mentions.append(mention)

        standardized_text = self._build_standardized_text(
            original_text=text,
            mentions=mentions,
            policy=self.config.note_output_policy,
        )

        return StandardizationResult(
            original_text=text,
            normalized_text=normalized.text,
            standardized_text=standardized_text,
            output_language=self.config.output_language,
            note_output_policy=self.config.note_output_policy,
            matches=mentions,
            ambiguous=ambiguous,
            metadata={
                "num_exact_candidates": len(exact_candidates),
                "num_exact_resolved_matches": len(exact_resolved_matches),
                "num_fallback_candidates": len(fallback_candidates),
                "num_fallback_resolved_matches": len(fallback_resolved_matches),
                "num_resolved_matches": len(resolved_matches),
                "num_matches": len(mentions),
                "num_ambiguous": len(ambiguous),
            },
        )

    def _enrich_match(
        self,
        match: ResolvedMatch,
        normalized: NormalizedText,
        original_text: str,
    ) -> StandardizedMention:
        """Attach canonical concept/product/brand information to a resolved match."""
        if match.target_type == TargetType.INGREDIENT:
            return self._enrich_ingredient_match(match, normalized, original_text)

        if match.target_type == TargetType.DRUG_PRODUCT:
            return self._enrich_product_match(match, normalized, original_text)

        if match.target_type == TargetType.DRUG_BRAND:
            return self._enrich_brand_match(match, normalized, original_text)

        raise ValueError(f"Unsupported target type: {match.target_type}")

    def _enrich_ingredient_match(
        self,
        match: ResolvedMatch,
        normalized: NormalizedText,
        original_text: str,
    ) -> StandardizedMention:
        """Enrich an ingredient match with its canonical concept."""
        concept = self.repository.get_concept(match.target_id)

        concepts = []
        if concept:
            concepts.append(self._to_standardized_concept(concept))

        display_names = self._display_names(concepts)

        return StandardizedMention(
            **self._span_payload(match, normalized, original_text),
            target_id=match.target_id,
            target_type=match.target_type,
            matched_alias=match.alias_raw,
            method=match.method,
            confidence=match.confidence,
            concepts=concepts,
            annotation_text="; ".join(display_names) if display_names else None,
            replacement_text="; ".join(display_names) if display_names else None,
            status="resolved",
            metadata=match.metadata,
        )

    def _enrich_product_match(
        self,
        match: ResolvedMatch,
        normalized: NormalizedText,
        original_text: str,
    ) -> StandardizedMention:
        """Enrich a drug product match using its exact product composition."""
        product = self.repository.get_product(match.target_id)

        if not product:
            return self._unresolved_mention(match, reason="missing_product")

        concepts = self._concepts_from_product(product)
        display_names = self._display_names(concepts)

        return StandardizedMention(
            **self._span_payload(match, normalized, original_text),
            target_id=match.target_id,
            target_type=match.target_type,
            matched_alias=match.alias_raw,
            method=match.method,
            confidence=match.confidence,
            concepts=concepts,
            annotation_text="; ".join(display_names) if display_names else None,
            replacement_text="; ".join(display_names) if display_names else None,
            status="resolved",
            metadata={
                **match.metadata,
                "drug_name": product.drug_name,
                "drug_code": product.drug_code,
            },
        )

    def _enrich_brand_match(
        self,
        match: ResolvedMatch,
        normalized: NormalizedText,
        original_text: str,
    ) -> StandardizedMention:
        """Enrich a drug brand match and handle ambiguous brands conservatively."""
        brand = self.repository.get_brand(match.target_id)

        if not brand:
            return self._unresolved_mention(match, reason="missing_brand")

        if brand.brand_status == BrandStatus.AMBIGUOUS_BRAND:
            return StandardizedMention(
                **self._span_payload(match, normalized, original_text),
                target_id=match.target_id,
                target_type=match.target_type,
                matched_alias=match.alias_raw,
                method=match.method,
                confidence=match.confidence,
                concepts=[],
                annotation_text=None,
                replacement_text=None,
                status="ambiguous",
                metadata={
                    **match.metadata,
                    "brand_name": brand.brand_name,
                    "brand_status": brand.brand_status.value,
                    "active_ingredient_signatures": brand.active_ingredient_signatures,
                    "policy": self.config.ambiguous_brand_policy,
                },
            )

        concepts = self._concepts_from_brand(brand)
        display_names = self._display_names(concepts)

        return StandardizedMention(
            **self._span_payload(match, normalized, original_text),
            target_id=match.target_id,
            target_type=match.target_type,
            matched_alias=match.alias_raw,
            method=match.method,
            confidence=match.confidence,
            concepts=concepts,
            annotation_text="; ".join(display_names) if display_names else None,
            replacement_text="; ".join(display_names) if display_names else None,
            status="resolved",
            metadata={
                **match.metadata,
                "brand_name": brand.brand_name,
                "brand_status": brand.brand_status.value,
            },
        )

    def _concepts_from_product(self, product: DrugProduct) -> list[StandardizedConcept]:
        """Return standardized concepts associated with a specific product."""
        concepts: list[StandardizedConcept] = []

        for concept_id in product.active_ingredient_ids:
            concept = self.repository.get_concept(concept_id)
            if concept:
                concepts.append(
                    self._to_standardized_concept(concept, role="active_ingredient")
                )

        if self.config.include_excipients_for_product_mentions:
            for concept_id in product.excipient_ids:
                concept = self.repository.get_concept(concept_id)
                if concept:
                    concepts.append(
                        self._to_standardized_concept(concept, role="excipient")
                    )

        return concepts

    def _concepts_from_brand(self, brand: DrugBrand) -> list[StandardizedConcept]:
        """Return active ingredient concepts associated with a non-ambiguous brand."""
        concepts: list[StandardizedConcept] = []

        for concept_id in brand.active_ingredient_ids:
            concept = self.repository.get_concept(concept_id)
            if concept:
                concepts.append(
                    self._to_standardized_concept(concept, role="active_ingredient")
                )

        return concepts

    def _to_standardized_concept(
        self,
        concept: IngredientConcept,
        role: str | None = None,
    ) -> StandardizedConcept:
        """Convert a KB concept into a pipeline output concept."""
        return StandardizedConcept(
            concept_id=concept.concept_id,
            entity_type=EntityType(concept.entity_type),
            canonical_it=concept.canonical_it,
            canonical_en=concept.canonical_en,
            role=role,
        )

    def _display_names(self, concepts: list[StandardizedConcept]) -> list[str]:
        """Return unique display names preserving order."""
        seen: set[str] = set()
        names: list[str] = []

        for concept in concepts:
            name = concept.display_name(self.config.output_language)
            if name not in seen:
                seen.add(name)
                names.append(name)

        return names

    def _build_standardized_text(
        self,
        original_text: str,
        mentions: list[StandardizedMention],
        policy: NoteOutputPolicy,
    ) -> str:
        """Build the standardized text according to the configured policy.

        In replace mode, only direct ingredient mentions are replaced. Drug products
        and drug brands are always rendered as annotations because a commercial drug
        mention is not clinically equivalent to a direct ingredient mention.
        """
        if policy == "structured_only":
            return original_text

        if not mentions:
            return original_text

        output_parts: list[str] = []
        cursor = 0

        for mention in sorted(mentions, key=lambda item: item.start):
            output_parts.append(original_text[cursor : mention.start])
            output_parts.append(self._render_mention(original_text, mention, policy))
            cursor = mention.end

        output_parts.append(original_text[cursor:])
        return "".join(output_parts)

    def _render_mention(
        self,
        original_text: str,
        mention: StandardizedMention,
        policy: NoteOutputPolicy,
    ) -> str:
        """Render a single mention according to the output policy."""
        original_surface = original_text[mention.start : mention.end]

        if policy == "replace" and mention.target_type == TargetType.INGREDIENT:
            return mention.replacement_text or original_surface

        if mention.annotation_text:
            return f"{original_surface} [{mention.annotation_text}]"

        return original_surface

    def _unresolved_mention(
        self,
        match: ResolvedMatch,
        normalized: NormalizedText,
        original_text: str,
        reason: str,
    ) -> StandardizedMention:
        """Create a structured unresolved mention for missing KB records."""
        return StandardizedMention(
            **self._span_payload(match, normalized, original_text),
            target_id=match.target_id,
            target_type=match.target_type,
            matched_alias=match.alias_raw,
            method=match.method,
            confidence=match.confidence,
            concepts=[],
            annotation_text=None,
            replacement_text=None,
            status="unresolved",
            metadata={**match.metadata, "reason": reason},
        )

    def _span_payload(
        self,
        match: ResolvedMatch,
        normalized: NormalizedText,
        original_text: str,
    ) -> dict:
        """Build common span fields using original-text offsets."""
        original_start, original_end = normalized.original_span(match.start, match.end)

        return {
            "surface": original_text[original_start:original_end],
            "normalized_surface": match.normalized_surface,
            "start": original_start,
            "end": original_end,
            "normalized_start": match.start,
            "normalized_end": match.end,
        }

    def _run_fallback_matchers(
        self,
        text: str,
        protected_spans: list[tuple[int, int]],
    ) -> list:
        """Run configured fallback matchers on unprotected text spans."""
        fallback_candidates = []

        for fallback_matcher in self.fallback_matchers:
            fallback_candidates.extend(
                fallback_matcher.match(
                    text=text,
                    protected_spans=protected_spans,
                )
            )

        return fallback_candidates

    def _merge_exact_and_fallback_matches(
        self,
        exact_matches: list[ResolvedMatch],
        fallback_matches: list[ResolvedMatch],
    ) -> list[ResolvedMatch]:
        """Merge exact and fallback matches, giving exact matches priority."""
        selected = list(exact_matches)

        for fallback_match in fallback_matches:
            if self._overlaps_any_resolved(fallback_match, selected):
                continue
            selected.append(fallback_match)

        return sorted(selected, key=lambda item: (item.start, item.end))

    def _overlaps_any_resolved(
        self,
        match: ResolvedMatch,
        selected: list[ResolvedMatch],
    ) -> bool:
        """Return True if a resolved match overlaps any selected match."""
        return any(self._resolved_spans_overlap(match, other) for other in selected)

    def _resolved_spans_overlap(
        self,
        left: ResolvedMatch,
        right: ResolvedMatch,
    ) -> bool:
        """Return True if two resolved spans overlap."""
        return left.start < right.end and right.start < left.end