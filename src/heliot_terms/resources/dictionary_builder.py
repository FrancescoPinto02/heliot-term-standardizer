from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from heliot_terms.domain.enums import (
    AliasLanguage,
    AliasSource,
    BrandStatus,
    EntityType,
    IssueSeverity,
    TargetType,
)
from heliot_terms.domain.ids import make_ingredient_concept_id
from heliot_terms.domain.models import (
    Alias,
    BuildIssue,
    DrugBrand,
    DrugProduct,
    IngredientConcept,
    KnowledgeBaseBuildReport,
)
from heliot_terms.normalization.drug_name_parser import DrugNameParser
from heliot_terms.normalization.text_normalizer import TextNormalizer
from heliot_terms.resources.alias_policy import AliasPolicy
from heliot_terms.resources.jsonl import write_json, write_jsonl
from heliot_terms.resources.raw_models import RawDrugRecord, RawSynonymEntry


@dataclass(frozen=True)
class KnowledgeBase:
    """Processed terminology knowledge base."""

    concepts: list[IngredientConcept]
    aliases: list[Alias]
    drug_products: list[DrugProduct]
    drug_brands: list[DrugBrand]
    report: KnowledgeBaseBuildReport


class KnowledgeBaseBuilder:
    """Build terminology resources from raw synonym and drug records."""

    def __init__(
        self,
        normalizer: TextNormalizer | None = None,
        drug_name_parser: DrugNameParser | None = None,
        alias_policy: AliasPolicy | None = None,
    ) -> None:
        self.normalizer = normalizer or TextNormalizer()
        self.drug_name_parser = drug_name_parser or DrugNameParser(self.normalizer)
        self.alias_policy = alias_policy or AliasPolicy()

    def build(
        self,
        synonym_entries: list[RawSynonymEntry],
        drug_records: list[RawDrugRecord],
    ) -> KnowledgeBase:
        """Build a processed knowledge base from raw records."""
        concepts_by_id: dict[str, IngredientConcept] = {}
        aliases: list[Alias] = []
        drug_products: list[DrugProduct] = []

        issues: list[BuildIssue] = []

        # 1. Build ingredient concepts and aliases from synonyms.json.
        for entry in synonym_entries:
            concept = self._concept_from_synonym_entry(entry)
            concepts_by_id[concept.concept_id] = self._merge_concepts(
                concepts_by_id.get(concept.concept_id),
                concept,
            )

            aliases.extend(self._aliases_from_synonym_entry(entry, concept))

        # This index helps map AIFA ingredient names to existing concepts.
        ingredient_alias_index = self._build_ingredient_alias_index(aliases)

        # 2. Build product records from drugs.csv, creating missing concepts when needed.
        for record in drug_records:
            product, new_concepts, product_issues = self._drug_product_from_record(
                record=record,
                ingredient_alias_index=ingredient_alias_index,
            )

            drug_products.append(product)
            issues.extend(product_issues)

            for concept in new_concepts:
                concepts_by_id[concept.concept_id] = self._merge_concepts(
                    concepts_by_id.get(concept.concept_id),
                    concept,
                )

                # Add the AIFA name itself as a clinical alias for the new concept.
                aliases.append(
                    self._make_alias(
                        alias_raw=concept.canonical_it,
                        target_id=concept.concept_id,
                        target_type=TargetType.INGREDIENT,
                        language=AliasLanguage.IT,
                        source=AliasSource.AIFA,
                        priority=90,
                    )
                )

        # 3. Add aliases for drug products and product base names.
        aliases.extend(self._drug_aliases_from_products(drug_products))

        # 4. Build commercial brand aggregations from product base names.
        drug_brands = self._build_drug_brands(drug_products)

        # 5. Remove exact duplicates, then mark aliases that point to multiple targets as unsafe.
        aliases = self._deduplicate_aliases(aliases)
        aliases = self._mark_ambiguous_aliases(aliases)

        report = self._build_report(
            concepts=list(concepts_by_id.values()),
            aliases=aliases,
            drug_products=drug_products,
            drug_brands=drug_brands,
            issues=issues,
        )

        return KnowledgeBase(
            concepts=sorted(concepts_by_id.values(), key=lambda item: item.concept_id),
            aliases=sorted(aliases, key=lambda item: (item.alias_normalized, item.target_id)),
            drug_products=sorted(drug_products, key=lambda item: item.product_id),
            drug_brands=sorted(drug_brands, key=lambda item: item.brand_id),
            report=report,
        )

    def write(self, kb: KnowledgeBase, output_dir: str | Path) -> None:
        """Write processed knowledge-base files to disk."""
        output_dir = Path(output_dir)

        write_jsonl(output_dir / "concepts.jsonl", kb.concepts)
        write_jsonl(output_dir / "aliases.jsonl", kb.aliases)
        write_jsonl(output_dir / "drug_products.jsonl", kb.drug_products)
        write_jsonl(output_dir / "drug_brands.jsonl", kb.drug_brands)
        write_json(output_dir / "build_report.json", kb.report)

    def _concept_from_synonym_entry(self, entry: RawSynonymEntry) -> IngredientConcept:
        """Create an IngredientConcept from a synonym entry."""
        entity_type = EntityType(entry.entity_type)
        normalized_name = self.normalizer.normalize_for_id(entry.ingredient_it)

        return IngredientConcept(
            concept_id=make_ingredient_concept_id(entity_type, normalized_name),
            entity_type=entity_type,
            canonical_it=entry.ingredient_it,
            canonical_en=entry.ingredient_en,
            sources=[AliasSource.UMLS.value],
        )

    def _aliases_from_synonym_entry(
        self,
        entry: RawSynonymEntry,
        concept: IngredientConcept,
    ) -> list[Alias]:
        """Create aliases from canonical names and synonym lists."""
        aliases: list[Alias] = []

        aliases.append(
            self._make_alias(
                alias_raw=entry.ingredient_it,
                target_id=concept.concept_id,
                target_type=TargetType.INGREDIENT,
                language=AliasLanguage.IT,
                source=AliasSource.UMLS,
                priority=95,
            )
        )

        if entry.ingredient_en:
            aliases.append(
                self._make_alias(
                    alias_raw=entry.ingredient_en,
                    target_id=concept.concept_id,
                    target_type=TargetType.INGREDIENT,
                    language=AliasLanguage.EN,
                    source=AliasSource.UMLS,
                    priority=90,
                )
            )

        for synonym in entry.synonyms_it:
            aliases.append(
                self._make_alias(
                    alias_raw=synonym,
                    target_id=concept.concept_id,
                    target_type=TargetType.INGREDIENT,
                    language=AliasLanguage.IT,
                    source=AliasSource.UMLS,
                    priority=80,
                )
            )

        for synonym in entry.synonyms_en:
            aliases.append(
                self._make_alias(
                    alias_raw=synonym,
                    target_id=concept.concept_id,
                    target_type=TargetType.INGREDIENT,
                    language=AliasLanguage.EN,
                    source=AliasSource.UMLS,
                    priority=75,
                )
            )

        return aliases

    def _drug_product_from_record(
        self,
        record: RawDrugRecord,
        ingredient_alias_index: dict[str, str],
    ) -> tuple[DrugProduct, list[IngredientConcept], list[BuildIssue]]:
        """Create a DrugProduct and missing concepts from one raw drug record."""
        parsed_name = self.drug_name_parser.parse(record.drug_name)

        new_concepts: list[IngredientConcept] = []
        issues: list[BuildIssue] = []

        active_ids = []
        for composition in record.compositions:
            concept_id, maybe_concept = self._resolve_or_create_ingredient_concept(
                raw_name=composition,
                entity_type=EntityType.ACTIVE,
                ingredient_alias_index=ingredient_alias_index,
            )
            active_ids.append(concept_id)
            if maybe_concept:
                new_concepts.append(maybe_concept)
                issues.append(
                    BuildIssue(
                        severity=IssueSeverity.INFO,
                        code="created_missing_active_concept",
                        message="Created missing active ingredient concept from drug CSV.",
                        item_id=concept_id,
                        metadata={"raw_name": composition, "drug_code": record.drug_code},
                    )
                )

        excipient_ids = []
        for excipient in record.excipients:
            concept_id, maybe_concept = self._resolve_or_create_ingredient_concept(
                raw_name=excipient,
                entity_type=EntityType.INACTIVE,
                ingredient_alias_index=ingredient_alias_index,
            )
            excipient_ids.append(concept_id)
            if maybe_concept:
                new_concepts.append(maybe_concept)
                issues.append(
                    BuildIssue(
                        severity=IssueSeverity.INFO,
                        code="created_missing_inactive_concept",
                        message="Created missing inactive ingredient concept from drug CSV.",
                        item_id=concept_id,
                        metadata={"raw_name": excipient, "drug_code": record.drug_code},
                    )
                )

        product = DrugProduct(
            product_id=f"drug:aic:{record.drug_code}",
            drug_code=record.drug_code,
            drug_name=record.drug_name,
            brand_name=parsed_name.base_name or None,
            normalized_brand_name=parsed_name.normalized_base_name or None,
            drug_form=record.drug_form,
            atc=record.atc,
            active_ingredient_ids=sorted(set(active_ids)),
            excipient_ids=sorted(set(excipient_ids)),
            source=AliasSource.AIFA.value,
            metadata={
                "normalized_full_name": parsed_name.normalized_full_name,
            },
        )

        return product, new_concepts, issues

    def _resolve_or_create_ingredient_concept(
        self,
        raw_name: str,
        entity_type: EntityType,
        ingredient_alias_index: dict[str, str],
    ) -> tuple[str, IngredientConcept | None]:
        """Resolve an ingredient name to an existing concept or create a new one."""
        normalized_alias = self.normalizer.normalize(raw_name)
        existing_id = ingredient_alias_index.get(normalized_alias)

        if existing_id:
            return existing_id, None

        concept_id = make_ingredient_concept_id(
            entity_type,
            self.normalizer.normalize_for_id(raw_name),
        )

        return concept_id, IngredientConcept(
            concept_id=concept_id,
            entity_type=entity_type,
            canonical_it=raw_name,
            canonical_en=None,
            sources=[AliasSource.AIFA.value],
        )

    def _drug_aliases_from_products(self, products: list[DrugProduct]) -> list[Alias]:
        """Create aliases for full product names and base commercial names."""
        aliases: list[Alias] = []

        for product in products:
            aliases.append(
                self._make_alias(
                    alias_raw=product.drug_name,
                    target_id=product.product_id,
                    target_type=TargetType.DRUG_PRODUCT,
                    language=AliasLanguage.IT,
                    source=AliasSource.AIFA,
                    priority=90,
                )
            )

            if product.brand_name and product.normalized_brand_name:
                brand_id = f"brand:{self.normalizer.normalize_for_id(product.brand_name)}"
                aliases.append(
                    self._make_alias(
                        alias_raw=product.brand_name,
                        target_id=brand_id,
                        target_type=TargetType.DRUG_BRAND,
                        language=AliasLanguage.IT,
                        source=AliasSource.AIFA,
                        priority=85,
                    )
                )

        return aliases

    def _build_drug_brands(self, products: list[DrugProduct]) -> list[DrugBrand]:
        """Aggregate DrugProduct objects into DrugBrand objects."""
        products_by_brand: dict[str, list[DrugProduct]] = defaultdict(list)

        for product in products:
            if product.brand_name and product.normalized_brand_name:
                brand_id = f"brand:{self.normalizer.normalize_for_id(product.brand_name)}"
                products_by_brand[brand_id].append(product)

        brands: list[DrugBrand] = []

        for brand_id, brand_products in products_by_brand.items():
            product_ids = [product.product_id for product in brand_products]

            signatures_set = {
                tuple(sorted(product.active_ingredient_ids))
                for product in brand_products
                if product.active_ingredient_ids
            }
            signatures = [list(signature) for signature in sorted(signatures_set)]

            flattened_active_ids = sorted({item for signature in signatures for item in signature})

            if len(signatures) == 1:
                brand_status = (
                    BrandStatus.SINGLE_ACTIVE_SIGNATURE
                    if len(signatures[0]) == 1
                    else BrandStatus.MULTI_ACTIVE_SIGNATURE
                )
            else:
                brand_status = BrandStatus.AMBIGUOUS_BRAND

            # Use the most common raw base name. In practice these should be very similar.
            brand_name = brand_products[0].brand_name or brand_id.replace("brand:", "")

            brands.append(
                DrugBrand(
                    brand_id=brand_id,
                    brand_name=brand_name,
                    normalized_brand_name=brand_products[0].normalized_brand_name or "",
                    product_ids=sorted(product_ids),
                    active_ingredient_ids=flattened_active_ids,
                    active_ingredient_signatures=signatures,
                    brand_status=brand_status,
                )
            )

        return brands

    def _build_ingredient_alias_index(self, aliases: list[Alias]) -> dict[str, str]:
        """Build a normalized alias -> concept ID index for ingredient aliases.

        Ambiguous aliases are intentionally ignored here. If the same normalized
        alias points to multiple concepts, the builder should not use it to
        resolve AIFA ingredients automatically.
        """
        targets_by_alias: dict[str, set[str]] = defaultdict(set)

        for alias in aliases:
            if alias.target_type == TargetType.INGREDIENT:
                targets_by_alias[alias.alias_normalized].add(alias.target_id)

        return {
            alias_normalized: next(iter(target_ids))
            for alias_normalized, target_ids in targets_by_alias.items()
            if len(target_ids) == 1
        }

    def _mark_ambiguous_aliases(self, aliases: list[Alias]) -> list[Alias]:
        """Mark aliases that point to more than one target as unsafe."""
        targets_by_alias: dict[tuple[str, TargetType], set[str]] = defaultdict(set)

        for alias in aliases:
            key = (alias.alias_normalized, alias.target_type)
            targets_by_alias[key].add(alias.target_id)

        ambiguous_keys = {
            key: sorted(targets)
            for key, targets in targets_by_alias.items()
            if len(targets) > 1
        }

        updated_aliases: list[Alias] = []

        for alias in aliases:
            key = (alias.alias_normalized, alias.target_type)
            ambiguous_targets = ambiguous_keys.get(key)

            if not ambiguous_targets:
                updated_aliases.append(alias)
                continue

            metadata = dict(alias.metadata)
            metadata["ambiguous_targets"] = ambiguous_targets

            updated_aliases.append(
                alias.model_copy(
                    update={
                        "safe_for_exact_match": False,
                        "requires_context": True,
                        "metadata": metadata,
                    }
                )
            )

        return updated_aliases

    def _make_alias(
        self,
        alias_raw: str,
        target_id: str,
        target_type: TargetType,
        language: AliasLanguage,
        source: AliasSource,
        priority: int,
    ) -> Alias:
        """Create an Alias using the configured normalizer and policy."""
        alias_normalized = self.normalizer.normalize(alias_raw)
        policy = self.alias_policy.classify(alias_raw, alias_normalized)

        return Alias(
            alias_raw=alias_raw,
            alias_normalized=alias_normalized,
            target_id=target_id,
            target_type=target_type,
            language=language,
            source=source,
            alias_category=policy.category,
            priority=priority,
            safe_for_exact_match=policy.safe_for_exact_match,
            requires_context=policy.requires_context,
            metadata={"policy_reason": policy.reason},
        )

    def _merge_concepts(
        self,
        existing: IngredientConcept | None,
        new: IngredientConcept,
    ) -> IngredientConcept:
        """Merge duplicate concepts that share the same concept ID."""
        if existing is None:
            return new

        sources = sorted(set(existing.sources) | set(new.sources))

        return existing.model_copy(
            update={
                "canonical_en": existing.canonical_en or new.canonical_en,
                "sources": sources,
            }
        )

    def _build_report(
        self,
        concepts: list[IngredientConcept],
        aliases: list[Alias],
        drug_products: list[DrugProduct],
        drug_brands: list[DrugBrand],
        issues: list[BuildIssue],
    ) -> KnowledgeBaseBuildReport:
        """Create a build summary report."""
        return KnowledgeBaseBuildReport(
            num_concepts=len(concepts),
            num_aliases=len(aliases),
            num_drug_products=len(drug_products),
            num_drug_brands=len(drug_brands),
            num_ambiguous_aliases=sum(1 for alias in aliases if "ambiguous_targets" in alias.metadata),
            num_unsafe_short_aliases=sum(
                1 for alias in aliases if alias.metadata.get("policy_reason") == "short_alias"
            ),
            num_unresolved_product_ingredients=sum(
                1 for issue in issues if issue.code.startswith("created_missing_")
            ),
            num_ambiguous_brands=sum(
                1 for brand in drug_brands if brand.brand_status == BrandStatus.AMBIGUOUS_BRAND
            ),
            issues=issues,
        )

    def _deduplicate_aliases(self, aliases: list[Alias]) -> list[Alias]:
        """Remove duplicate aliases that point to the same target.

        If the same normalized alias points to the same target multiple times, we
        keep one record and preserve the highest priority. True ambiguity, where the
        same alias points to different targets, is handled later.
        """
        aliases_by_key: dict[tuple[str, str, TargetType], Alias] = {}

        for alias in aliases:
            key = (alias.alias_normalized, alias.target_id, alias.target_type)
            existing = aliases_by_key.get(key)

            if existing is None:
                aliases_by_key[key] = alias
                continue

            if alias.priority > existing.priority:
                merged_metadata = dict(existing.metadata)
                merged_metadata.update(alias.metadata)
                merged_metadata["deduplicated_from"] = "higher_priority_duplicate"

                aliases_by_key[key] = alias.model_copy(
                    update={
                        "metadata": merged_metadata,
                    }
                )

        return list(aliases_by_key.values())