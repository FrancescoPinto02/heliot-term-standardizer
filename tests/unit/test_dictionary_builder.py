"""Unit tests for the knowledge-base builder."""

from heliot_terms.domain.enums import BrandStatus
from heliot_terms.resources.dictionary_builder import KnowledgeBaseBuilder
from heliot_terms.resources.raw_models import RawDrugRecord, RawSynonymEntry


def test_builder_creates_concepts_aliases_products_and_brands() -> None:
    synonym_entries = [
        RawSynonymEntry(
            ingredient_it="paracetamolo",
            ingredient_en="Paracetamol",
            entity_type="active",
            synonyms_it=["acetaminofene"],
            synonyms_en=["acetaminophen", "APAP"],
        )
    ]

    drug_records = [
        RawDrugRecord(
            drug_code="012345678",
            drug_name="TACHIPIRINA*20CPR 500MG",
            atc="N02BE01",
            drug_form="COMPRESSE",
            compositions=["paracetamolo"],
            excipients=["lattosio monoidrato"],
        )
    ]

    builder = KnowledgeBaseBuilder()
    kb = builder.build(synonym_entries=synonym_entries, drug_records=drug_records)

    concept_ids = {concept.concept_id for concept in kb.concepts}
    assert "active:paracetamolo" in concept_ids
    assert "inactive:lattosio_monoidrato" in concept_ids

    alias_targets = {(alias.alias_normalized, alias.target_id) for alias in kb.aliases}
    assert ("acetaminofene", "active:paracetamolo") in alias_targets
    assert ("tachipirina", "brand:tachipirina") in alias_targets

    assert len(kb.drug_products) == 1
    assert kb.drug_products[0].brand_name == "TACHIPIRINA"

    assert len(kb.drug_brands) == 1
    assert kb.drug_brands[0].brand_status == BrandStatus.SINGLE_ACTIVE_SIGNATURE


def test_short_alias_is_not_safe_for_exact_matching() -> None:
    synonym_entries = [
        RawSynonymEntry(
            ingredient_it="paracetamolo",
            ingredient_en="Paracetamol",
            entity_type="active",
            synonyms_it=["APAP"],
            synonyms_en=[],
        )
    ]

    builder = KnowledgeBaseBuilder()
    kb = builder.build(synonym_entries=synonym_entries, drug_records=[])

    apap_alias = next(alias for alias in kb.aliases if alias.alias_normalized == "apap")

    assert apap_alias.safe_for_exact_match is False
    assert apap_alias.requires_context is True


def test_builder_supports_active_inactive_synonym_entries() -> None:
    synonym_entries = [
        RawSynonymEntry(
            ingredient_it="alcool etilico",
            ingredient_en="Ethyl alcohol",
            entity_type="active/inactive",
            synonyms_it=["etanolo"],
            synonyms_en=["Ethanol"],
        )
    ]

    drug_records = [
        RawDrugRecord(
            drug_code="000000001",
            drug_name="ALCOOL TEST*1FL",
            compositions=["alcool etilico"],
            excipients=[],
        ),
        RawDrugRecord(
            drug_code="000000002",
            drug_name="ECCIPIENTE TEST*1FL",
            compositions=[],
            excipients=["alcool etilico"],
        ),
    ]

    builder = KnowledgeBaseBuilder()
    kb = builder.build(synonym_entries=synonym_entries, drug_records=drug_records)

    concept_ids = {concept.concept_id for concept in kb.concepts}
    assert "active_inactive:alcool_etilico" in concept_ids

    product_by_code = {product.drug_code: product for product in kb.drug_products}

    assert product_by_code["000000001"].active_ingredient_ids == [
        "active_inactive:alcool_etilico"
    ]
    assert product_by_code["000000002"].excipient_ids == [
        "active_inactive:alcool_etilico"
    ]