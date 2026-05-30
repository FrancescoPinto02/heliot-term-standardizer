"""Build the processed terminology knowledge base."""

from pathlib import Path

import yaml

from heliot_terms.resources.dictionary_builder import KnowledgeBaseBuilder
from heliot_terms.resources.loaders import load_drugs_csv, load_synonyms_json


def main() -> None:
    config_path = Path("configs/default.yaml")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    raw_drugs_csv = config["paths"]["raw_drugs_csv"]
    raw_synonyms_json = config["paths"]["raw_synonyms_json"]
    processed_dir = config["paths"]["processed_dir"]

    synonym_entries = load_synonyms_json(raw_synonyms_json)
    drug_records = load_drugs_csv(raw_drugs_csv)

    builder = KnowledgeBaseBuilder()
    kb = builder.build(synonym_entries=synonym_entries, drug_records=drug_records)
    builder.write(kb, processed_dir)

    print("Knowledge base built successfully.")
    print(f"Concepts: {kb.report.num_concepts}")
    print(f"Aliases: {kb.report.num_aliases}")
    print(f"Drug products: {kb.report.num_drug_products}")
    print(f"Drug brands: {kb.report.num_drug_brands}")
    print(f"Ambiguous aliases: {kb.report.num_ambiguous_aliases}")
    print(f"Ambiguous brands: {kb.report.num_ambiguous_brands}")


if __name__ == "__main__":
    main()