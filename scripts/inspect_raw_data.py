"""Inspect raw input files.

This script is intentionally simple and is meant for local development only.
It helps verify that the configured raw files can be read correctly.
"""

from heliot_terms.resources.loaders import load_drugs_csv, load_synonyms_json


def main() -> None:
    synonyms = load_synonyms_json("data/raw/synonyms.json")
    drugs = load_drugs_csv("data/raw/drugs.csv")

    print(f"Loaded synonym entries: {len(synonyms)}")
    print(f"Loaded drug records: {len(drugs)}")

    print("\nFirst synonym entry:")
    print(synonyms[0].model_dump())

    print("\nFirst drug record:")
    print(drugs[0].model_dump())


if __name__ == "__main__":
    main()