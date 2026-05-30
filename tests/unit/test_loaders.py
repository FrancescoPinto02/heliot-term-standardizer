"""Unit tests for raw resource loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from heliot_terms.resources.loaders import load_drugs_csv, load_synonyms_json


def test_load_synonyms_json(tmp_path: Path) -> None:
    synonyms_path = tmp_path / "synonyms.json"
    synonyms_path.write_text(
        json.dumps(
            [
                {
                    "ingredient_it": "paracetamolo",
                    "ingredient_en": "Paracetamol",
                    "type": "active",
                    "synonyms_it": ["Acetaminofene", "APAP"],
                    "synonyms_en": ["Acetaminophen"],
                }
            ]
        ),
        encoding="utf-8",
    )

    entries = load_synonyms_json(synonyms_path)

    assert len(entries) == 1
    assert entries[0].ingredient_it == "paracetamolo"
    assert entries[0].ingredient_en == "Paracetamol"
    assert entries[0].entity_type == "active"
    assert "Acetaminofene" in entries[0].synonyms_it


def test_load_drugs_csv(tmp_path: Path) -> None:
    drugs_path = tmp_path / "drugs.csv"

    dataframe = pd.DataFrame(
        [
            {
                "drug_code": "012345678",
                "drug_name": "OLMESARTAN AM TEV*30CPR 40+5MG",
                "atc": "C09DB02",
                "drug_form": "COMPRESSE",
                "composition": "Olmesartan medoxomil#Amlodipina",
                "excipients": "Lattosio monoidrato#Magnesio stearato",
            }
        ]
    )
    dataframe.to_csv(drugs_path, index=False)

    records = load_drugs_csv(drugs_path)

    assert len(records) == 1
    assert records[0].drug_code == "012345678"
    assert records[0].drug_name == "OLMESARTAN AM TEV*30CPR 40+5MG"
    assert records[0].compositions == ["Olmesartan medoxomil", "Amlodipina"]
    assert records[0].excipients == ["Lattosio monoidrato", "Magnesio stearato"]