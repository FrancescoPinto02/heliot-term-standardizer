from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from heliot_terms.resources.raw_models import RawDrugRecord, RawSynonymEntry


def load_synonyms_json(path: str | Path) -> list[RawSynonymEntry]:
    """Load synonym entries from a JSON file.

    Parameters
    ----------
    path:
        Path to the synonyms JSON file. The file is expected to contain a list
        of objects with fields such as ingredient_it, ingredient_en, type,
        synonyms_it, and synonyms_en.

    Returns
    -------
    list[RawSynonymEntry]
        Cleaned and validated synonym entries.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    if not isinstance(raw_data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(raw_data).__name__}.")

    entries: list[RawSynonymEntry] = []

    for index, item in enumerate(raw_data):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid synonym entry at index {index}: expected object.")

        entries.append(
            RawSynonymEntry(
                ingredient_it=item.get("ingredient_it", ""),
                ingredient_en=item.get("ingredient_en"),
                entity_type=item.get("type", ""),
                synonyms_it=item.get("synonyms_it", []),
                synonyms_en=item.get("synonyms_en", []),
            )
        )

    return entries


def load_drugs_csv(path: str | Path) -> list[RawDrugRecord]:
    """Load drug records from a CSV file.

    Parameters
    ----------
    path:
        Path to the drugs CSV file.

    Returns
    -------
    list[RawDrugRecord]
        Cleaned and validated drug records.

    Notes
    -----
    The loader is intentionally tolerant with column names because different
    exports may use slightly different labels, for example ``drug_form`` or
    ``drug_form_full_descr``.
    """
    path = Path(path)

    dataframe = pd.read_csv(path, dtype=str, encoding="utf-8")
    dataframe = dataframe.fillna("")

    records: list[RawDrugRecord] = []

    for row_index, row in dataframe.iterrows():
        row_dict = {str(key): value for key, value in row.to_dict().items()}

        drug_code = _get_first_available(row_dict, ["drug_code", "aic", "AIC"])
        drug_name = _get_first_available(row_dict, ["drug_name", "name", "denominazione"])
        atc = _get_first_available(row_dict, ["atc", "atc_code", "ATC"])
        drug_form = _get_first_available(
            row_dict,
            ["drug_form", "drug_form_full_descr", "form", "pharmaceutical_form"],
        )

        composition_raw = _get_first_available(
            row_dict,
            ["composition", "compositions", "active_ingredients"],
        )
        excipients_raw = _get_first_available(
            row_dict,
            ["excipients", "inactive_ingredients"],
        )

        try:
            records.append(
                RawDrugRecord(
                    drug_code=drug_code,
                    drug_name=drug_name,
                    atc=atc,
                    drug_form=drug_form,
                    compositions=_split_hash_list(composition_raw),
                    excipients=_split_hash_list(excipients_raw),
                )
            )
        except ValueError as exc:
            raise ValueError(f"Invalid drug record at CSV row {row_index + 2}: {exc}") from exc

    return records


def _split_hash_list(value: str | None) -> list[str]:
    """Split a hash-separated ingredient list.

    The source CSV stores multiple active ingredients and excipients using
    ``#`` as separator. Empty fragments are ignored.
    """
    if value is None:
        return []

    return [fragment.strip() for fragment in str(value).split("#") if fragment.strip()]


def _get_first_available(row: dict[str, Any], column_names: list[str]) -> str:
    """Return the first available value among a list of possible column names."""
    for column_name in column_names:
        if column_name in row:
            value = str(row[column_name] or "").strip()
            if value:
                return value

    return ""