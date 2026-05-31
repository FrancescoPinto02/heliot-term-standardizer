"""Loaders for processed knowledge-base files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from heliot_terms.domain.models import Alias, DrugBrand, DrugProduct, IngredientConcept


ModelT = TypeVar("ModelT", bound=BaseModel)


def load_aliases(path: str | Path) -> list[Alias]:
    """Load processed aliases from aliases.jsonl."""
    return _load_jsonl(path, Alias)


def load_concepts(path: str | Path) -> list[IngredientConcept]:
    """Load processed ingredient concepts from concepts.jsonl."""
    return _load_jsonl(path, IngredientConcept)


def load_drug_products(path: str | Path) -> list[DrugProduct]:
    """Load processed drug products from drug_products.jsonl."""
    return _load_jsonl(path, DrugProduct)


def load_drug_brands(path: str | Path) -> list[DrugBrand]:
    """Load processed drug brands from drug_brands.jsonl."""
    return _load_jsonl(path, DrugBrand)


def _load_jsonl(path: str | Path, model_cls: type[ModelT]) -> list[ModelT]:
    """Load JSONL records and validate them with a Pydantic model."""
    path = Path(path)

    records: list[ModelT] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
                records.append(model_cls.model_validate(payload))
            except Exception as exc:
                raise ValueError(f"Invalid record at {path}:{line_number}") from exc

    return records