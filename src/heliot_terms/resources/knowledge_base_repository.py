from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from heliot_terms.domain.models import Alias, DrugBrand, DrugProduct, IngredientConcept
from heliot_terms.resources.processed_loaders import (
    load_aliases,
    load_concepts,
    load_drug_brands,
    load_drug_products,
)


@dataclass(frozen=True)
class KnowledgeBaseRepository:
    """In-memory access layer for processed KB records."""

    aliases: list[Alias]
    concepts_by_id: dict[str, IngredientConcept]
    products_by_id: dict[str, DrugProduct]
    brands_by_id: dict[str, DrugBrand]

    @classmethod
    def from_processed_dir(cls, processed_dir: str | Path) -> KnowledgeBaseRepository:
        """Load all processed KB files from a directory."""
        processed_dir = Path(processed_dir)

        aliases = load_aliases(processed_dir / "aliases.jsonl")
        concepts = load_concepts(processed_dir / "concepts.jsonl")
        products = load_drug_products(processed_dir / "drug_products.jsonl")
        brands = load_drug_brands(processed_dir / "drug_brands.jsonl")

        return cls(
            aliases=aliases,
            concepts_by_id={concept.concept_id: concept for concept in concepts},
            products_by_id={product.product_id: product for product in products},
            brands_by_id={brand.brand_id: brand for brand in brands},
        )

    def get_concept(self, concept_id: str) -> IngredientConcept | None:
        """Return a concept by ID."""
        return self.concepts_by_id.get(concept_id)

    def get_product(self, product_id: str) -> DrugProduct | None:
        """Return a drug product by ID."""
        return self.products_by_id.get(product_id)

    def get_brand(self, brand_id: str) -> DrugBrand | None:
        """Return a drug brand by ID."""
        return self.brands_by_id.get(brand_id)