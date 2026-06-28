from __future__ import annotations

from pathlib import Path

from heliot_terms.config import load_config
from heliot_terms.domain.enums import AliasCategory, TargetType
from heliot_terms.domain.models import Alias
from heliot_terms.fallback.semantic.encoder import SemanticEncoder
from heliot_terms.fallback.semantic.hnsw_index import SemanticVectorIndex
from heliot_terms.fallback.semantic.models import SemanticIndexMetadata
from heliot_terms.resources.processed_loaders import load_aliases


SEMANTIC_INDEX_SUBDIR = "semantic"
ALLOWED_POLICY_REASONS = frozenset({"clinical_alias"})


def main() -> None:
    config = load_config("configs/default.yaml")

    aliases = load_aliases(config.paths.processed_dir / "aliases.jsonl")
    selected_aliases = _select_semantic_aliases(aliases)

    if not selected_aliases:
        raise ValueError("No aliases selected for semantic index.")

    encoder = SemanticEncoder()

    texts = [alias.alias_normalized for alias in selected_aliases]
    vectors = encoder.encode(texts)

    metadata = [
        SemanticIndexMetadata(
            vector_id=index,
            target_id=alias.target_id,
            target_type=alias.target_type,
            alias_raw=alias.alias_raw,
            alias_normalized=alias.alias_normalized,
            language=alias.language.value,
            source=alias.source.value,
            metadata={
                "priority": alias.priority,
                "policy_reason": alias.metadata.get("policy_reason"),
            },
        )
        for index, alias in enumerate(selected_aliases)
    ]

    semantic_index = SemanticVectorIndex.build(
        vectors=vectors,
        metadata=metadata,
    )

    output_dir = Path(config.paths.indexes_dir) / SEMANTIC_INDEX_SUBDIR
    semantic_index.save(
        index_dir=output_dir,
        model_name=encoder.model_name,
    )

    print("Semantic index built successfully.")
    print(f"Aliases indexed: {len(selected_aliases)}")
    print(f"Output directory: {output_dir}")


def _select_semantic_aliases(aliases: list[Alias]) -> list[Alias]:
    """Select aliases suitable for semantic embedding fallback."""
    selected: list[Alias] = []
    seen: set[tuple[str, str]] = set()

    for alias in aliases:
        if alias.target_type != TargetType.INGREDIENT:
            continue

        if alias.alias_category != AliasCategory.CLINICAL:
            continue

        if not alias.safe_for_exact_match:
            continue

        if alias.metadata.get("policy_reason") not in ALLOWED_POLICY_REASONS:
            continue

        if not alias.alias_normalized:
            continue

        key = (alias.alias_normalized, alias.target_id)
        if key in seen:
            continue

        seen.add(key)
        selected.append(alias)

    return selected


if __name__ == "__main__":
    main()