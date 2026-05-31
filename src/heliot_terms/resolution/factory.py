"""Factory helpers for the resolution layer."""

from __future__ import annotations

from heliot_terms.domain.enums import TargetType
from heliot_terms.resolution.overlap_resolver import OverlapResolver, OverlapResolverConfig


def build_overlap_resolver(config: dict) -> OverlapResolver:
    """Build an OverlapResolver from a configuration dictionary."""
    resolution_config = config.get("resolution", {})

    priority_names = resolution_config.get(
        "target_type_priority",
        ["drug_product", "drug_brand", "ingredient"],
    )

    # Higher number means higher priority.
    target_type_priority = {
        TargetType(name): len(priority_names) - index
        for index, name in enumerate(priority_names)
    }

    return OverlapResolver(
        OverlapResolverConfig(
            prefer_longest_match=resolution_config.get("prefer_longest_match", True),
            target_type_priority=target_type_priority,
        )
    )