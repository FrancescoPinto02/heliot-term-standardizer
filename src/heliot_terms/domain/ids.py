"""Helpers for stable domain identifiers."""

from heliot_terms.domain.enums import EntityType


def entity_type_prefix(entity_type: EntityType) -> str:
    """Return the namespace prefix used in ingredient concept IDs.

    We avoid using raw enum values directly because some values, such as
    'active/inactive', contain characters that are not ideal inside IDs.
    """
    if entity_type == EntityType.ACTIVE:
        return "active"

    if entity_type == EntityType.INACTIVE:
        return "inactive"

    if entity_type == EntityType.ACTIVE_INACTIVE:
        return "active_inactive"

    raise ValueError(f"Unsupported entity type: {entity_type}")


def make_ingredient_concept_id(entity_type: EntityType, normalized_name: str) -> str:
    """Build a stable ingredient concept ID."""
    return f"{entity_type_prefix(entity_type)}:{normalized_name}"