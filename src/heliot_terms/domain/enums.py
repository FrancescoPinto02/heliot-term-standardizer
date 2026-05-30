from enum import StrEnum


class EntityType(StrEnum):
    """Type of ingredient concept."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ACTIVE_INACTIVE = "active/inactive"


class TargetType(StrEnum):
    """Type of entity pointed to by an alias."""

    INGREDIENT = "ingredient"
    DRUG_PRODUCT = "drug_product"
    DRUG_BRAND = "drug_brand"


class AliasLanguage(StrEnum):
    """Language associated with an alias."""

    IT = "it"
    EN = "en"
    UNKNOWN = "unknown"


class AliasSource(StrEnum):
    """Source from which an alias was collected."""

    AIFA = "AIFA"
    UMLS = "UMLS"
    DERIVED = "DERIVED"


class AliasCategory(StrEnum):
    """Practical category used to decide how an alias should be matched."""

    CLINICAL = "clinical"
    TECHNICAL = "technical"
    UNSAFE = "unsafe"


class BrandStatus(StrEnum):
    """Composition status of a commercial drug brand."""

    SINGLE_ACTIVE_SIGNATURE = "single_active_signature"
    MULTI_ACTIVE_SIGNATURE = "multi_active_signature"
    AMBIGUOUS_BRAND = "ambiguous_brand"


class IssueSeverity(StrEnum):
    """Severity of an issue found while building the knowledge base."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"