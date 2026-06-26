from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ConfigModel(BaseModel):
    """Base model for configuration sections.

    Extra fields are ignored so the YAML can evolve without immediately breaking
    older code.
    """

    model_config = ConfigDict(extra="ignore")


class ProjectConfig(ConfigModel):
    """Project metadata."""

    name: str = "heliot-term-standardizer"
    version: str = "0.1.0"


class PathsConfig(ConfigModel):
    """Input/output paths."""

    raw_drugs_csv: Path = Path("data/raw/drugs.csv")
    raw_synonyms_json: Path = Path("data/raw/synonyms.json")
    processed_dir: Path = Path("data/processed")
    indexes_dir: Path = Path("data/indexes")


class StandardizationConfig(ConfigModel):
    """Standardization output configuration."""

    output_language: Literal["it", "en"] = "en"
    note_output_policy: Literal["annotate", "replace", "structured_only"] = "annotate"


class KnowledgeBaseConfig(ConfigModel):
    """Knowledge-base construction options."""

    include_active_ingredients: bool = True
    include_inactive_ingredients: bool = True
    include_drug_products: bool = True
    include_drug_brands: bool = True


class AliasesConfig(ConfigModel):
    """Alias filtering options."""

    min_safe_alias_length: int = 4
    enable_short_aliases: bool = False
    short_alias_requires_context: bool = True
    include_technical_aliases_in_exact_index: bool = False


class DrugNamesConfig(ConfigModel):
    """Drug-name parsing options."""

    split_on_asterisk: bool = True
    generate_short_brand_variants: bool = False
    include_full_product_name_alias: bool = True
    include_base_product_name_alias: bool = True


class DrugBrandsConfig(ConfigModel):
    """Drug-brand handling options."""

    enabled: bool = True
    annotate_active_ingredients: bool = True
    include_excipients_for_brand_mentions: bool = False
    include_excipients_for_exact_product_mentions: bool = False
    ambiguous_brand_policy: str = "structured_only"


class DeterministicMatcherConfig(ConfigModel):
    """Deterministic matcher configuration."""

    type: str = "aho_corasick"
    enabled: bool = True
    include_unsafe_aliases: bool = False


class MatcherConfig(ConfigModel):
    """Matcher configuration."""

    deterministic: DeterministicMatcherConfig = Field(
        default_factory=DeterministicMatcherConfig
    )


class ResolutionConfig(ConfigModel):
    """Overlap resolution configuration."""

    prefer_longest_match: bool = True
    target_type_priority: list[str] = Field(
        default_factory=lambda: ["drug_product", "drug_brand", "ingredient"]
    )


class IndexesConfig(ConfigModel):
    """Index persistence configuration."""

    persist: bool = False
    path: Path = Path("data/indexes/aho_corasick.pkl")


class FallbackConfig(ConfigModel):
    """Generic fallback configuration."""

    enabled: bool = False


class FuzzyFallbackConfig(ConfigModel):
    """SymSpell fuzzy fallback configuration."""

    enabled: bool = False
    type: str = "symspell"

    target_types: list[str] = Field(default_factory=lambda: ["ingredient"])

    max_ngram_tokens: int = 4
    min_token_chars: int = 3
    min_candidate_chars: int = 6

    max_dictionary_edit_distance: int = 2
    prefix_length: int = 7
    max_lookup_edit_distance: int = 2

    max_suggestions: int = 5
    ambiguity_margin: float = 0.05

    short_max_chars: int = 8
    medium_max_chars: int = 14
    min_score_short: float = 0.95
    min_score_medium: float = 0.90
    min_score_long: float = 0.86

    allowed_policy_reasons: list[str] = Field(
        default_factory=lambda: ["clinical_alias"]
    )
    extra_stopwords: list[str] = Field(default_factory=list)


class FallbacksConfig(ConfigModel):
    """Fallback layer configuration.

    These are intentionally present already, even if disabled in v0, so future
    fuzzy, embedding, or LLM fallbacks can be added without changing the YAML
    structure.
    """

    fuzzy: FuzzyFallbackConfig = Field(default_factory=FuzzyFallbackConfig)
    embeddings: FallbackConfig = Field(default_factory=FallbackConfig)
    llm: FallbackConfig = Field(default_factory=FallbackConfig)


class AppConfig(ConfigModel):
    """Full application configuration."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    standardization: StandardizationConfig = Field(default_factory=StandardizationConfig)
    knowledge_base: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)
    aliases: AliasesConfig = Field(default_factory=AliasesConfig)
    drug_names: DrugNamesConfig = Field(default_factory=DrugNamesConfig)
    drug_brands: DrugBrandsConfig = Field(default_factory=DrugBrandsConfig)
    matcher: MatcherConfig = Field(default_factory=MatcherConfig)
    resolution: ResolutionConfig = Field(default_factory=ResolutionConfig)
    indexes: IndexesConfig = Field(default_factory=IndexesConfig)
    fallbacks: FallbacksConfig = Field(default_factory=FallbacksConfig)


def load_config(path: str | Path) -> AppConfig:
    """Load a YAML configuration file."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    return AppConfig.model_validate(payload)