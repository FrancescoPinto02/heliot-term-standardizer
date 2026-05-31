from __future__ import annotations

from pathlib import Path

from heliot_terms.matching.factory import build_matcher
from heliot_terms.normalization.text_normalizer import TextNormalizer
from heliot_terms.pipeline.standardization_pipeline import (
    StandardizationPipeline,
    StandardizationPipelineConfig,
)
from heliot_terms.resources.knowledge_base_repository import KnowledgeBaseRepository
from heliot_terms.resolution.factory import build_overlap_resolver


def build_standardization_pipeline(config: dict) -> StandardizationPipeline:
    """Build a complete standardization pipeline from configuration."""
    processed_dir = Path(config["paths"]["processed_dir"])

    repository = KnowledgeBaseRepository.from_processed_dir(processed_dir)

    matcher_config = config.get("matcher", {}).get("deterministic", {})
    matcher = build_matcher(
        matcher_type=matcher_config.get("type", "aho_corasick"),
        aliases=repository.aliases,
        include_unsafe=matcher_config.get("include_unsafe_aliases", False),
    )

    resolver = build_overlap_resolver(config)

    standardization_config = config.get("standardization", {})
    drug_brand_config = config.get("drug_brands", {})

    pipeline_config = StandardizationPipelineConfig(
        output_language=standardization_config.get("output_language", "en"),
        note_output_policy=standardization_config.get("note_output_policy", "annotate"),
        ambiguous_brand_policy=drug_brand_config.get(
            "ambiguous_brand_policy",
            "structured_only",
        ),
        include_excipients_for_product_mentions=drug_brand_config.get(
            "include_excipients_for_exact_product_mentions",
            False,
        ),
    )

    return StandardizationPipeline(
        repository=repository,
        matcher=matcher,
        resolver=resolver,
        normalizer=TextNormalizer(),
        config=pipeline_config,
    )