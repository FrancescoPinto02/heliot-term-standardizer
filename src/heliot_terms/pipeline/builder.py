from __future__ import annotations

from heliot_terms.config import AppConfig
from heliot_terms.domain.enums import TargetType
from heliot_terms.matching.aho_corasick_matcher import AhoCorasickMatcher
from heliot_terms.normalization.text_normalizer import TextNormalizer
from heliot_terms.pipeline.standardization_pipeline import (
    StandardizationPipeline,
    StandardizationPipelineConfig,
)
from heliot_terms.resources.knowledge_base_repository import KnowledgeBaseRepository
from heliot_terms.resolution.overlap_resolver import OverlapResolver, OverlapResolverConfig


def build_pipeline(config: AppConfig) -> StandardizationPipeline:
    """Build a complete standardization pipeline from typed configuration."""
    repository = KnowledgeBaseRepository.from_processed_dir(config.paths.processed_dir)

    matcher = _build_exact_matcher(config, repository)
    resolver = _build_overlap_resolver(config)

    pipeline_config = StandardizationPipelineConfig(
        output_language=config.standardization.output_language,
        note_output_policy=config.standardization.note_output_policy,
        ambiguous_brand_policy=config.drug_brands.ambiguous_brand_policy,
        include_excipients_for_product_mentions=(
            config.drug_brands.include_excipients_for_exact_product_mentions
        ),
    )

    return StandardizationPipeline(
        repository=repository,
        matcher=matcher,
        resolver=resolver,
        normalizer=TextNormalizer(),
        config=pipeline_config,
    )


def _build_exact_matcher(
    config: AppConfig,
    repository: KnowledgeBaseRepository,
) -> AhoCorasickMatcher:
    """Build the deterministic exact matcher.

    For v0, Aho-Corasick is the only supported implementation. Keeping this
    logic here makes it easy to add a second implementation later without
    spreading factory code across the project.
    """
    matcher_config = config.matcher.deterministic

    if not matcher_config.enabled:
        raise ValueError("The deterministic matcher is disabled, but v0 requires it.")

    matcher_type = matcher_config.type.strip().lower()

    if matcher_type not in {"aho_corasick", "ahocorasick", "aho-corasick"}:
        raise ValueError(f"Unsupported deterministic matcher type: {matcher_config.type}")

    return AhoCorasickMatcher.from_aliases(
        aliases=repository.aliases,
        include_unsafe=matcher_config.include_unsafe_aliases,
    )


def _build_overlap_resolver(config: AppConfig) -> OverlapResolver:
    """Build the overlap resolver from configuration."""
    priority_names = config.resolution.target_type_priority

    target_type_priority = {
        TargetType(name): len(priority_names) - index
        for index, name in enumerate(priority_names)
    }

    return OverlapResolver(
        OverlapResolverConfig(
            prefer_longest_match=config.resolution.prefer_longest_match,
            target_type_priority=target_type_priority,
        )
    )