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
from heliot_terms.fallback.acceptance import FuzzyAcceptanceConfig
from heliot_terms.fallback.base import BaseFallbackMatcher
from heliot_terms.fallback.candidate_extractor import CandidateExtractorConfig
from heliot_terms.fallback.symspell_fuzzy_matcher import (
    SymSpellFuzzyMatcher,
    SymSpellFuzzyMatcherConfig,
)


def build_pipeline(config: AppConfig) -> StandardizationPipeline:
    """Build a complete standardization pipeline from typed configuration."""
    repository = KnowledgeBaseRepository.from_processed_dir(config.paths.processed_dir)

    matcher = _build_exact_matcher(config, repository)
    resolver = _build_overlap_resolver(config)
    fallback_matchers = _build_fallback_matchers(config, repository)

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
        fallback_matchers=fallback_matchers,
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


def _build_fallback_matchers(
    config: AppConfig,
    repository: KnowledgeBaseRepository,
) -> list[BaseFallbackMatcher]:
    """Build optional fallback matchers from configuration."""
    fallback_matchers: list[BaseFallbackMatcher] = []

    fuzzy_config = config.fallbacks.fuzzy

    if fuzzy_config.enabled:
        fuzzy_type = fuzzy_config.type.strip().lower()

        if fuzzy_type not in {"symspell", "sym_spell"}:
            raise ValueError(f"Unsupported fuzzy fallback type: {fuzzy_config.type}")

        target_types = tuple(TargetType(name) for name in fuzzy_config.target_types)

        stopwords = None
        if fuzzy_config.extra_stopwords:
            from heliot_terms.fallback.candidate_extractor import _DEFAULT_STOPWORDS

            stopwords = frozenset(_DEFAULT_STOPWORDS | set(fuzzy_config.extra_stopwords))

        candidate_extractor_config = CandidateExtractorConfig(
            max_ngram_tokens=fuzzy_config.max_ngram_tokens,
            min_token_chars=fuzzy_config.min_token_chars,
            min_candidate_chars=fuzzy_config.min_candidate_chars,
            stopwords=stopwords if stopwords is not None else CandidateExtractorConfig().stopwords,
        )

        acceptance_config = FuzzyAcceptanceConfig(
            max_suggestions=fuzzy_config.max_suggestions,
            ambiguity_margin=fuzzy_config.ambiguity_margin,
            short_max_chars=fuzzy_config.short_max_chars,
            medium_max_chars=fuzzy_config.medium_max_chars,
            min_score_short=fuzzy_config.min_score_short,
            min_score_medium=fuzzy_config.min_score_medium,
            min_score_long=fuzzy_config.min_score_long,
        )

        matcher_config = SymSpellFuzzyMatcherConfig(
            max_dictionary_edit_distance=fuzzy_config.max_dictionary_edit_distance,
            prefix_length=fuzzy_config.prefix_length,
            max_lookup_edit_distance=fuzzy_config.max_lookup_edit_distance,
            target_types=target_types,
            allowed_policy_reasons=tuple(fuzzy_config.allowed_policy_reasons),
            candidate_extractor=candidate_extractor_config,
            acceptance=acceptance_config,
        )

        fallback_matchers.append(
            SymSpellFuzzyMatcher.from_aliases(
                aliases=repository.aliases,
                config=matcher_config,
            )
        )

    return fallback_matchers