from heliot_terms.config import load_config
from heliot_terms.domain.enums import TargetType
from heliot_terms.matching.aho_corasick_matcher import AhoCorasickMatcher
from heliot_terms.resolution.overlap_resolver import OverlapResolver, OverlapResolverConfig
from heliot_terms.normalization.text_normalizer import TextNormalizer
from heliot_terms.resources.processed_loaders import load_aliases


def main() -> None:
    config = load_config("configs/default.yaml")

    processed_dir = config.paths.processed_dir
    aliases = load_aliases(processed_dir / "aliases.jsonl")

    matcher = AhoCorasickMatcher.from_aliases(
        aliases=aliases,
        include_unsafe=config.matcher.deterministic.include_unsafe_aliases,
    )

    priority_names = config.resolution.target_type_priority
    target_type_priority = {
        TargetType(name): len(priority_names) - index
        for index, name in enumerate(priority_names)
    }

    resolver = OverlapResolver(
        OverlapResolverConfig(
            prefer_longest_match=config.resolution.prefer_longest_match,
            target_type_priority=target_type_priority,
        )
    )

    normalizer = TextNormalizer()

    examples = [
        "Il paziente ha presentato una reazione allergica ad acetaminofene.",
        "Paziente allergico alla Brumixol.",
        "Rash dopo assunzione di TRANSENE. Intolleranza a Methoxypropiocin documentata.",
        "Intolleranza a F D & C #3.",
        "Allergia al sale di sodio dell'acido carbonico.",
        "Tolleranza ad Actiq",
        "Tolleranza ad actiq 15pastl mucosa os 200mcg"
    ]

    for example in examples:
        normalized = normalizer.normalize(example)
        matches = resolver.resolve(matcher.match(normalized))

        print("\n" + "=" * 80)
        print(f"Original:   {example}")
        print(f"Normalized: {normalized}")
        print("Matches:")

        for match in matches:
            print(
                f"- {match.surface!r} -> {match.target_id} "
                f"({match.target_type}) method={match.method}"
            )


if __name__ == "__main__":
    main()