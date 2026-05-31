from pathlib import Path

import yaml

from heliot_terms.matching.factory import build_matcher
from heliot_terms.normalization.text_normalizer import TextNormalizer
from heliot_terms.resolution.factory import build_overlap_resolver
from heliot_terms.resources.processed_loaders import load_aliases


def main() -> None:
    config_path = Path("configs/default.yaml")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    processed_dir = Path(config["paths"]["processed_dir"])
    matcher_type = config["matcher"]["deterministic"]["type"]

    aliases = load_aliases(processed_dir / "aliases.jsonl")

    matcher = build_matcher(
        matcher_type=matcher_type,
        aliases=aliases,
        include_unsafe=False,
    )

    resolver = build_overlap_resolver(config)

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