from __future__ import annotations

from heliot_terms.config import load_config
from heliot_terms.pipeline.builder import build_pipeline


def main() -> None:
    config = load_config("configs/default.yaml")
    pipeline = build_pipeline(config)

    examples = [
        "Il paziente ha presentato una reazione allergica ad acetaminofene.",
        "Paziente allergico alla Brumixol.",
        "Rash dopo assunzione di TRANSENE. Intolleranza a Methoxypropiocin documentata.",
        "Intolleranza a F D & C #3.",
        "Allergia al sale di sodio dell'acido carbonico.",
        "Tolleranza ad Actiq",
        "Tolleranza ad actiq 15pastl mucosa os 200mcg",
        "Il paziente non tollera macrogol 3350.",
    ]

    for example in examples:
        result = pipeline.standardize(example)

        print("\n" + "=" * 100)
        print(f"Original:      {result.original_text}")
        print(f"Normalized:    {result.normalized_text}")
        print(f"Standardized:  {result.standardized_text}")

        print("\nMatches:")
        for match in result.matches:
            concepts = [
                concept.display_name(result.output_language)
                for concept in match.concepts
            ]
            print(
                f"- {match.surface!r} -> {match.target_id} "
                f"type={match.target_type} concepts={concepts}"
            )

        print("\nAmbiguous:")
        for match in result.ambiguous:
            print(
                f"- {match.surface!r} -> {match.target_id} "
                f"metadata={match.metadata}"
            )


if __name__ == "__main__":
    main()