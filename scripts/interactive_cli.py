from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from heliot_terms.config import load_config
from heliot_terms.pipeline.builder import build_pipeline
from heliot_terms.pipeline.models import StandardizationResult


EXIT_COMMANDS = {":q", ":quit", ":exit", "exit", "quit"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive CLI for HELIOT term standardization."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full standardization result as JSON.",
    )
    parser.add_argument(
        "--show-matches",
        action="store_true",
        help="Print structured matches after each standardized note.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)

    print("Loading configuration...")
    config = load_config(config_path)

    print("Initializing pipeline...")
    start_time = time.perf_counter()
    pipeline = build_pipeline(config)
    elapsed = time.perf_counter() - start_time

    print(f"Pipeline initialized in {elapsed:.2f}s.")
    print()
    print("Insert a clinical note and press Enter.")
    print("Type ':q', ':quit' or ':exit' to terminate.")
    print()

    while True:
        try:
            note = input("note> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("Terminated.")
            break

        if not note:
            continue

        if note.lower() in EXIT_COMMANDS:
            print("Terminated.")
            break

        start_time = time.perf_counter()
        result = pipeline.standardize(note)
        elapsed = time.perf_counter() - start_time

        _print_result(
            result=result,
            elapsed=elapsed,
            as_json=args.json,
            show_matches=args.show_matches,
        )


def _print_result(
    result: StandardizationResult,
    elapsed: float,
    as_json: bool,
    show_matches: bool,
) -> None:
    """Print a standardization result in CLI-friendly format."""
    if as_json:
        print(
            json.dumps(
                result.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"Elapsed: {elapsed:.4f}s")
        print()
        return

    print()
    print("Standardized text:")
    print(result.standardized_text)
    print()
    print(
        f"Matches: {len(result.matches)} | "
        f"Ambiguous: {len(result.ambiguous)} | "
        f"Elapsed: {elapsed:.4f}s"
    )

    if show_matches:
        _print_matches(result)

    print()


def _print_matches(result: StandardizationResult) -> None:
    """Print structured matches in a compact readable format."""
    if result.matches:
        print()
        print("Matches:")

        for mention in result.matches:
            concept_names = [
                concept.display_name(result.output_language)
                for concept in mention.concepts
            ]

            print(
                "- "
                f"'{mention.surface}' "
                f"→ {mention.target_id} "
                f"({mention.target_type}, method={mention.method}, "
                f"confidence={mention.confidence:.3f})"
            )

            if concept_names:
                print(f"  concepts: {concept_names}")

    if result.ambiguous:
        print()
        print("Ambiguous:")

        for mention in result.ambiguous:
            print(
                "- "
                f"'{mention.surface}' "
                f"→ {mention.target_id} "
                f"({mention.target_type}, method={mention.method})"
            )


if __name__ == "__main__":
    main()