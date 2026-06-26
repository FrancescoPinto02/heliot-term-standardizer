"""Unit tests for fuzzy candidate extraction."""

from heliot_terms.fallback.candidate_extractor import (
    CandidateExtractorConfig,
    ResidualCandidateExtractor,
)


def test_candidate_extractor_ignores_protected_spans() -> None:
    extractor = ResidualCandidateExtractor(
        CandidateExtractorConfig(
            max_ngram_tokens=2,
            min_token_chars=3,
            min_candidate_chars=6,
        )
    )

    text = "allergia ad acetaminofene e macrogl 3350"

    # Protect exact match "acetaminofene".
    protected_start = text.index("acetaminofene")
    protected_end = protected_start + len("acetaminofene")

    candidates = extractor.extract(
        text=text,
        protected_spans=[(protected_start, protected_end)],
    )

    candidate_texts = {candidate.text for candidate in candidates}

    assert "acetaminofene" not in candidate_texts
    assert "macrogl" in candidate_texts
    assert "macrogl 3350" in candidate_texts


def test_candidate_extractor_does_not_include_sentence_final_punctuation() -> None:
    extractor = ResidualCandidateExtractor()

    text = "reazione allergica ad acetaminofne."

    candidates = extractor.extract(text=text, protected_spans=[])

    candidate_texts = {candidate.text for candidate in candidates}

    assert "acetaminofne" in candidate_texts
    assert "acetaminofne." not in candidate_texts


def test_candidate_extractor_filters_stopword_only_candidates() -> None:
    extractor = ResidualCandidateExtractor(
        CandidateExtractorConfig(
            max_ngram_tokens=2,
            min_token_chars=3,
            min_candidate_chars=2,
        )
    )

    text = "ad alla con di"

    candidates = extractor.extract(text=text, protected_spans=[])

    assert candidates == []