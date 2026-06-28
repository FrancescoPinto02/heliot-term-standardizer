from unittest.mock import Mock, patch

from heliot_terms.fallback.semantic.ner_candidate_extractor import (
    NerSemanticCandidateExtractor,
)


def test_ner_candidate_extractor_keeps_drug_entities() -> None:
    fake_pipeline = Mock(
        return_value=[
            {
                "entity_group": "DRUG",
                "word": "Paracetamolo",
                "start": 11,
                "end": 23,
                "score": 0.98,
            }
        ]
    )

    with patch(
        "heliot_terms.fallback.semantic.ner_candidate_extractor.pipeline",
        return_value=fake_pipeline,
    ):
        extractor = NerSemanticCandidateExtractor()
        candidates = extractor.extract("Allergia a Paracetamolo.", protected_spans=[])

    assert len(candidates) == 1
    assert candidates[0].text == "Paracetamolo"
    assert candidates[0].label == "DRUG"
    assert candidates[0].score == 0.98


def test_ner_candidate_extractor_filters_non_drug_entities() -> None:
    fake_pipeline = Mock(
        return_value=[
            {
                "entity_group": "DISEASE",
                "word": "asma",
                "start": 10,
                "end": 14,
                "score": 0.95,
            }
        ]
    )

    with patch(
        "heliot_terms.fallback.semantic.ner_candidate_extractor.pipeline",
        return_value=fake_pipeline,
    ):
        extractor = NerSemanticCandidateExtractor()
        candidates = extractor.extract("Paziente con asma.", protected_spans=[])

    assert candidates == []


def test_ner_candidate_extractor_respects_protected_spans() -> None:
    fake_pipeline = Mock(
        return_value=[
            {
                "entity_group": "DRUG",
                "word": "Paracetamolo",
                "start": 11,
                "end": 23,
                "score": 0.98,
            }
        ]
    )

    with patch(
        "heliot_terms.fallback.semantic.ner_candidate_extractor.pipeline",
        return_value=fake_pipeline,
    ):
        extractor = NerSemanticCandidateExtractor()
        candidates = extractor.extract(
            "Allergia a Paracetamolo.",
            protected_spans=[(13, 25)],
        )

    assert candidates == []


def test_ner_candidate_extractor_trims_punctuation() -> None:
    fake_pipeline = Mock(
        return_value=[
            {
                "entity_group": "DRUG",
                "word": "(Paracetamolo)",
                "start": 11,
                "end": 25,
                "score": 0.98,
            }
        ]
    )

    with patch(
        "heliot_terms.fallback.semantic.ner_candidate_extractor.pipeline",
        return_value=fake_pipeline,
    ):
        extractor = NerSemanticCandidateExtractor()
        candidates = extractor.extract("Allergia a (Paracetamolo).", protected_spans=[])

    assert len(candidates) == 1
    assert candidates[0].text == "Paracetamolo"