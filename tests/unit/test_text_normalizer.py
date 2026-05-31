from heliot_terms.normalization.text_normalizer import TextNormalizer


def test_normalizer_lowercases_and_strips_accents() -> None:
    normalizer = TextNormalizer()

    assert normalizer.normalize("Èritrosina") == "eritrosina"


def test_normalizer_handles_e_numbers() -> None:
    normalizer = TextNormalizer()

    assert normalizer.normalize("Biossido di titanio (E 171)") == "biossido di titanio e171"
    assert normalizer.normalize("Biossido di titanio (e171)") == "biossido di titanio e171"


def test_normalizer_handles_fd_c_alias() -> None:
    normalizer = TextNormalizer()

    assert normalizer.normalize("F D & C #3") == "f d c 3"


def test_normalizer_handles_unicode_dashes() -> None:
    normalizer = TextNormalizer()

    assert normalizer.normalize("alcool 2–feniletilico") == "alcool 2-feniletilico"


def test_normalize_for_id() -> None:
    normalizer = TextNormalizer()

    assert normalizer.normalize_for_id("Lattosio monoidrato") == "lattosio_monoidrato"
    assert normalizer.normalize_for_id("Biossido di titanio (E171)") == "biossido_di_titanio_e171"


def test_normalize_with_mapping_maps_simple_match_to_original_span() -> None:
    normalizer = TextNormalizer()

    original = "Paziente allergico ad Acetaminofene."
    normalized = normalizer.normalize_with_mapping(original)

    start = normalized.text.index("acetaminofene")
    end = start + len("acetaminofene")

    original_start, original_end = normalized.original_span(start, end)

    assert original[original_start:original_end] == "Acetaminofene"


def test_normalize_with_mapping_handles_e_numbers() -> None:
    normalizer = TextNormalizer()

    original = "Allergia a biossido di titanio (E 171)."
    normalized = normalizer.normalize_with_mapping(original)

    assert "e171" in normalized.text

    start = normalized.text.index("e171")
    end = start + len("e171")

    original_start, original_end = normalized.original_span(start, end)

    assert original[original_start:original_end] == "E 171"


def test_normalize_with_mapping_handles_fd_c_alias() -> None:
    normalizer = TextNormalizer()

    original = "Intolleranza a F D & C #3."
    normalized = normalizer.normalize_with_mapping(original)

    start = normalized.text.index("f d c 3")
    end = start + len("f d c 3")

    original_start, original_end = normalized.original_span(start, end)

    assert original[original_start:original_end] == "F D & C #3"