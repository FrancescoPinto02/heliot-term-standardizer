from heliot_terms.normalization.drug_name_parser import DrugNameParser


def test_parse_transene_name() -> None:
    parser = DrugNameParser()

    parsed = parser.parse("TRANSENE*30CPS 10MG")

    assert parsed.full_name == "TRANSENE*30CPS 10MG"
    assert parsed.base_name == "TRANSENE"
    assert parsed.normalized_base_name == "transene"


def test_parse_olanzapina_acc_name() -> None:
    parser = DrugNameParser()

    parsed = parser.parse("OLANZAPINA ACC*56CPR RIV5MG")

    assert parsed.base_name == "OLANZAPINA ACC"
    assert parsed.normalized_base_name == "olanzapina acc"


def test_parse_olmesartan_combination_name() -> None:
    parser = DrugNameParser()

    parsed = parser.parse("OLMESARTAN AM TEV*30CPR 40+5MG")

    assert parsed.base_name == "OLMESARTAN AM TEV"
    assert parsed.normalized_full_name == "olmesartan am tev 30cpr 40+5mg"


def test_parse_name_without_asterisk() -> None:
    parser = DrugNameParser()

    parsed = parser.parse("TACHIPIRINA 500 MG COMPRESSE")

    assert parsed.base_name == "TACHIPIRINA 500 MG COMPRESSE"