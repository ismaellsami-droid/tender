from tender.engine.key_concepts import extract_key_concepts


def test_extract_personal_interest() -> None:
    assert extract_key_concepts("What is Hobbes understanding of personal interest?")[:1] == ["personal interest"]


def test_extract_logic() -> None:
    assert extract_key_concepts("What does Hobbes mean by logic?")[:1] == ["logic"]


def test_extract_prudence_and_science() -> None:
    concepts = extract_key_concepts("How does Hobbes distinguish prudence from science?")
    assert concepts[:2] == ["prudence", "science"]


def test_extract_fear_and_social_contract() -> None:
    concepts = extract_key_concepts("What is the role of fear in the social contract?")
    assert concepts[:2] == ["social contract", "fear"]


def test_extract_conceptual_verb_fight() -> None:
    concepts = extract_key_concepts("Why does Hobbes think humans fight so much?")
    assert "fight" in concepts


def test_extract_right_and_wrong_as_contrast_pair() -> None:
    concepts = extract_key_concepts("How do we define what is right and wrong to do according to Hobbes?")
    assert concepts[:1] == ["right and wrong"]


def test_extract_natural_condition_without_mankind_tail() -> None:
    concepts = extract_key_concepts("How does Hobbes describe the natural condition of mankind?")
    assert concepts[:1] == ["natural condition"]
    assert "mankind" not in concepts


def test_do_not_extract_generic_drive_verb() -> None:
    concepts = extract_key_concepts("According to Hobbes, what drives human actions?")
    assert concepts[:1] == ["human actions"]
    assert "drive" not in concepts


def test_extract_sense_of_time_as_of_phrase() -> None:
    concepts = extract_key_concepts("What does Hobbes think about sense of time?")
    assert concepts[:1] == ["sense of time"]
