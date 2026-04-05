from eval.run_anchor_diagnostics import _extract_json_payload, _stage_enabled, _validate_reformulation


def test_validate_reformulation_accepts_choice_from_corrected_candidates() -> None:
    accepted, reason = _validate_reformulation(
        "What is Hobbes understanding of personal interest?",
        [
            "What is Hobbes' desire about personal interest?",
            "What is Hobbes' pleasure about personal interest?",
        ],
        "What is Hobbes' desire about personal interest?",
    )
    assert accepted == "What is Hobbes' desire about personal interest?"
    assert reason == "accepted"


def test_validate_reformulation_rejects_choice_outside_corrected_candidates() -> None:
    accepted, reason = _validate_reformulation(
        "How to build true knowledge according to Hobbes?",
        [
            "How to build understanding according to Hobbes?",
            "How to build science according to Hobbes?",
        ],
        "How to build reason according to Hobbes?",
    )
    assert accepted is None
    assert reason == "choice_not_in_corrected_candidates"


def test_validate_reformulation_rejects_no_meaningful_change() -> None:
    accepted, reason = _validate_reformulation(
        "What does Hobbes mean by logic?",
        [
            "What does Hobbes mean by logic?",
            "What does Hobbes mean by reason?",
        ],
        "What does Hobbes mean by logic?",
    )
    assert accepted is None
    assert reason == "no_meaningful_change_from_original"


def test_extract_json_payload_salvages_truncated_json() -> None:
    payload = _extract_json_payload(
        '{'
        '"choice": "According to Hobbes, what characterizes human life without authority?",'
        '"corrected_candidates": ['
        '"According to Hobbes, what characterizes human life without authority?",'
        '"According to Hobbes, what characterizes human life without justice?"'
        '],'
        '"reason": "The first candidate maintains'
    )
    assert payload is not None
    assert payload["choice"] == "According to Hobbes, what characterizes human life without authority?"
    assert payload["corrected_candidates"] == [
        "According to Hobbes, what characterizes human life without authority?",
        "According to Hobbes, what characterizes human life without justice?",
    ]


def test_stage_enabled_uses_suite_default_when_test_has_no_override() -> None:
    assert _stage_enabled({"run_reformulation": False}, {}, "run_reformulation") is False


def test_stage_enabled_prefers_test_override_over_suite_default() -> None:
    assert _stage_enabled({"run_reformulation": False}, {"run_reformulation": True}, "run_reformulation") is True
