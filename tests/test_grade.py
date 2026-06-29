"""Deterministic graders + the LLM open-answer grader (with a fake model)."""

import pytest

from quizforge import (
    QuizGrade,
    difficulty_weight,
    grade_fill_blank,
    grade_match,
    grade_open_answer,
    normalize,
)


def test_difficulty_weight_scales_and_defaults():
    assert difficulty_weight({"difficulty": "easy"}) == 1
    assert difficulty_weight({"difficulty": "medium"}) == 2
    assert difficulty_weight({"difficulty": "hard"}) == 3
    # missing / unknown difficulty falls back to medium (2)
    assert difficulty_weight({}) == 2
    assert difficulty_weight({"difficulty": "spicy"}) == 2
from fakes import FakeLLM


def test_normalize_collapses_case_space_punct():
    assert normalize("  ICA.  ") == "ica"
    assert normalize("Active   Directory") == "active directory"
    assert normalize("(NetScaler)!") == "netscaler"
    assert normalize(None) == ""


def test_fill_blank_accepts_synonyms_and_is_case_insensitive():
    q = {"accepted_answers": ["ICA", "Independent Computing Architecture"]}
    assert grade_fill_blank(q, "ica")["correct"] is True
    assert grade_fill_blank(q, "  Independent Computing Architecture ")["score"] == 1.0
    wrong = grade_fill_blank(q, "RDP")
    assert wrong["correct"] is False and wrong["score"] == 0.0


def test_fill_blank_empty_answer_is_wrong():
    q = {"accepted_answers": ["x"]}
    assert grade_fill_blank(q, "")["score"] == 0.0


def test_fill_blank_forgives_single_typo():
    q = {"accepted_answers": ["privileges", "rights", "permissions", "access"]}
    res = grade_fill_blank(q, "priviliges")  # the real-world miss
    assert res["correct"] is True and res["score"] == 1.0
    assert res["fuzzy"] is True
    # an exact match is not flagged fuzzy
    exact = grade_fill_blank(q, "Privileges")
    assert exact["correct"] is True and exact["fuzzy"] is False


def test_fill_blank_typo_tolerance_scales_with_length():
    q = {"accepted_answers": ["authentication"]}
    assert grade_fill_blank(q, "authentcation")["correct"] is True   # 1 missing char
    assert grade_fill_blank(q, "authentiction")["correct"] is True   # 2 edits, len>=8


def test_fill_blank_does_not_overaccept_wrong_words():
    # 'permissions' vs a genuinely different answer should stay wrong
    q = {"accepted_answers": ["privileges", "rights", "permissions", "access"]}
    assert grade_fill_blank(q, "passwords")["correct"] is False
    # short answers must match exactly — no fuzzing 'PA' into 'PB'
    short = {"accepted_answers": ["PA", "the firewall"]}
    assert grade_fill_blank(short, "pb")["correct"] is False
    assert grade_fill_blank(short, "PA")["correct"] is True


def test_match_partial_credit_and_rows():
    q = {"pairs": [{"left": "L1", "right": "RDP"}, {"left": "L2", "right": "ICA"}]}
    res = grade_match(q, {"0": "RDP", "1": "wrong"})
    assert res["total"] == 2
    assert res["correct_count"] == 1
    assert res["score"] == 0.5
    assert res["rows"][0]["correct"] is True
    assert res["rows"][1]["correct"] is False


def test_match_all_correct_and_empty():
    q = {"pairs": [{"left": "L1", "right": "A"}, {"left": "L2", "right": "B"}]}
    assert grade_match(q, {"0": "A", "1": "b"})["score"] == 1.0  # case-insensitive
    assert grade_match({"pairs": []}, {})["score"] == 0.0


def test_open_answer_empty_is_zero_without_calling_model():
    q = {"prompt": "Explain X", "rubric": ["a", "b"]}
    # llm that would explode if called — proves the empty path short-circuits.
    boom = FakeLLM(lambda *_: (_ for _ in ()).throw(AssertionError("should not call model")))
    g = grade_open_answer(q, "   ", boom)
    assert g.score == 0.0 and g.verdict == "incorrect" and g.missed == ["a", "b"]


def test_open_answer_uses_model_and_clamps_score():
    q = {"prompt": "Explain X", "model_answer": "...", "rubric": ["a"]}
    over = QuizGrade(score=1.7, verdict="correct", feedback="great", covered=["a"], missed=[])
    llm = FakeLLM(lambda schema, msgs: over)
    g = grade_open_answer(q, "my answer", llm)
    assert g.score == 1.0  # clamped into 0..1


def test_open_answer_returns_none_on_model_failure():
    q = {"prompt": "Explain X", "model_answer": "...", "rubric": ["a"]}
    llm = FakeLLM(lambda *_: (_ for _ in ()).throw(RuntimeError("provider down")))
    assert grade_open_answer(q, "my answer", llm) is None
