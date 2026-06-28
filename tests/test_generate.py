"""Generation: targets/shortfall, validation, dedup, id assignment, steering."""

import random

from quizforge import generate_bank
from quizforge.schemas import MCBatch
from fakes import FakeLLM, bank_responder


def test_generate_to_targets_returns_new_questions_with_ids():
    llm = FakeLLM(bank_responder())
    targets = {"mc": 8, "fill_blank": 4, "match": 2, "short": 4, "freetext": 2}
    new = generate_bank("some material", llm, targets=targets, existing=[])
    counts = {}
    for q in new:
        counts[q["type"]] = counts.get(q["type"], 0) + 1
    assert counts == targets
    ids = [q["id"] for q in new]
    assert ids == [f"q{i}" for i in range(1, len(new) + 1)]  # sequential from q1
    assert all(q["difficulty"] in ("easy", "medium", "hard") for q in new)


def test_generate_only_makes_the_shortfall():
    llm = FakeLLM(bank_responder())
    existing = [{"id": f"q{i}", "type": "mc", "prompt": f"old mc {i}", "difficulty": "easy",
                 "choices": ["a", "b", "c"], "answer_idx": 0, "explanation": "e"} for i in range(1, 6)]
    new = generate_bank("m", llm, targets={"mc": 8, "fill_blank": 0, "match": 0,
                                            "short": 0, "freetext": 0}, existing=existing)
    assert len([q for q in new if q["type"] == "mc"]) == 3  # 8 target - 5 existing
    # New ids continue after the highest existing id.
    assert new[0]["id"] == "q6"


def test_generate_dedupes_against_existing_prompts():
    # Responder always returns the SAME prompt -> everything dedupes -> 0 made.
    def same(schema, _msgs):
        from quizforge.schemas import GenMC
        return schema(questions=[GenMC(prompt="identical prompt", choices=["a", "b", "c"],
                                       answer_idx=0, explanation="e", difficulty="easy")])
    llm = FakeLLM(same)
    existing = [{"id": "q1", "type": "mc", "prompt": "identical prompt", "difficulty": "easy",
                 "choices": ["a", "b", "c"], "answer_idx": 0, "explanation": "e"}]
    new = generate_bank("m", llm, targets={"mc": 5, "fill_blank": 0, "match": 0,
                                            "short": 0, "freetext": 0}, existing=existing)
    assert new == []  # nothing new survived dedup


def test_generate_drops_malformed_questions():
    # MC with too few choices and a bad answer_idx must be rejected.
    from quizforge.schemas import GenMC
    bad = GenMC(prompt="bad one", choices=["only-one"], answer_idx=9, explanation="e", difficulty="easy")
    good = GenMC(prompt="good one", choices=["a", "b", "c"], answer_idx=1, explanation="e", difficulty="hard")

    def mixed(schema, _msgs):
        return MCBatch(questions=[bad, good])
    llm = FakeLLM(mixed)
    new = generate_bank("m", llm, targets={"mc": 1, "fill_blank": 0, "match": 0,
                                            "short": 0, "freetext": 0}, existing=[])
    assert len(new) == 1 and new[0]["prompt"] == "good one"


def test_generate_skips_unknown_format_in_targets():
    # Zero out the real formats so only the bogus (skipped) one and mc remain.
    llm = FakeLLM(bank_responder())
    targets = {"mc": 2, "fill_blank": 0, "match": 0, "short": 0, "freetext": 0, "bogus": 5}
    new = generate_bank("m", llm, targets=targets, existing=[])
    types = {q["type"] for q in new}
    assert "bogus" not in types  # unknown format skipped, no crash
    assert types == {"mc"}


def test_generate_passes_coverage_into_prompt():
    captured = {}

    def capture(schema, messages):
        captured["human"] = messages[-1][1]
        return bank_responder(per_call=2)(schema, messages)
    llm = FakeLLM(capture)
    generate_bank("MATERIAL-XYZ", llm, targets={"mc": 2, "fill_blank": 0, "match": 0,
                  "short": 0, "freetext": 0}, existing=[], coverage="MY-COVERAGE-HINT")
    assert "MY-COVERAGE-HINT" in captured["human"]
    assert "MATERIAL-XYZ" in captured["human"]
