"""Bank: load/save round-trip, counts, grow (with a fake model)."""

import yaml

from quizforge import Bank
from fakes import FakeLLM, bank_responder


def _write_bank(tmp_path, data):
    p = tmp_path / "bank.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


def test_load_splits_meta_and_questions(tmp_path):
    p = _write_bank(tmp_path, {
        "id": "citrix", "title": "Citrix", "material": "some source text",
        "questions": [{"id": "q1", "type": "mc", "difficulty": "easy", "prompt": "x"}],
    })
    bank = Bank.load(p)
    assert bank.meta["title"] == "Citrix"
    assert bank.material == "some source text"
    assert len(bank) == 1
    assert "questions" not in bank.meta


def test_counts_helpers():
    bank = Bank(questions=[
        {"type": "mc", "difficulty": "easy"}, {"type": "mc", "difficulty": "hard"},
        {"type": "match", "difficulty": "medium"},
    ])
    assert bank.counts_by_type() == {"mc": 2, "match": 1}
    assert bank.counts_by_difficulty() == {"easy": 1, "hard": 1, "medium": 1}


def test_save_round_trip_preserves_meta_and_questions(tmp_path):
    bank = Bank(questions=[{"id": "q1", "type": "mc", "difficulty": "easy", "prompt": "x"}],
                meta={"id": "t", "title": "T"})
    out = tmp_path / "out.yaml"
    bank.save(out)
    reloaded = Bank.load(out)
    assert reloaded.meta["title"] == "T"
    assert reloaded.questions == bank.questions
    # questions key comes last in the dumped file
    assert list(yaml.safe_load(out.read_text())) == ["id", "title", "questions"]


def test_grow_appends_and_grounds_on_material():
    bank = Bank(questions=[], meta={"material": "grounding text"})
    llm = FakeLLM(bank_responder())
    new = bank.grow(llm, targets={"mc": 4, "fill_blank": 0, "match": 0, "short": 0, "freetext": 0})
    assert len(new) == 4
    assert len(bank) == 4
    assert all(q["type"] == "mc" for q in new)


def test_grow_without_material_raises():
    bank = Bank(questions=[], meta={})
    llm = FakeLLM(bank_responder())
    try:
        bank.grow(llm, targets={"mc": 2})
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "material" in str(exc)
