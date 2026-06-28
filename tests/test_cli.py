"""CLI: stats / sample (no model) and grow (via a fake-model factory)."""

import json

import pytest
import yaml

from quizforge.cli import main


def _bank_file(tmp_path, with_material=True):
    data = {
        "id": "t", "title": "T",
        "questions": [
            {"id": "q1", "type": "mc", "difficulty": "easy", "prompt": "m1",
             "choices": ["a", "b", "c"], "answer_idx": 0, "explanation": "e"},
            {"id": "q2", "type": "match", "difficulty": "hard", "prompt": "m2",
             "pairs": [{"left": "L", "right": "R"}], "explanation": "e"},
        ],
    }
    if with_material:
        data["material"] = "grounding source text"
    p = tmp_path / "bank.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


def test_stats(tmp_path, capsys):
    p = _bank_file(tmp_path)
    assert main(["stats", str(p)]) == 0
    out = capsys.readouterr().out
    assert "total questions : 2" in out
    assert "'mc': 1" in out and "'match': 1" in out


def test_sample_text_and_json(tmp_path, capsys):
    p = _bank_file(tmp_path)
    assert main(["sample", str(p), "--seed", "1"]) == 0
    assert "Sampled" in capsys.readouterr().out

    assert main(["sample", str(p), "--seed", "1", "--json"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert isinstance(parsed, list) and len(parsed) == 2  # only 2 in the bank


def test_grow_dry_run_does_not_write(tmp_path, capsys):
    p = _bank_file(tmp_path)
    before = p.read_text()
    rc = main(["grow", str(p), "--llm", "fakes:make_fake_llm", "--dry-run",
               "--mc", "5", "--fill-blank", "0", "--match", "0",
               "--short", "0", "--freetext", "0"])
    assert rc == 0
    assert "Generated" in capsys.readouterr().out
    assert p.read_text() == before  # unchanged on dry-run


def test_grow_writes_and_grows_bank(tmp_path, capsys):
    p = _bank_file(tmp_path)
    rc = main(["grow", str(p), "--llm", "fakes:make_fake_llm",
               "--mc", "5", "--fill-blank", "0", "--match", "0",
               "--short", "0", "--freetext", "0"])
    assert rc == 0
    data = yaml.safe_load(p.read_text())
    mc = [q for q in data["questions"] if q["type"] == "mc"]
    assert len(mc) == 5  # 1 existing + 4 new
    # new ids continue after the highest existing (q2)
    assert any(q["id"] == "q3" for q in data["questions"])


def test_grow_without_model_errors(tmp_path):
    p = _bank_file(tmp_path)
    with pytest.raises(SystemExit):
        main(["grow", str(p), "--mc", "2"])
