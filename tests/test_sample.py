"""Blueprint sampling: blueprint mix, unseen-first, difficulty spread, shortfall."""

import random

from quizforge import DEFAULT_BLUEPRINT, pick_spread, sample_test


def _bank():
    """A deep bank: plenty of every format across difficulties."""
    out = []
    n = 0
    for qtype, count in {"mc": 30, "fill_blank": 20, "match": 12, "short": 16, "freetext": 12}.items():
        for i in range(count):
            n += 1
            out.append({"id": f"q{n}", "type": qtype,
                        "difficulty": ["easy", "medium", "hard"][i % 3],
                        "prompt": f"{qtype} {i}"})
    return out


def test_sample_test_matches_blueprint_counts():
    bank = _bank()
    test = sample_test(bank, seen_ids=set(), rng=random.Random(1))
    assert len(test) == sum(DEFAULT_BLUEPRINT.values()) == 20
    by_type = {}
    for q in test:
        by_type[q["type"]] = by_type.get(q["type"], 0) + 1
    assert by_type == DEFAULT_BLUEPRINT


def test_sample_test_is_shuffled_not_grouped_by_type():
    bank = _bank()
    test = sample_test(bank, seen_ids=set(), rng=random.Random(7))
    types = [q["type"] for q in test]
    # Extremely unlikely to be perfectly grouped if shuffled.
    grouped = sorted(types, key=lambda t: list(DEFAULT_BLUEPRINT).index(t))
    assert types != grouped


def test_pick_spread_prefers_unseen():
    pool = [{"id": f"q{i}", "difficulty": "easy", "type": "mc"} for i in range(10)]
    seen = {f"q{i}" for i in range(5)}  # q0..q4 seen
    picked = pick_spread(pool, 5, seen, rng=random.Random(3))
    assert {q["id"] for q in picked}.isdisjoint(seen)  # all 5 are unseen


def test_pick_spread_spreads_difficulty():
    pool = []
    for d in ("easy", "medium", "hard"):
        pool += [{"id": f"{d}{i}", "difficulty": d, "type": "mc"} for i in range(5)]
    picked = pick_spread(pool, 3, set(), rng=random.Random(0))
    assert {q["difficulty"] for q in picked} == {"easy", "medium", "hard"}


def test_sample_test_handles_shortfall_gracefully():
    # Bank with only 2 mc and nothing else — still returns what it can.
    bank = [{"id": "a", "type": "mc", "difficulty": "easy", "prompt": "x"},
            {"id": "b", "type": "mc", "difficulty": "hard", "prompt": "y"}]
    test = sample_test(bank, seen_ids=set(), rng=random.Random(2))
    assert len(test) == 2
    assert all(q["type"] == "mc" for q in test)


def test_pick_spread_empty_pool_and_zero_want():
    assert pick_spread([], 5, set()) == []
    assert pick_spread([{"id": "a", "difficulty": "easy"}], 0, set()) == []
