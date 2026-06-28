"""Sampling — draw a fresh, mixed-format test from a deep bank.

A deep bank (far more questions than any one test shows) plus unseen-first
sampling is what makes two learners rarely see the same test, which defeats
answer-sharing. No LLM here — pure, deterministic given an rng.
"""

import random
from collections import defaultdict
from typing import Iterable, List, Optional

# How many of each format a single test draws. Sums to 20 by default.
DEFAULT_BLUEPRINT = {"mc": 8, "fill_blank": 4, "match": 2, "short": 4, "freetext": 2}
DIFFICULTY_ORDER = ("easy", "medium", "hard")


def pick_spread(pool: List[dict], want: int, seen_ids: Iterable[str],
                rng: Optional[random.Random] = None) -> List[dict]:
    """Pick ``want`` questions from one format's pool, unseen-first + difficulty-spread.

    Unseen questions are exhausted before reusing seen ones; within that order we
    round-robin easy/medium/hard so a draw doesn't come back all-hard or all-easy.
    """
    rng = rng or random
    if want <= 0 or not pool:
        return []
    seen = set(seen_ids)
    unseen = [q for q in pool if q.get("id") not in seen]
    reused = [q for q in pool if q.get("id") in seen]
    rng.shuffle(unseen)
    rng.shuffle(reused)
    ordered = unseen + reused  # exhaust unseen before recycling

    buckets: dict = defaultdict(list)
    for q in ordered:
        buckets[q.get("difficulty", "medium")].append(q)

    target = min(want, len(ordered))
    picked: List[dict] = []
    while len(picked) < target:
        progressed = False
        for level in DIFFICULTY_ORDER:
            if buckets[level]:
                picked.append(buckets[level].pop(0))
                progressed = True
                if len(picked) >= target:
                    break
        if not progressed:  # difficulties outside the canonical order — drain them
            leftovers = [q for qs in buckets.values() for q in qs]
            picked.extend(leftovers[: target - len(picked)])
            break
    return picked


def sample_test(questions: List[dict], blueprint: Optional[dict] = None,
                seen_ids: Iterable[str] = (), rng: Optional[random.Random] = None) -> List[dict]:
    """Draw a mixed-format test per ``blueprint``, unseen-first within each format.

    Falls short of the blueprint only when the bank lacks enough of a format; the
    test still assembles with whatever it can. Final order is shuffled so formats
    interleave instead of arriving grouped by type.
    """
    rng = rng or random
    blueprint = blueprint or DEFAULT_BLUEPRINT
    by_type: dict = defaultdict(list)
    for q in questions:
        by_type[q.get("type", "mc")].append(q)

    picked: List[dict] = []
    for qtype, want in blueprint.items():
        picked.extend(pick_spread(by_type.get(qtype, []), want, seen_ids, rng=rng))
    rng.shuffle(picked)
    return picked
