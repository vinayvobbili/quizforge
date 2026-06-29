"""Sampling — draw a fresh, mixed-format test from a deep bank.

A deep bank (far more questions than any one test shows) plus unseen-first
sampling is what makes two learners rarely see the same test, which defeats
answer-sharing. No LLM here — pure, deterministic given an rng.

Questions may carry an optional ``concept`` tag — a short key for the *idea* a
question tests (several differently-worded questions can share one). When tags
are present the sampler spreads a test across concepts, so one attempt doesn't
stack three near-identical questions on the same idea. Untagged banks behave
exactly as before (each question is treated as its own unique concept).
"""

import random
from collections import Counter, defaultdict
from typing import Iterable, List, Optional

# How many of each format a single test draws. Sums to 20 by default.
DEFAULT_BLUEPRINT = {"mc": 8, "fill_blank": 4, "match": 2, "short": 4, "freetext": 2}
DIFFICULTY_ORDER = ("easy", "medium", "hard")


def _concept(q: dict) -> str:
    """The idea a question tests. Untagged → its own id, so it never collides."""
    return q.get("concept") or q.get("id") or ""


def _difficulty_spread(ordered: List[dict]) -> List[dict]:
    """Re-order a list so easy/medium/hard round-robin, preserving incoming order
    (and thus unseen-first) within each difficulty bucket."""
    buckets: dict = defaultdict(list)
    for q in ordered:
        buckets[q.get("difficulty", "medium")].append(q)
    out: List[dict] = []
    progressed = True
    while progressed:
        progressed = False
        for level in DIFFICULTY_ORDER:
            if buckets[level]:
                out.append(buckets[level].pop(0))
                progressed = True
    for qs in buckets.values():  # difficulties outside the canonical order
        out.extend(qs)
    return out


def pick_spread(pool: List[dict], want: int, seen_ids: Iterable[str],
                rng: Optional[random.Random] = None,
                concept_counts: Optional[dict] = None) -> List[dict]:
    """Pick ``want`` questions from one format's pool.

    Ordering preferences, strongest first: fewest-used *concept* (so the draw
    favours ideas not already on the test), then unseen-before-seen, then an
    easy/medium/hard round-robin. ``concept_counts`` carries how many times each
    concept has already been chosen elsewhere on the same test, so diversity
    holds across formats too — not just within one.
    """
    rng = rng or random
    if want <= 0 or not pool:
        return []
    seen = set(seen_ids)
    counts = Counter(concept_counts or {})

    unseen = [q for q in pool if q.get("id") not in seen]
    reused = [q for q in pool if q.get("id") in seen]
    rng.shuffle(unseen)
    rng.shuffle(reused)
    ordered = _difficulty_spread(unseen + reused)  # exhaust unseen before recycling
    pos = {id(q): i for i, q in enumerate(ordered)}

    target = min(want, len(ordered))
    picked: List[dict] = []
    chosen: set = set()
    while len(picked) < target:
        cands = [q for q in ordered if id(q) not in chosen]
        if not cands:
            break
        # Lowest concept count wins; ties broken by the unseen-first / difficulty order.
        best = min(cands, key=lambda q: (counts[_concept(q)], pos[id(q)]))
        picked.append(best)
        chosen.add(id(best))
        counts[_concept(best)] += 1
    return picked


def sample_test(questions: List[dict], blueprint: Optional[dict] = None,
                seen_ids: Iterable[str] = (), rng: Optional[random.Random] = None) -> List[dict]:
    """Draw a mixed-format test per ``blueprint``, unseen-first and concept-spread.

    Falls short of the blueprint only when the bank lacks enough of a format; the
    test still assembles with whatever it can. Concept counts accumulate across
    formats so the whole test spreads over ideas, then the final order is shuffled
    so formats interleave instead of arriving grouped by type.
    """
    rng = rng or random
    blueprint = blueprint or DEFAULT_BLUEPRINT
    by_type: dict = defaultdict(list)
    for q in questions:
        by_type[q.get("type", "mc")].append(q)

    picked: List[dict] = []
    concept_counts: Counter = Counter()
    for qtype, want in blueprint.items():
        chosen = pick_spread(by_type.get(qtype, []), want, seen_ids,
                             rng=rng, concept_counts=concept_counts)
        picked.extend(chosen)
        for q in chosen:
            concept_counts[_concept(q)] += 1
    rng.shuffle(picked)
    return picked
