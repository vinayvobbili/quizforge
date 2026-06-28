"""Generation — grow a question bank from source material with an injected LLM.

``generate_bank`` drafts questions strictly from the material you provide,
validates each one, dedupes against what's already in the bank (and within the
run), assigns sequential ids, and returns the NEW questions as plain dicts. It
never writes files and never duplicates an existing prompt, so it's safe to
re-run to "top up" a bank to target sizes.
"""

import logging
from collections import Counter
from typing import List, Optional

from .llm import structured_output
from .schemas import DIFFICULTIES, FORMATS
from .text import normalize

logger = logging.getLogger("quizforge")

# Default TOTAL bank size per format — comfortably deep for a 20-question test
# (default blueprint draws mc8/fill4/match2/short4/freetext2) so unseen
# questions last across many retakes.
DEFAULT_TARGETS = {"mc": 40, "fill_blank": 20, "match": 12, "short": 16, "freetext": 12}
DEFAULT_BATCH = 8
DEFAULT_DIFF_SPLIT = {"easy": 0.30, "medium": 0.45, "hard": 0.25}

DEFAULT_SYSTEM = (
    "You are an expert quiz-content author. Write questions STRICTLY grounded in the provided "
    "material — do not invent facts beyond it. Questions must be accurate, unambiguous, and test "
    "real understanding, not trivia. For multiple choice, exactly one option is correct and the "
    "distractors are plausible. Vary difficulty as requested. Return strict JSON for the "
    "requested schema."
)

# Domain-neutral default. Callers with a specific emphasis (e.g. attack-scenario
# coverage for security training) pass their own ``coverage`` string.
DEFAULT_COVERAGE = (
    "Cover the breadth of the material. Frame at least some questions as realistic applied "
    "scenarios rather than abstract recall."
)


def _diff_counts(n: int, split: dict) -> dict:
    """Split a batch of n into easy/medium/hard per ``split`` (sums to n)."""
    easy = round(n * split.get("easy", 0))
    hard = round(n * split.get("hard", 0))
    medium = max(0, n - easy - hard)
    return {"easy": easy, "medium": medium, "hard": hard}


def _valid_difficulty(d: str) -> str:
    d = (d or "").strip().lower()
    return d if d in DIFFICULTIES else "medium"


def _gen_batch(llm, schema, fmt_desc, material, n, existing_prompts, *,
               system, coverage, diff_split):
    counts = _diff_counts(n, diff_split)
    avoid = "\n".join(f"  - {p}" for p in list(existing_prompts)[-40:]) or "  (none yet)"
    coverage_line = f"COVERAGE: {coverage}\n" if coverage else ""
    human = (
        f"{material}\n\n"
        f"Write exactly {n} {fmt_desc}.\n"
        f"Difficulty mix: {counts['easy']} easy, {counts['medium']} medium, {counts['hard']} hard "
        f"(set each question's difficulty field accordingly).\n"
        f"{coverage_line}"
        "Ground every fact in the material above; do not invent details.\n"
        f"Do NOT duplicate or lightly reword any of these existing questions:\n{avoid}\n"
    )
    chain = structured_output(llm, schema)
    batch = chain.invoke([("system", system), ("human", human)])
    return batch.questions


def _validate(qtype: str, q, seen_norms: set) -> Optional[dict]:
    """Coerce a generated question to a clean dict, or drop it (return None)."""
    prompt = (getattr(q, "prompt", "") or "").strip()
    if not prompt:
        return None
    norm = normalize(prompt)
    if not norm or norm in seen_norms:
        return None

    out = {"type": qtype, "difficulty": _valid_difficulty(q.difficulty), "prompt": prompt}
    if qtype == "mc":
        choices = [c.strip() for c in (q.choices or []) if c.strip()]
        if len(choices) < 3 or not (0 <= q.answer_idx < len(choices)) or not q.explanation.strip():
            return None
        out.update(choices=choices, answer_idx=q.answer_idx, explanation=q.explanation.strip())
    elif qtype == "fill_blank":
        accepted = [a.strip() for a in (q.accepted_answers or []) if a.strip()]
        if not accepted:
            return None
        out.update(accepted_answers=accepted, explanation=(q.explanation or "").strip())
    elif qtype == "match":
        pairs = [{"left": p.left.strip(), "right": p.right.strip()}
                 for p in (q.pairs or []) if p.left.strip() and p.right.strip()]
        rights = [p["right"].lower() for p in pairs]
        if not (2 <= len(pairs) <= 6) or len(set(rights)) != len(rights):  # distinct rights
            return None
        out.update(pairs=pairs, explanation=(q.explanation or "").strip())
    else:  # short / freetext
        rubric = [r.strip() for r in (q.rubric or []) if r.strip()]
        if not q.model_answer.strip() or not rubric:
            return None
        out.update(model_answer=q.model_answer.strip(), rubric=rubric)

    seen_norms.add(norm)
    return out


def _next_id_start(existing: List[dict]) -> int:
    """Highest q<N> id in the existing bank, plus one (1 if none)."""
    max_idx = 0
    for q in existing or []:
        qid = str(q.get("id", ""))
        if qid.startswith("q") and qid[1:].isdigit():
            max_idx = max(max_idx, int(qid[1:]))
    return max_idx + 1


def generate_bank(material: str, llm, *, targets: Optional[dict] = None,
                  existing: Optional[List[dict]] = None, batch_size: int = DEFAULT_BATCH,
                  diff_split: Optional[dict] = None, system: str = DEFAULT_SYSTEM,
                  coverage: str = DEFAULT_COVERAGE, max_batches_per_format: int = 6,
                  id_prefix: str = "q") -> List[dict]:
    """Generate the shortfall needed to bring a bank up to ``targets``.

    Args:
        material: the source text to ground questions in.
        llm: any LangChain-style chat model (sync ``with_structured_output``).
        targets: desired TOTAL count per format (defaults to ``DEFAULT_TARGETS``).
        existing: the current bank (counted by type; prompts are not reused).
        batch_size: questions requested per LLM call.
        diff_split: easy/medium/hard proportions per batch.
        system / coverage: prompt steering (override ``coverage`` for a domain).
        max_batches_per_format: cap on retries when validation rejects items.
        id_prefix: id scheme; ids continue after the highest existing ``<prefix><N>``.

    Returns:
        The NEW questions as dicts (with ``id``/``type``/``difficulty`` + fields).
        Does not mutate ``existing`` and does not include it in the result.
    """
    targets = {**DEFAULT_TARGETS, **(targets or {})}
    diff_split = diff_split or DEFAULT_DIFF_SPLIT
    existing = existing or []

    existing_by_type: dict = {}
    seen_norms: set = set()
    for q in existing:
        existing_by_type.setdefault(q.get("type", "mc"), []).append(q)
        seen_norms.add(normalize(q.get("prompt", "")))

    all_new: List[dict] = []
    for fmt, target in targets.items():
        if fmt not in FORMATS:
            logger.warning("Unknown format %r in targets — skipping", fmt)
            continue
        have = len(existing_by_type.get(fmt, []))
        need = max(0, target - have)
        if need == 0:
            continue
        schema, fmt_desc = FORMATS[fmt]
        prompts_seen = {q["prompt"] for q in existing if q.get("type", "mc") == fmt}
        made = batches = 0
        while made < need and batches < max_batches_per_format:
            batches += 1
            batch_n = min(batch_size, need - made)
            try:
                raw = _gen_batch(llm, schema, fmt_desc, material, batch_n, prompts_seen,
                                 system=system, coverage=coverage, diff_split=diff_split)
            except Exception as exc:  # noqa: BLE001 — skip a failed batch, keep going
                logger.warning("%s batch failed (%s): %s", fmt, type(exc).__name__, str(exc)[:160])
                continue
            for q in raw:
                v = _validate(fmt, q, seen_norms)
                if v:
                    all_new.append(v)
                    prompts_seen.add(v["prompt"])
                    made += 1
                    if made >= need:
                        break
        logger.info("%-11s have %2d / target %2d -> added %2d", fmt, have, target, made)

    # Assign sequential ids continuing after the existing bank (id first).
    start = _next_id_start(existing)
    all_new = [{"id": f"{id_prefix}{i}", **q} for i, q in enumerate(all_new, start=start)]

    if all_new:
        logger.info("Generated %d new questions: %s | difficulty %s", len(all_new),
                    dict(Counter(q["type"] for q in all_new)),
                    dict(Counter(q["difficulty"] for q in all_new)))
    return all_new
