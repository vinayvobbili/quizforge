"""Grading — deterministic where we can, LLM where we must.

Multiple-choice is graded by the caller (it's a trivial index compare).
Fill-in-the-blank and match-the-following are graded deterministically here
(no LLM, fast and free). Open-ended answers are scored 0..1 with feedback by an
injected chat model against the question's model answer + rubric.
"""

import logging
from typing import List, Optional

from pydantic import BaseModel, Field

from .llm import structured_output
from .text import is_typo_match, normalize

logger = logging.getLogger("quizforge")

# Points a question is worth by difficulty, so harder questions count for more
# than the trivial recall ones. A test's score is the sum of earned points over
# the sum of available points, so percentages (and pass thresholds) still work.
DIFFICULTY_WEIGHTS = {"easy": 1, "medium": 2, "hard": 3}


def difficulty_weight(question: dict, default: str = "medium") -> int:
    """Point value for ``question`` based on its ``difficulty`` (defaults medium=2)."""
    d = question.get("difficulty") or default
    return DIFFICULTY_WEIGHTS.get(d, DIFFICULTY_WEIGHTS[default])


def grade_fill_blank(question: dict, user_answer: str) -> dict:
    """Deterministic grade for a fill-in-the-blank against accepted_answers.

    Binary credit: 1.0 if the normalized answer matches any accepted answer
    exactly, or is within a small typo of one (so 'priviliges' still earns credit
    for 'privileges'). The ``fuzzy`` flag reports when a typo was forgiven.
    """
    accepted = question.get("accepted_answers") or []
    ua = normalize(user_answer)
    exact = bool(ua) and ua in {normalize(a) for a in accepted}
    fuzzy = bool(ua) and not exact and any(is_typo_match(ua, a) for a in accepted)
    correct = exact or fuzzy
    return {"score": 1.0 if correct else 0.0, "correct": correct,
            "accepted": accepted, "fuzzy": fuzzy}


def grade_match(question: dict, selections: dict) -> dict:
    """Deterministic grade for match-the-following with per-pair partial credit.

    ``selections`` maps the stringified left-item index -> the right-side label
    chosen. Score = fraction of rows matched correctly.
    """
    pairs = question.get("pairs") or []
    total = len(pairs)
    if not total:
        return {"score": 0.0, "correct_count": 0, "total": 0, "rows": []}
    rows = []
    correct_count = 0
    for i, pair in enumerate(pairs):
        chosen = selections.get(str(i), "")
        is_ok = normalize(chosen) == normalize(pair.get("right", ""))
        if is_ok:
            correct_count += 1
        rows.append({
            "left": pair.get("left", ""),
            "right": pair.get("right", ""),
            "chosen": chosen,
            "correct": is_ok,
        })
    return {"score": correct_count / total, "correct_count": correct_count,
            "total": total, "rows": rows}


class QuizGrade(BaseModel):
    """The model's read on an open-ended answer."""

    score: float = Field(description="Credit awarded, 0.0 to 1.0. 1.0 = covers the key points "
                         "correctly; ~0.5 = partially correct or missing a key idea; 0.0 = wrong, "
                         "empty, or just restates the question.")
    verdict: str = Field(description="One of: 'correct', 'partial', 'incorrect'")
    feedback: str = Field(description="2-3 sentences, encouraging coach tone. Say what was right, "
                          "then the single most important thing to add or fix.")
    covered: List[str] = Field(description="Key points the answer got right (short phrases). Empty if none.")
    missed: List[str] = Field(description="Key points the answer missed or got wrong (short phrases). Empty if none.")


DEFAULT_GRADE_SYSTEM = (
    "You are a training assessor grading a learner's open-ended quiz answer. Grade on "
    "CONCEPTS, not phrasing or grammar — credit the learner when they demonstrate the right "
    "understanding in their own words. Be fair but rigorous: award partial credit when an "
    "answer is on the right track but misses a key point, and award 0 when it is wrong, empty, "
    "evasive ('I don't know'), or merely restates the question. Do not be fooled by "
    "confident-sounding but incorrect answers. Return strict JSON for the QuizGrade schema."
)


def _open_prompt(question: dict, user_answer: str) -> str:
    parts = [
        f"QUESTION:\n{question['prompt']}",
        "",
        f"MODEL ANSWER (what a strong response covers):\n"
        f"{(question.get('model_answer') or '').strip() or '(none provided)'}",
    ]
    rubric = question.get("rubric") or []
    if rubric:
        parts += ["", "RUBRIC — key points to look for:"]
        parts += [f"  - {point}" for point in rubric]
    parts += ["", f"LEARNER'S ANSWER:\n{(user_answer or '').strip()}",
              "", "Grade the learner's answer. score is fractional 0..1."]
    return "\n".join(parts)


def grade_open_answer(question: dict, user_answer: str, llm,
                      system: str = DEFAULT_GRADE_SYSTEM) -> Optional[QuizGrade]:
    """Score an open-ended answer 0..1 with feedback using the injected ``llm``.

    Returns a :class:`QuizGrade`, or ``None`` if automated grading was
    unavailable (callers should exclude the question from the attempt's max
    score rather than penalize the learner for an outage).
    """
    if not (user_answer or "").strip():
        return QuizGrade(score=0.0, verdict="incorrect", feedback="No answer was provided.",
                         covered=[], missed=list(question.get("rubric") or []))
    try:
        chain = structured_output(llm, QuizGrade)
        grade = chain.invoke([
            ("system", system),
            ("human", _open_prompt(question, user_answer)),
        ])
    except Exception as exc:  # noqa: BLE001 — degrade gracefully, never crash the quiz
        logger.warning("Open-answer grading failed for %s: %s: %s",
                       question.get("id", "?"), type(exc).__name__, str(exc)[:200])
        return None
    grade.score = max(0.0, min(1.0, float(grade.score)))
    return grade
