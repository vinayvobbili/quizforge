"""Integrity signals — flag attempts that are too fast to be genuine.

A deep, unseen-first bank (see :mod:`quizforge.sample`) already makes
answer-sharing hard. This adds a cheap, deterministic backstop: an attempt that
*passed* in less time than it physically takes to read and answer the questions
is suspicious — it points at a leaked answer key, a shared screenshot, or
automation, not knowledge.

This is intentionally a dumb arithmetic heuristic, not an LLM call: "is 40s
plausible for 20 questions including two written answers" is a timing question,
and an LLM adds nothing to it. The per-format floors and thresholds are tunable
so a caller can dial sensitivity to their own population.
"""

from typing import Iterable, List, Optional

from pydantic import BaseModel, Field

# Minimum *realistic* seconds to genuinely read + answer one question of each
# format — not an absolute physical floor, but the point below which engagement
# is implausible. Open formats dominate because a passing written answer takes
# real time to compose. Deliberately generous to keep false positives low.
DEFAULT_MIN_SECONDS_PER_TYPE = {
    "mc": 5.0,
    "fill_blank": 7.0,
    "match": 10.0,
    "short": 20.0,
    "freetext": 40.0,
}
# Seconds assumed for a format not in the table (unknown / custom).
DEFAULT_FALLBACK_SECONDS = 6.0


class IntegrityFlag(BaseModel):
    """The verdict on a single attempt's timing."""

    suspicious: bool = Field(description="True if the pass was implausibly fast.")
    severity: str = Field(description="One of: 'none', 'low', 'high'.")
    elapsed_seconds: int = Field(description="Seconds the learner actually took.")
    expected_seconds: float = Field(description="Realistic minimum for these questions.")
    speed_ratio: float = Field(description="elapsed / expected; <1 is faster than realistic.")
    reasons: List[str] = Field(default_factory=list, description="Human-readable explanation.")


def expected_min_seconds(question_types: Iterable[str], per_type: Optional[dict] = None,
                         fallback: float = DEFAULT_FALLBACK_SECONDS) -> float:
    """Sum the per-format realistic-minimum seconds for a set of question types."""
    table = {**DEFAULT_MIN_SECONDS_PER_TYPE, **(per_type or {})}
    return sum(table.get(t, fallback) for t in question_types)


def assess_speed(*, elapsed_seconds: int, question_types: Iterable[str], passed: bool,
                 per_type: Optional[dict] = None, fallback: float = DEFAULT_FALLBACK_SECONDS,
                 low_ratio: float = 0.5, high_ratio: float = 0.25) -> IntegrityFlag:
    """Flag a *passing* attempt completed implausibly fast.

    Only passes are assessed — a fast *fail* is someone giving up, not cheating.
    Attempts with no recorded timing (``elapsed_seconds <= 0``, e.g. legacy rows)
    are never flagged: absence of a signal is not evidence.

    Args:
        elapsed_seconds: wall-clock seconds the attempt took.
        question_types: the format of each question in the attempt (e.g.
            ``["mc", "mc", "freetext", ...]``).
        passed: whether the attempt passed (only passes are flagged).
        per_type: override realistic-minimum seconds per format.
        low_ratio / high_ratio: speed_ratio thresholds for 'low' / 'high'
            severity (lower ratio = faster = more suspicious).

    Returns:
        An :class:`IntegrityFlag`. ``severity`` is 'high' when
        ``speed_ratio < high_ratio``, 'low' when ``< low_ratio``, else 'none'.
    """
    types = list(question_types)
    expected = expected_min_seconds(types, per_type, fallback)

    if elapsed_seconds <= 0:
        return IntegrityFlag(suspicious=False, severity="none", elapsed_seconds=int(elapsed_seconds),
                             expected_seconds=round(expected, 1), speed_ratio=0.0,
                             reasons=["No timing was recorded for this attempt."])
    if not passed:
        return IntegrityFlag(suspicious=False, severity="none", elapsed_seconds=int(elapsed_seconds),
                             expected_seconds=round(expected, 1),
                             speed_ratio=round(elapsed_seconds / expected, 2) if expected else 0.0,
                             reasons=["Attempt did not pass — timing not assessed."])

    ratio = (elapsed_seconds / expected) if expected > 0 else 1.0
    if ratio < high_ratio:
        severity = "high"
    elif ratio < low_ratio:
        severity = "low"
    else:
        severity = "none"

    reasons: List[str] = []
    if severity != "none":
        reasons.append(
            f"Passed in {int(elapsed_seconds)}s — {round(ratio * 100)}% of the "
            f"~{round(expected)}s it realistically takes to read and answer "
            f"{len(types)} questions."
        )
    return IntegrityFlag(suspicious=severity != "none", severity=severity,
                         elapsed_seconds=int(elapsed_seconds), expected_seconds=round(expected, 1),
                         speed_ratio=round(ratio, 2), reasons=reasons)
