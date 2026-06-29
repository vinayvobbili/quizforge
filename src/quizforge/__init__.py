"""quizforge — generate a deep, mixed-format question bank from source material
and grade it. Deterministic where it can, LLM where it must. Bring your own
chat model.

Public API
----------
Generate:
    generate_bank(material, llm, *, targets=..., existing=...) -> list[dict]
Sample:
    sample_test(questions, blueprint=..., seen_ids=...) -> list[dict]
    pick_spread(pool, want, seen_ids) -> list[dict]
    DEFAULT_BLUEPRINT
Grade:
    grade_fill_blank(question, user_answer) -> dict
    grade_match(question, selections) -> dict
    grade_open_answer(question, user_answer, llm) -> QuizGrade | None
    QuizGrade
Integrity:
    assess_speed(*, elapsed_seconds, question_types, passed) -> IntegrityFlag
    expected_min_seconds(question_types) -> float
    within_time_limit(elapsed_seconds, limit_seconds=...) -> bool
    IntegrityFlag
Certificate:
    make_certificate(*, learner_id, ..., score_pct, awarded_on) -> Certificate
    verification_code(...) -> str   |   verify(certificate) -> bool
    Certificate
Utilities:
    normalize(text) -> str
    structured_output(llm, schema) -> chain
"""

from .bank import Bank
from .certificate import (
    Certificate,
    is_eligible,
    level_for,
    make_certificate,
    verification_code,
    verify,
)
from .generate import (
    DEFAULT_COVERAGE,
    DEFAULT_DIFF_SPLIT,
    DEFAULT_SYSTEM,
    DEFAULT_TARGETS,
    generate_bank,
)
from .grade import (
    DEFAULT_GRADE_SYSTEM,
    DIFFICULTY_WEIGHTS,
    QuizGrade,
    difficulty_weight,
    grade_fill_blank,
    grade_match,
    grade_open_answer,
)
from .integrity import (
    DEFAULT_MIN_SECONDS_PER_TYPE,
    DEFAULT_TIME_LIMIT_SECONDS,
    IntegrityFlag,
    assess_speed,
    expected_min_seconds,
    within_time_limit,
)
from .llm import structured_output
from .sample import DEFAULT_BLUEPRINT, DIFFICULTY_ORDER, pick_spread, sample_test
from .schemas import DIFFICULTIES, FORMAT_KEYS, FORMATS
from .text import edit_distance, is_typo_match, normalize

__version__ = "0.3.0"

__all__ = [
    "Bank",
    "generate_bank",
    "sample_test",
    "pick_spread",
    "grade_fill_blank",
    "grade_match",
    "grade_open_answer",
    "difficulty_weight",
    "DIFFICULTY_WEIGHTS",
    "QuizGrade",
    "assess_speed",
    "expected_min_seconds",
    "within_time_limit",
    "IntegrityFlag",
    "DEFAULT_MIN_SECONDS_PER_TYPE",
    "DEFAULT_TIME_LIMIT_SECONDS",
    "Certificate",
    "make_certificate",
    "verification_code",
    "verify",
    "is_eligible",
    "level_for",
    "normalize",
    "is_typo_match",
    "edit_distance",
    "structured_output",
    "DEFAULT_BLUEPRINT",
    "DEFAULT_TARGETS",
    "DEFAULT_DIFF_SPLIT",
    "DEFAULT_SYSTEM",
    "DEFAULT_COVERAGE",
    "DEFAULT_GRADE_SYSTEM",
    "DIFFICULTY_ORDER",
    "DIFFICULTIES",
    "FORMAT_KEYS",
    "FORMATS",
    "__version__",
]
