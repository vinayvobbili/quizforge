"""Completion certificates — the data + a tamper-evident verification code.

quizforge owns the *facts* of a certificate (who, which topic, what level, when)
and a deterministic verification code derived from them, so any consumer can
re-derive the code to confirm a certificate wasn't altered. Rendering — PDF,
HTML, image — is deliberately left to the consumer, where the branding lives, so
this module stays dependency-light (pydantic only) and clockless (the caller
supplies ``awarded_on``, keeping generation reproducible).
"""

import hashlib
from typing import Optional

from pydantic import BaseModel, Field

PASS = "passed"
DISTINCTION = "distinction"
_SEP = "\x1f"  # unit separator — unlikely to appear in any field


class Certificate(BaseModel):
    """An earned completion certificate's facts + its verification code."""

    learner_id: str = Field(description="Stable learner identity (e.g. email).")
    learner_name: str = Field(description="Display name to print on the certificate.")
    topic_id: str = Field(description="Topic/lesson identifier.")
    topic_title: str = Field(description="Human-readable topic title.")
    score_pct: int = Field(description="Best score as a 0-100 percentage.")
    level: str = Field(description="'passed' or 'distinction'.")
    awarded_on: str = Field(description="Award date as the caller's display string.")
    verification_code: str = Field(description="Deterministic, tamper-evident code.")


def verification_code(*, learner_id: str, topic_id: str, awarded_on: str, level: str,
                      score_pct: int, secret: str = "", prefix: str = "QF") -> str:
    """Derive a short, stable verification code from a certificate's fields.

    Same fields (+ same ``secret``) always yield the same code; changing any
    field changes the code, so a printed certificate can be checked against the
    record. With a non-empty ``secret`` the code is unforgeable without it.
    """
    raw = _SEP.join([learner_id.lower().strip(), topic_id, awarded_on, level,
                     str(score_pct), secret])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest().upper()
    return f"{prefix}-{digest[:4]}-{digest[4:8]}"


def is_eligible(score_pct: int, pass_threshold: float = 0.6) -> bool:
    """Whether a score earns a certificate at all."""
    return score_pct / 100.0 >= pass_threshold


def level_for(score_pct: int, distinction_threshold: float = 0.8) -> str:
    """'distinction' at/above the threshold, else 'passed'."""
    return DISTINCTION if score_pct / 100.0 >= distinction_threshold else PASS


def make_certificate(*, learner_id: str, learner_name: str, topic_id: str, topic_title: str,
                     score_pct: int, awarded_on: str, pass_threshold: float = 0.6,
                     distinction_threshold: float = 0.8, secret: str = "",
                     prefix: str = "QF") -> Certificate:
    """Build a :class:`Certificate` for an eligible score.

    Raises ``ValueError`` if ``score_pct`` is below ``pass_threshold`` — there is
    no certificate for a non-pass.
    """
    if not is_eligible(score_pct, pass_threshold):
        raise ValueError(f"score {score_pct}% is below the pass threshold "
                         f"{round(pass_threshold * 100)}% — no certificate")
    level = level_for(score_pct, distinction_threshold)
    code = verification_code(learner_id=learner_id, topic_id=topic_id, awarded_on=awarded_on,
                             level=level, score_pct=score_pct, secret=secret, prefix=prefix)
    return Certificate(learner_id=learner_id, learner_name=learner_name, topic_id=topic_id,
                       topic_title=topic_title, score_pct=score_pct, level=level,
                       awarded_on=awarded_on, verification_code=code)


def verify(certificate: Certificate, secret: str = "", prefix: str = "QF") -> bool:
    """Re-derive the code from the certificate's fields and confirm it matches."""
    expected = verification_code(
        learner_id=certificate.learner_id, topic_id=certificate.topic_id,
        awarded_on=certificate.awarded_on, level=certificate.level,
        score_pct=certificate.score_pct, secret=secret, prefix=prefix)
    return expected == certificate.verification_code
