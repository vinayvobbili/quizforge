"""Certificate: eligibility, level, and a stable tamper-evident code."""

import pytest

from quizforge import Certificate, make_certificate, verification_code, verify
from quizforge.certificate import is_eligible, level_for


def _cert(score=92, **kw):
    args = dict(learner_id="jane.doe@example.com", learner_name="Jane Doe", topic_id="citrix",
                topic_title="Citrix", score_pct=score, awarded_on="06/28/2026")
    args.update(kw)
    return make_certificate(**args)


def test_eligibility_and_level():
    assert not is_eligible(55) and is_eligible(60)
    assert level_for(60) == "passed"
    assert level_for(80) == "distinction"


def test_make_certificate_distinction():
    cert = _cert(92)
    assert cert.level == "distinction"
    assert cert.verification_code.startswith("QF-")
    assert verify(cert)


def test_make_certificate_pass():
    assert _cert(70).level == "passed"


def test_below_pass_raises():
    with pytest.raises(ValueError):
        _cert(40)


def test_code_is_deterministic_and_field_sensitive():
    a = verification_code(learner_id="x@e.com", topic_id="citrix", awarded_on="06/28/2026",
                          level="passed", score_pct=70)
    b = verification_code(learner_id="x@e.com", topic_id="citrix", awarded_on="06/28/2026",
                          level="passed", score_pct=70)
    c = verification_code(learner_id="x@e.com", topic_id="citrix", awarded_on="06/28/2026",
                          level="passed", score_pct=71)  # one field changed
    assert a == b and a != c


def test_learner_id_is_case_insensitive():
    a = verification_code(learner_id="X@E.com", topic_id="t", awarded_on="d", level="passed", score_pct=60)
    b = verification_code(learner_id="x@e.com", topic_id="t", awarded_on="d", level="passed", score_pct=60)
    assert a == b


def test_secret_changes_code_and_tamper_is_detected():
    cert = _cert(92, secret="house-key")
    assert verify(cert, secret="house-key")
    assert not verify(cert)  # wrong/no secret
    # tampering with a field invalidates the code
    tampered = Certificate(**{**cert.model_dump(), "score_pct": 100})
    assert not verify(tampered, secret="house-key")
