"""Integrity: fast-pass detection is conservative and only flags passes."""

from quizforge import assess_speed, expected_min_seconds, within_time_limit
from quizforge.integrity import DEFAULT_MIN_SECONDS_PER_TYPE, DEFAULT_TIME_LIMIT_SECONDS

# A representative 20-question test (the default blueprint shape).
TEST_TYPES = ["mc"] * 8 + ["fill_blank"] * 4 + ["match"] * 2 + ["short"] * 4 + ["freetext"] * 2


def test_expected_min_seconds_sums_per_type():
    assert expected_min_seconds(["mc", "mc"]) == 2 * DEFAULT_MIN_SECONDS_PER_TYPE["mc"]
    # unknown format falls back
    assert expected_min_seconds(["mystery"], fallback=9.0) == 9.0


def test_blazing_fast_pass_is_high_severity():
    flag = assess_speed(elapsed_seconds=30, question_types=TEST_TYPES, passed=True)
    assert flag.suspicious and flag.severity == "high"
    assert flag.speed_ratio < 0.25
    assert flag.reasons


def test_moderately_fast_pass_is_low_severity():
    expected = expected_min_seconds(TEST_TYPES)
    # ~40% of the realistic floor -> below low_ratio (0.5), above high_ratio (0.25)
    flag = assess_speed(elapsed_seconds=int(expected * 0.4), question_types=TEST_TYPES, passed=True)
    assert flag.suspicious and flag.severity == "low"


def test_genuine_pace_pass_is_clean():
    expected = expected_min_seconds(TEST_TYPES)
    flag = assess_speed(elapsed_seconds=int(expected * 1.5), question_types=TEST_TYPES, passed=True)
    assert not flag.suspicious and flag.severity == "none"


def test_fast_but_failed_is_not_flagged():
    flag = assess_speed(elapsed_seconds=10, question_types=TEST_TYPES, passed=False)
    assert not flag.suspicious and flag.severity == "none"


def test_missing_timing_is_never_flagged():
    # Legacy rows with no recorded elapsed time must not all read as instant cheats.
    flag = assess_speed(elapsed_seconds=0, question_types=TEST_TYPES, passed=True)
    assert not flag.suspicious and flag.severity == "none"
    assert "No timing" in flag.reasons[0]


def test_thresholds_are_tunable():
    # With a strict low_ratio, an otherwise-clean pace becomes a low flag.
    expected = expected_min_seconds(TEST_TYPES)
    flag = assess_speed(elapsed_seconds=int(expected * 0.9), question_types=TEST_TYPES,
                        passed=True, low_ratio=1.0, high_ratio=0.25)
    assert flag.severity == "low"


def test_within_time_limit_default_is_30_minutes():
    assert DEFAULT_TIME_LIMIT_SECONDS == 1800
    assert within_time_limit(1799) is True
    assert within_time_limit(1800) is True          # exactly at the budget is OK
    assert within_time_limit(1801) is False         # one second over


def test_within_time_limit_untimed_and_no_limit_pass():
    # Legacy / untimed attempts are not violations, and a non-positive limit
    # means "no limit" and always passes.
    assert within_time_limit(0) is True
    assert within_time_limit(-5) is True
    assert within_time_limit(99999, limit_seconds=0) is True


def test_within_time_limit_custom_budget():
    assert within_time_limit(120, limit_seconds=300) is True
    assert within_time_limit(600, limit_seconds=300) is False
