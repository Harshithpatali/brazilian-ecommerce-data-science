import pytest

from scripts.validate_churn_data import (
    EXPECTED_ZERO_CHECKS,
    assert_zero_integrity_checks,
)


def zero_checks():
    return {name: 0 for name in EXPECTED_ZERO_CHECKS}


def test_zero_integrity_counts_pass():
    assert_zero_integrity_checks(zero_checks())


def test_nonzero_integrity_count_fails():
    checks = zero_checks()
    checks["review_feature_mismatches"] = 2
    with pytest.raises(SystemExit, match="review_feature_mismatches"):
        assert_zero_integrity_checks(checks)


def test_missing_integrity_count_fails():
    checks = zero_checks()
    checks.pop("target_mismatches")
    with pytest.raises(SystemExit, match="target_mismatches"):
        assert_zero_integrity_checks(checks)
