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



def test_full_olist_population_passes_v5_contract():
    from scripts.validate_churn_data import assert_minimum_dataset_population

    assert_minimum_dataset_population({
        "rows": 357413,
        "customers": 79594,
        "repeat_purchase_rows": 2686,
    })


def test_truncated_second_or_third_purchase_cohort_fails_v5_contract():
    from scripts.validate_churn_data import assert_minimum_dataset_population

    with pytest.raises(SystemExit, match="full Olist v5 contract"):
        assert_minimum_dataset_population({
            "rows": 10570,
            "customers": 2374,
            "repeat_purchase_rows": 306,
        })


def test_population_contract_requires_all_counts():
    from scripts.validate_churn_data import assert_minimum_dataset_population

    with pytest.raises(SystemExit, match="repeat_purchase_rows"):
        assert_minimum_dataset_population({
            "rows": 357413,
            "customers": 79594,
        })
