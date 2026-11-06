import pandas as pd
import pytest

from src.features.churn_target import is_churned_in_forward_window


def test_purchase_inside_90_day_window_means_not_churned():
    snapshot = pd.Timestamp("2018-01-01")
    assert is_churned_in_forward_window(
        snapshot, [snapshot + pd.Timedelta(days=89, hours=23)]
    ) == 0


def test_purchase_on_snapshot_date_is_inside_window():
    assert is_churned_in_forward_window(
        "2018-01-01", ["2018-01-01 08:00:00"]
    ) == 0


def test_purchase_exactly_at_90_day_boundary_is_outside_window():
    snapshot = pd.Timestamp("2018-01-01")
    assert is_churned_in_forward_window(
        snapshot, [snapshot + pd.Timedelta(days=90)]
    ) == 1


def test_purchase_before_snapshot_does_not_prevent_churn_label():
    assert is_churned_in_forward_window(
        "2018-01-01", ["2017-12-31 23:59:59"]
    ) == 1


def test_no_future_purchase_means_churned():
    assert is_churned_in_forward_window("2018-01-01", []) == 1


def test_null_future_timestamps_are_ignored():
    assert is_churned_in_forward_window("2018-01-01", [None, pd.NaT]) == 1


def test_horizon_must_be_positive():
    with pytest.raises(ValueError):
        is_churned_in_forward_window("2018-01-01", [], horizon_days=0)
