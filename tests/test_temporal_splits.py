import numpy as np
import pandas as pd

from src.models.train_churn import (
    chronological_partition_indices,
    make_date_grouped_splits,
)


def sample_dates():
    rows = []
    for month in pd.date_range("2017-01-01", periods=20, freq="MS"):
        rows.extend([month, month, month])
    return pd.Series(rows)


def test_partitions_are_chronological_and_keep_dates_together():
    dates = sample_dates()
    train, validation, test, periods = chronological_partition_indices(dates)

    train_dates = pd.to_datetime(dates.iloc[train]).dt.normalize()
    validation_dates = pd.to_datetime(dates.iloc[validation]).dt.normalize()
    test_dates = pd.to_datetime(dates.iloc[test]).dt.normalize()

    assert train_dates.max() + pd.Timedelta(days=90) <= validation_dates.min()
    assert validation_dates.max() + pd.Timedelta(days=90) <= test_dates.min()
    assert periods["train_start"] < periods["validation_start"] < periods["test_start"]

    membership = {}
    for name, indices in {
        "train": train, "validation": validation, "test": test
    }.items():
        for date in pd.to_datetime(dates.iloc[indices]).dt.normalize().unique():
            membership.setdefault(date, set()).add(name)
    assert all(len(names) == 1 for names in membership.values())


def test_cross_validation_splits_unique_dates_not_rows():
    dates = sample_dates()
    for train, validation in make_date_grouped_splits(dates, n_splits=4):
        train_dates = pd.to_datetime(dates.iloc[train]).dt.normalize().unique()
        validation_dates = pd.to_datetime(dates.iloc[validation]).dt.normalize().unique()
        assert len(np.intersect1d(train_dates, validation_dates)) == 0
        assert train_dates.max() + pd.Timedelta(days=90) <= validation_dates.min()


def test_multiple_rows_from_one_snapshot_date_are_never_split():
    dates = sample_dates()
    for train, validation in make_date_grouped_splits(dates, n_splits=4):
        train_dates = set(pd.to_datetime(dates.iloc[train]).dt.normalize())
        validation_dates = set(pd.to_datetime(dates.iloc[validation]).dt.normalize())
        assert train_dates.isdisjoint(validation_dates)


def test_class_aware_cv_folds_have_both_classes_in_train_and_validation():
    dates = sample_dates()
    labels = []
    for month_index in range(20):
        labels.extend([month_index % 2, (month_index + 1) % 2, 1])
    folds = make_date_grouped_splits(dates, y=pd.Series(labels), n_splits=4)

    for train, validation in folds:
        assert set(pd.Series(labels).iloc[train].unique()) == {0, 1}
        assert set(pd.Series(labels).iloc[validation].unique()) == {0, 1}


def _realistic_sparse_repeat_purchase_history():
    months = pd.date_range("2016-11-01", "2018-07-01", freq="MS")
    repeat_counts = {
        "2017-03-01": 13,
        "2017-04-01": 37,
        "2017-05-01": 49,
        "2017-06-01": 75,
        "2017-07-01": 106,
        "2017-08-01": 119,
        "2017-09-01": 167,
        "2017-10-01": 182,
        "2017-11-01": 206,
        "2017-12-01": 187,
        "2018-01-01": 205,
        "2018-02-01": 214,
        "2018-03-01": 241,
        "2018-04-01": 246,
        "2018-05-01": 242,
        "2018-06-01": 235,
        "2018-07-01": 159,
    }
    dates = []
    labels = []
    for month in months:
        key = str(month.date())
        count_repeat = repeat_counts.get(key, 0)
        count_churn = 100 if count_repeat else 20
        dates.extend([month] * (count_repeat + count_churn))
        labels.extend([0] * count_repeat + [1] * count_churn)
    return pd.Series(dates), pd.Series(labels)


def test_sparse_repeat_purchase_history_has_valid_purged_temporal_cv():
    dates, y_churned = _realistic_sparse_repeat_purchase_history()
    train, validation, test, periods = chronological_partition_indices(dates)

    y_train = y_churned.iloc[train].reset_index(drop=True)
    y_validation = y_churned.iloc[validation].reset_index(drop=True)
    y_test = y_churned.iloc[test].reset_index(drop=True)

    assert (y_train == 0).sum() > 0
    assert (y_validation == 0).sum() > 0
    assert (y_test == 0).sum() > 0

    train_dates = dates.iloc[train].reset_index(drop=True)
    folds = make_date_grouped_splits(
        train_dates,
        y=y_train,
        n_splits=min(4, train_dates.nunique() - 5),
    )
    assert len(folds) >= 2
    for fit_idx, validation_idx in folds:
        assert set(y_train.iloc[fit_idx].unique()) == {0, 1}
        assert set(y_train.iloc[validation_idx].unique()) == {0, 1}
        fit_dates = train_dates.iloc[fit_idx]
        fold_validation_dates = train_dates.iloc[validation_idx]
        assert fit_dates.max() + pd.Timedelta(days=90) <= fold_validation_dates.min()



def test_sparse_repeat_cv_can_require_minimum_positive_event_support():
    dates, y_churned = _realistic_sparse_repeat_purchase_history()
    train, _, _, _ = chronological_partition_indices(dates)
    train_dates = dates.iloc[train].reset_index(drop=True)
    y_train = y_churned.iloc[train].reset_index(drop=True)

    folds = make_date_grouped_splits(
        train_dates,
        y=y_train,
        n_splits=min(4, train_dates.nunique() - 5),
        min_train_repeat_events=10,
        min_validation_repeat_events=5,
    )

    for fit_idx, validation_idx in folds:
        assert int((y_train.iloc[fit_idx] == 0).sum()) >= 10
        assert int((y_train.iloc[validation_idx] == 0).sum()) >= 5
        fit_dates = train_dates.iloc[fit_idx]
        validation_dates = train_dates.iloc[validation_idx]
        assert fit_dates.max() + pd.Timedelta(days=90) <= validation_dates.min()
