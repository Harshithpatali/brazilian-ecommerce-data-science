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
    # Repeat customers/snapshots across dates; both labels appear throughout.
    labels = []
    for month_index in range(20):
        labels.extend([month_index % 2, (month_index + 1) % 2, 1])
    folds = make_date_grouped_splits(dates, y=pd.Series(labels), n_splits=4)

    for train, validation in folds:
        assert set(pd.Series(labels).iloc[train].unique()) == {0, 1}
        assert set(pd.Series(labels).iloc[validation].unique()) == {0, 1}
