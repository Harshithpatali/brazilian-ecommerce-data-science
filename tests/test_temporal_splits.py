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

    assert train_dates.max() < validation_dates.min()
    assert validation_dates.max() < test_dates.min()
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
        assert train_dates.max() < validation_dates.min()


def test_multiple_rows_from_one_snapshot_date_are_never_split():
    dates = sample_dates()
    for train, validation in make_date_grouped_splits(dates, n_splits=4):
        train_dates = set(pd.to_datetime(dates.iloc[train]).dt.normalize())
        validation_dates = set(pd.to_datetime(dates.iloc[validation]).dt.normalize())
        assert train_dates.isdisjoint(validation_dates)
