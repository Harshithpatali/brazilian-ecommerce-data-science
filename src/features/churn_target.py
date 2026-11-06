from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def is_churned_in_forward_window(
    snapshot_date: str | pd.Timestamp,
    future_delivered_purchase_timestamps: Iterable[str | pd.Timestamp],
    horizon_days: int = 90,
) -> int:
    """Reference contract for the SQL half-open forward prediction window."""
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
    snapshot = pd.Timestamp(snapshot_date)
    if pd.isna(snapshot):
        raise ValueError("snapshot_date must be valid.")
    end = snapshot + pd.Timedelta(days=horizon_days)
    for value in future_delivered_purchase_timestamps:
        if value is None or pd.isna(value):
            continue
        purchase = pd.Timestamp(value)
        if snapshot <= purchase < end:
            return 0
    return 1
