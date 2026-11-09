from __future__ import annotations

import numpy as np
import pandas as pd


def build_rfm(customers: pd.DataFrame, as_of: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Build transparent customer-level RFM scores from delivered-customer aggregates."""
    required = {"customer_unique_id", "orders", "first_order_date", "last_order_date", "revenue"}
    missing = required.difference(customers.columns)
    if missing:
        raise ValueError(f"Customer metrics missing columns: {sorted(missing)}")
    d = customers.copy()
    d["first_order_date"] = pd.to_datetime(d["first_order_date"], errors="coerce")
    d["last_order_date"] = pd.to_datetime(d["last_order_date"], errors="coerce")
    d["orders"] = pd.to_numeric(d["orders"], errors="coerce").fillna(0).clip(lower=0)
    d["revenue"] = pd.to_numeric(d["revenue"], errors="coerce").fillna(0).clip(lower=0)
    cutoff = pd.Timestamp(as_of) if as_of is not None else d["last_order_date"].max()
    if pd.isna(cutoff):
        raise ValueError("No valid customer order dates available.")
    d["recency_days"] = (cutoff.normalize() - d["last_order_date"].dt.normalize()).dt.days.clip(lower=0)
    d["frequency"] = d["orders"]
    d["monetary"] = d["revenue"]
    # Rank-based quintiles are stable when the underlying data has many ties.
    d["R_score"] = (6 - d["recency_days"].rank(method="average", pct=True).mul(5).apply(np.ceil)).clip(1, 5).astype(int)
    d["F_score"] = (d["frequency"].rank(method="average", pct=True).mul(5).apply(np.ceil)).clip(1, 5).astype(int)
    d["M_score"] = (d["monetary"].rank(method="average", pct=True).mul(5).apply(np.ceil)).clip(1, 5).astype(int)
    d["RFM_score"] = d[["R_score", "F_score", "M_score"]].sum(axis=1)
    conditions = [
        (d["R_score"] >= 4) & (d["F_score"] >= 4) & (d["M_score"] >= 4),
        (d["R_score"] >= 4) & (d["F_score"] <= 2),
        (d["R_score"] <= 2) & (d["F_score"] >= 3),
        (d["R_score"] <= 2) & (d["F_score"] <= 2),
    ]
    labels = ["Champions", "New / Promising", "At Risk", "Hibernating"]
    d["rfm_segment"] = np.select(conditions, labels, default="Loyal / Developing")
    d["customer_explanation"] = d.apply(_explain_customer, axis=1)
    return d.sort_values(["RFM_score", "monetary"], ascending=False).reset_index(drop=True)


def _explain_customer(row: pd.Series) -> str:
    parts = [
        f"Recency: last delivered order was {int(row['recency_days'])} days before the analysis cutoff (R{int(row['R_score'])}/5).",
        f"Frequency: {int(row['frequency'])} delivered order(s) (F{int(row['F_score'])}/5).",
        f"Monetary: R$ {float(row['monetary']):,.2f} cumulative revenue (M{int(row['M_score'])}/5).",
    ]
    if row["frequency"] <= 1:
        parts.append("Only one delivered order is observed; repeat behaviour is not yet established.")
    elif row["recency_days"] > 180:
        parts.append("Long inactivity suggests a reactivation review.")
    else:
        parts.append("Repeat purchasing is observed; consider retention or loyalty treatment.")
    return " ".join(parts)


def kaplan_meier(customers: pd.DataFrame) -> pd.DataFrame:
    """KM estimate for first-to-second delivered purchase; customers without event are right-censored at data end."""
    required = {"orders", "first_order_date", "second_order_date", "last_order_date"}
    missing = required.difference(customers.columns)
    if missing:
        raise ValueError(
            f"Customer metrics missing {sorted(missing)}. Rebuild analytics.customer_metrics "
            "to include the exact second delivered purchase date."
        )
    d = customers.copy()
    for col in ["first_order_date", "second_order_date", "last_order_date"]:
        d[col] = pd.to_datetime(d[col], errors="coerce")
    d["orders"] = pd.to_numeric(d["orders"], errors="coerce").fillna(0)
    d = d.dropna(subset=["first_order_date", "last_order_date"])
    if d.empty:
        return pd.DataFrame(columns=["day", "at_risk", "events", "censored", "survival", "repeat_purchase_cumulative"])
    data_end = d["last_order_date"].max().normalize()
    observed = d["second_order_date"].notna() & (d["orders"] >= 2)
    event_duration = (d["second_order_date"] - d["first_order_date"]).dt.total_seconds() / 86400
    censor_duration = (data_end - d["first_order_date"]).dt.total_seconds() / 86400
    duration = np.where(observed, event_duration, censor_duration)
    d["duration_days"] = np.maximum(np.floor(np.nan_to_num(duration, nan=0)).astype(int), 0)
    d["event"] = observed.astype(int)
    rows = []
    survival = 1.0
    for day in sorted(d["duration_days"].unique()):
        at_risk = int((d["duration_days"] >= day).sum())
        events = int(((d["duration_days"] == day) & (d["event"] == 1)).sum())
        censored = int(((d["duration_days"] == day) & (d["event"] == 0)).sum())
        if at_risk and events:
            survival *= 1 - events / at_risk
        rows.append({"day": int(day), "at_risk": at_risk, "events": events,
                     "censored": censored, "survival": survival,
                     "repeat_purchase_cumulative": 1 - survival})
    return pd.DataFrame(rows)


def campaign_scenario(
    customers: pd.DataFrame,
    segment: str,
    contact_fraction: float,
    lift_pp: float,
    contact_cost: float,
    contribution_margin: float,
) -> dict:
    """Assumption-led scenario calculator; lift_pp is not estimated from observational Olist data."""
    if not 0 < contact_fraction <= 1:
        raise ValueError("contact_fraction must be in (0, 1].")
    if lift_pp < 0 or contact_cost < 0 or not 0 <= contribution_margin <= 1:
        raise ValueError("Lift/cost/margin assumptions are outside valid ranges.")
    d = customers.copy()
    if "rfm_segment" not in d:
        d = build_rfm(d)
    pool = d if segment == "All customers" else d[d["rfm_segment"] == segment]
    contacted = int(np.ceil(len(pool) * contact_fraction))
    baseline = float((pool["orders"] >= 2).mean()) if len(pool) else 0.0
    aov = float(pool["aov"].median()) if "aov" in pool and len(pool) else (
        float(pool["revenue"].sum() / max(pool["orders"].sum(), 1)) if len(pool) else 0.0
    )
    expected_baseline = contacted * baseline
    incremental = contacted * (lift_pp / 100.0)
    expected_incremental_revenue = incremental * aov
    gross_contribution = expected_incremental_revenue * contribution_margin
    campaign_cost = contacted * contact_cost
    return {
        "eligible_customers": int(len(pool)), "contacted_customers": contacted,
        "historical_repeat_rate": baseline, "assumed_incremental_lift_pp": lift_pp,
        "expected_baseline_repeaters": expected_baseline,
        "assumed_incremental_repeaters": incremental, "median_aov": aov,
        "expected_incremental_revenue": expected_incremental_revenue,
        "gross_contribution": gross_contribution, "campaign_cost": campaign_cost,
        "scenario_net_contribution": gross_contribution - campaign_cost,
        "break_even_lift_pp": (campaign_cost / (contacted * aov * contribution_margin) * 100)
            if contacted and aov > 0 and contribution_margin > 0 else None,
    }
