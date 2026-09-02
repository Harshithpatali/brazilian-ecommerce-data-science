from __future__ import annotations

import streamlit as st

from app.components.charts import cohort_heatmap
from app.components.data import load_csv
from app.components.ui import hero, metric_card, section_title


def render() -> None:
    hero("Cohort & Retention", "Track how customer cohorts behave after their first delivered purchase.")
    cohort = load_csv("cohort_retention.csv")

    n_cohorts = cohort["cohort_month"].nunique() if not cohort.empty else 0
    month_one_all = cohort[cohort["months_since_first_order"] == 1]
    avg_month1 = float(month_one_all["retention_rate"].mean()) if not month_one_all.empty else 0
    max_horizon = int(cohort["months_since_first_order"].max()) if not cohort.empty else 0

    cols = st.columns(3)
    metrics = [
        ("Cohorts Tracked", f"{n_cohorts:,}", "teal"),
        ("Avg Month-1 Retention", f"{avg_month1:.1%}", "blue"),
        ("Longest Horizon", f"{max_horizon} months", "violet"),
    ]
    for c, (label, value, accent) in zip(cols, metrics):
        with c: metric_card(label, value, accent=accent)

    section_title("Retention matrix", "Rows are acquisition cohorts; columns are months since first purchase.")
    st.plotly_chart(cohort_heatmap(cohort), use_container_width=True)

    section_title("Cohort detail")
    st.dataframe(cohort, use_container_width=True, hide_index=True)

    if not month_one_all.empty:
        section_title("Month-1 retention")
        st.dataframe(
            month_one_all[["cohort_month", "active_customers", "cohort_customers", "retention_rate"]]
            .sort_values("cohort_month", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
