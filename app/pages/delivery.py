from __future__ import annotations

import streamlit as st

from app.components.charts import bucket_bar, revenue_trend
from app.components.data import load_csv
from app.components.ui import hero, metric_card, section_title


def render() -> None:
    hero("Delivery & Experience", "Operational performance and the observed relationship between delay and review scores.")
    delivery = load_csv("delivery_performance.csv")
    reviews = load_csv("delivery_vs_review.csv")

    avg_delivery = delivery["avg_delivery_days"].mean()
    late_rate = delivery["late_order_rate"].mean()
    avg_delay = delivery["avg_delivery_delay_days"].mean()
    cols = st.columns(3)
    for c, (label, value, accent) in zip(cols, [
        ("Avg Delivery Time", f"{avg_delivery:.1f} days", "blue"),
        ("Avg Late-Order Rate", f"{late_rate:.1%}", "rose"),
        ("Avg Delivery Delay", f"{avg_delay:.1f} days", "rose"),
    ]):
        with c: metric_card(label, value, accent=accent)

    section_title("Delivery performance over time")
    st.plotly_chart(revenue_trend(delivery.rename(columns={"avg_delivery_days": "revenue"})[["month", "revenue"]]), use_container_width=True)

    section_title("Observed delay vs review score", "This is an association view; it should not be interpreted as causal evidence by itself.")
    st.plotly_chart(bucket_bar(reviews), use_container_width=True)
    st.dataframe(reviews, use_container_width=True, hide_index=True)
