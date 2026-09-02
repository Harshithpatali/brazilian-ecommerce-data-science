from __future__ import annotations

import streamlit as st

from app.components.charts import segment_donut
from app.components.data import load_csv
from app.components.ui import hero, metric_card, section_title


def render() -> None:
    hero("Customer Intelligence", "RFM segmentation and customer-value diagnostics built from customer_unique_id.")
    segments = load_csv("customer_segments.csv")
    customers = load_csv("customer_metrics.csv")

    total = int(segments["customers"].sum()) if not segments.empty else 0
    revenue = float(segments["revenue"].sum()) if not segments.empty else 0
    top = segments.sort_values("revenue", ascending=False).iloc[0] if not segments.empty else None
    repeat_rate = float((customers["orders"].fillna(0) >= 2).mean()) if not customers.empty else 0

    cols = st.columns(4)
    metrics = [
        ("Analyzed Customers", f"{total:,}", "teal"),
        ("Customer Revenue", f"R$ {revenue:,.0f}", "blue"),
        ("Repeat-Customer Rate", f"{repeat_rate:.1%}", "violet"),
        ("Largest Revenue Segment", str(top["segment"]) if top is not None else "N/A", "amber"),
    ]
    for c, (label, value, accent) in zip(cols, metrics):
        with c: metric_card(label, value, accent=accent)

    left, right = st.columns([1, 1])
    with left:
        section_title("RFM portfolio")
        st.plotly_chart(segment_donut(segments), use_container_width=True)
    with right:
        section_title("Segment economics")
        st.dataframe(
            segments.sort_values("revenue", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    section_title("Highest-value customers", "Top 100 customer records from the SQL customer metrics view.")
    st.dataframe(
        customers.sort_values("revenue", ascending=False).head(100),
        use_container_width=True,
        hide_index=True,
    )
