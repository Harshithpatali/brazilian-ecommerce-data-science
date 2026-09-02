from __future__ import annotations

import streamlit as st

from app.components.charts import horizontal_bar
from app.components.data import load_csv
from app.components.ui import hero, metric_card, section_title


def render() -> None:
    hero("Seller Analytics", "Seller contribution, customer experience and delivery performance.")
    sellers = load_csv("seller_performance.csv")

    total_revenue = float(sellers["revenue"].sum()) if not sellers.empty else 0
    top = sellers.iloc[0] if not sellers.empty else None
    top_share = float(top["revenue"] / total_revenue) if top is not None and total_revenue else 0

    cols = st.columns(3)
    metrics = [
        ("Active Sellers", f"{len(sellers):,}", "teal"),
        ("Top Seller Revenue", f"R$ {float(top['revenue']):,.0f}" if top is not None else "N/A", "blue"),
        ("Top Seller Share", f"{top_share:.1%}", "amber"),
    ]
    for c, (label, value, accent) in zip(cols, metrics):
        with c: metric_card(label, value, accent=accent)

    n = st.slider("Top sellers", 5, min(25, len(sellers)), min(12, len(sellers)))
    section_title("Seller revenue leaderboard")
    st.plotly_chart(horizontal_bar(sellers.head(n), "seller_id", "revenue"), use_container_width=True)
    section_title("Seller scorecard")
    st.dataframe(sellers, use_container_width=True, hide_index=True)
