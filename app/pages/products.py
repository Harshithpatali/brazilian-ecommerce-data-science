from __future__ import annotations

import streamlit as st

from app.components.charts import horizontal_bar
from app.components.data import load_csv
from app.components.ui import hero, metric_card, section_title


def render() -> None:
    hero("Product Analytics", "Category economics, order volume and customer experience.")
    categories = load_csv("category_performance.csv")

    total_revenue = float(categories["revenue"].sum()) if not categories.empty else 0
    top = categories.iloc[0] if not categories.empty else None
    top_share = float(top["revenue"] / total_revenue) if top is not None and total_revenue else 0

    cols = st.columns(3)
    metrics = [
        ("Categories Tracked", f"{len(categories):,}", "teal"),
        ("Top Category", str(top["category"]) if top is not None else "N/A", "blue"),
        ("Top Category Share", f"{top_share:.1%}", "amber"),
    ]
    for c, (label, value, accent) in zip(cols, metrics):
        with c: metric_card(label, value, accent=accent)

    n = st.slider("Top categories", 5, min(25, len(categories)), min(12, len(categories)))
    section_title("Revenue concentration")
    st.plotly_chart(horizontal_bar(categories.head(n), "category", "revenue"), use_container_width=True)
    section_title("Category scorecard")
    st.dataframe(categories, use_container_width=True, hide_index=True)
