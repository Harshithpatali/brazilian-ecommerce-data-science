from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.charts import horizontal_bar, revenue_and_orders, revenue_trend, segment_donut
from app.components.data import load_csv
from app.components.ui import hero, metric_card, section_title


def render() -> None:
    hero("Executive Overview", "Revenue, customer value, marketplace mix and operational health.")

    kpi = load_csv("executive_kpis.csv").iloc[0]
    rev = float(kpi["revenue"] or 0)
    orders = int(kpi["orders"] or 0)
    customers = int(kpi["customers"] or 0)
    aov = float(kpi["aov"] or 0)

    cols = st.columns(4)
    vals = [
        ("Total Revenue", f"R$ {rev:,.0f}", "Delivered orders", "blue"),
        ("Orders", f"{orders:,}", "Delivered orders", "violet"),
        ("Unique Customers", f"{customers:,}", "Customer identity", "teal"),
        ("Average Order Value", f"R$ {aov:,.2f}", "Revenue / order", "amber"),
    ]
    for col, (label, value, help_text, accent) in zip(cols, vals):
        with col: metric_card(label, value, help_text=help_text, accent=accent)

    monthly = load_csv("monthly_revenue.csv")
    section_title("Revenue performance", "Monthly revenue and order volume from the analytical SQL layer.")
    st.plotly_chart(revenue_and_orders(monthly), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        categories = load_csv("category_performance.csv").head(10)
        section_title("Top product categories", "Ranked by delivered product revenue.")
        st.plotly_chart(horizontal_bar(categories, "category", "revenue"), use_container_width=True)
    with col2:
        geo = load_csv("geographic_performance.csv").head(10)
        section_title("Top customer states", "Ranked by delivered order value.")
        st.plotly_chart(horizontal_bar(geo, "customer_state", "revenue"), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        segments = load_csv("customer_segments.csv")
        section_title("Customer portfolio", "RFM segment mix by customer count.")
        st.plotly_chart(segment_donut(segments), use_container_width=True)
    with col4:
        monthly_growth = load_csv("monthly_revenue.csv").copy()
        monthly_growth["month"] = pd.to_datetime(monthly_growth["month"])
        section_title("Latest monthly performance")
        st.dataframe(
            monthly_growth.tail(12).sort_values("month", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    section_title("Executive interpretation")
    top_category = load_csv("category_performance.csv").iloc[0]
    top_state = load_csv("geographic_performance.csv").iloc[0]
    st.info(
        f"The largest product category by delivered product revenue is **{top_category['category']}**, "
        f"while **{top_state['customer_state']}** is the leading customer state by delivered order value. "
        "Use the Customer and Operations pages to diagnose the drivers behind these outcomes."
    )
