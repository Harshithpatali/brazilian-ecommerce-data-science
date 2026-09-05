from __future__ import annotations

import sys
from pathlib import Path

# When Streamlit Cloud (or `streamlit run app/streamlit_app.py`) launches
# this file directly, it adds this file's own directory ("app/") to
# sys.path, not the repo root. That breaks "from app.components... import"
# because Python looks for a package named "app" inside app/ itself.
# Adding the repo root here makes the "app.*" imports work regardless of
# how the script is invoked.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.components.theme import inject_theme
from app.pages import churn, customers, delivery, executive, products, retention, sellers, statistics

st.set_page_config(
    page_title="Brazilian E-Commerce Intelligence",
    page_icon="🇧🇷",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()


def home() -> None:
    st.title("🇧🇷 Brazilian E-Commerce Intelligence")
    st.caption(
        "Supabase PostgreSQL → SQL analytical layer → dashboard datasets → Streamlit → ML"
    )
    st.write(
        "Explore marketplace performance, customer behavior, retention, products, "
        "sellers, delivery experience, statistical analysis, and predictive churn risk."
    )
    st.info(
        "The dashboard is downstream of the SQL analytical layer. "
        "The ML section consumes the point-in-time churn dataset generated from SQL."
    )


pages = {
    "Dashboard": [
        st.Page(home, title="Home", url_path="streamlit_app", icon="🏠"),
        st.Page(executive.render, title="Executive Overview", url_path="executive", icon="📊"),
        st.Page(customers.render, title="Customer Intelligence", url_path="customers", icon="👥"),
        st.Page(retention.render, title="Cohort & Retention", url_path="retention", icon="🔁"),
        st.Page(products.render, title="Product Analytics", url_path="products", icon="🛍️"),
        st.Page(sellers.render, title="Seller Analytics", url_path="sellers", icon="🏪"),
        st.Page(delivery.render, title="Delivery & Experience", url_path="delivery", icon="🚚"),
        st.Page(statistics.render, title="Statistical Analysis", url_path="statistics", icon="📈"),
        st.Page(churn.render, title="Churn Intelligence", url_path="churn", icon="🎯"),
    ]
}

current_page = st.navigation(pages, position="sidebar", expanded=True)

with st.sidebar:
    st.divider()
    st.caption("Source: Supabase PostgreSQL")
    st.caption("SQL → analytical datasets → Streamlit → ML")

current_page.run()
