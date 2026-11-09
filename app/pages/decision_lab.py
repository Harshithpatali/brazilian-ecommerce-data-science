from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.data import load_csv
from app.components.ui import hero, metric_card, section_title
from src.customer_intelligence import build_rfm, campaign_scenario, kaplan_meier


def render() -> None:
    hero("Customer Decision Lab", "From customer-level evidence to transparent retention scenarios.")
    raw = load_csv("customer_metrics.csv")
    rfm = build_rfm(raw)
    tabs = st.tabs(["RFM Explorer", "Purchase Survival", "Campaign Simulator"])

    with tabs[0]:
        section_title("RFM portfolio", "Quintile scores are relative to this Olist customer population; segments are descriptive rules, not learned personas.")
        summary = rfm.groupby("rfm_segment", as_index=False).agg(
            customers=("customer_unique_id", "count"), revenue=("monetary", "sum"),
            avg_recency_days=("recency_days", "mean"), avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"))
        cols = st.columns(4)
        for col, (label, value) in zip(cols, [
            ("Customers", f"{len(rfm):,}"), ("Segments", str(rfm["rfm_segment"].nunique())),
            ("Observed repeat rate", f"{(rfm['frequency'] >= 2).mean():.2%}"),
            ("Total customer revenue", f"R$ {rfm['monetary'].sum():,.0f}")]):
            with col: metric_card(label, value)
        left, right = st.columns([1.1, 1])
        with left:
            fig = px.scatter(summary, x="avg_recency_days", y="avg_monetary", size="customers",
                             color="rfm_segment", hover_data=["avg_frequency"], title="Segment recency vs value")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.dataframe(summary.sort_values("revenue", ascending=False), use_container_width=True, hide_index=True)
        chosen = st.selectbox("Inspect segment", ["All segments"] + sorted(rfm["rfm_segment"].unique().tolist()))
        view = rfm if chosen == "All segments" else rfm[rfm["rfm_segment"] == chosen]
        st.dataframe(view[["customer_unique_id", "recency_days", "frequency", "monetary",
                           "R_score", "F_score", "M_score", "rfm_segment", "customer_explanation"]].head(500),
                     use_container_width=True, hide_index=True)
        st.download_button("Download customer-level RFM", rfm.to_csv(index=False).encode("utf-8"),
                           "customer_rfm_explanations.csv", "text/csv")

    with tabs[1]:
        section_title("Time to second delivered purchase", "Kaplan–Meier estimate using the exact second delivered order date. Customers without a second purchase are right-censored at the latest observed order date; this is not a causal effect.")
        try:
            km = kaplan_meier(raw)
        except ValueError as exc:
            st.warning("The committed dashboard extract predates the exact second-purchase field.")
            st.code("python -m scripts.build_all", language="bash")
            st.caption(str(exc))
            km = pd.DataFrame()
        if km.empty:
            st.info("Not enough dated customer records to estimate the curve.")
        else:
            fig = px.line(km, x="day", y="repeat_purchase_cumulative",
                          labels={"day": "Days since first delivered order", "repeat_purchase_cumulative": "Estimated cumulative second-purchase probability"},
                          title="1 − Kaplan–Meier survival")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(km.tail(100), use_container_width=True, hide_index=True)
            st.caption(f"At day 90: estimated cumulative second-purchase incidence = {float(km.loc[km.day <= 90, 'repeat_purchase_cumulative'].iloc[-1]) if (km.day <= 90).any() else 0:.2%}.")
            st.download_button("Download survival table", km.to_csv(index=False).encode("utf-8"),
                               "time_to_second_purchase_km.csv", "text/csv")

    with tabs[2]:
        section_title("Campaign scenario planner", "Every lift and unit-economics input below is a user assumption. Olist has no randomized campaign exposure, so outputs are scenarios—not measured incremental lift or causal ROI.")
        segment = st.selectbox("Audience", ["All customers"] + sorted(rfm["rfm_segment"].unique().tolist()))
        c1, c2, c3 = st.columns(3)
        with c1: contact_fraction = st.slider("Contact share", 1, 100, 20, 1) / 100
        with c2: lift_pp = st.slider("Assumed incremental lift (percentage points)", 0.0, 10.0, 1.0, 0.1)
        with c3: contact_cost = st.number_input("Cost per contact (R$)", 0.0, 100.0, 0.50, 0.10)
        margin = st.slider("Contribution margin assumption", 0, 100, 30, 1) / 100
        result = campaign_scenario(rfm, segment, contact_fraction, lift_pp, contact_cost, margin)
        cols = st.columns(4)
        for col, (label, value) in zip(cols, [
            ("Eligible audience", f"{result['eligible_customers']:,}"),
            ("Customers contacted", f"{result['contacted_customers']:,}"),
            ("Assumed incremental buyers", f"{result['assumed_incremental_repeaters']:,.1f}"),
            ("Scenario net contribution", f"R$ {result['scenario_net_contribution']:,.0f}")]):
            with col: metric_card(label, value)
        st.write({
            "Historical repeat rate in selected audience": f"{result['historical_repeat_rate']:.2%}",
            "Median observed AOV": f"R$ {result['median_aov']:,.2f}",
            "Scenario incremental revenue": f"R$ {result['expected_incremental_revenue']:,.2f}",
            "Assumed gross contribution": f"R$ {result['gross_contribution']:,.2f}",
            "Campaign cost": f"R$ {result['campaign_cost']:,.2f}",
            "Break-even lift required": "N/A" if result["break_even_lift_pp"] is None else f"{result['break_even_lift_pp']:.2f} pp",
        })
        st.warning("Do not present this as a forecast or causal uplift model. Validate assumptions with a randomized holdout experiment before budget allocation.")
