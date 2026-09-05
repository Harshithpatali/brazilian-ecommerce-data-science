from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.data import load_json
from app.components.ui import hero, metric_card, section_title


def _format_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.4f}"


def render() -> None:
    hero(
        "Statistical Analysis",
        "Inference, effect sizes, post-hoc comparisons, and sampling estimates built from the SQL analytical layer.",
    )

    try:
        results = load_json("statistical_results.json")
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Run `python -m scripts.build_all` after pulling the latest changes.")
        return

    alpha = results["alpha"]
    anova = results["one_way_anova"]
    two_way = results["two_factor_anova"]
    sampling = results["sampling"]

    section_title("1. One-way ANOVA", "Does customer RFM segment differ in monetary value?")
    cols = st.columns(4)
    for c, (label, value, accent) in zip(
        cols,
        [
            ("F-statistic", f"{anova['f_statistic']:,.2f}", "blue"),
            ("p-value", _format_p(anova["p_value"]), "rose"),
            ("η² effect size", f"{anova['eta_squared']:.3f}", "blue"),
            ("Levene p-value", _format_p(anova["levene_p_value"]), "rose"),
        ],
    ):
        with c:
            metric_card(label, value, accent=accent)

    if anova["p_value"] < alpha:
        st.success(
            f"Reject H₀ at α={alpha:.2f}: mean log monetary value differs across customer segments."
        )
    else:
        st.info(f"Fail to reject H₀ at α={alpha:.2f}.")

    if anova.get("variance_heterogeneity"):
        kw = anova.get("kruskal_wallis", {})
        st.warning(
            "Levene's test indicates unequal variances. The Kruskal-Wallis result is shown as a sensitivity check. "
            f"H={kw.get('h_statistic', float('nan')):,.2f}, p={_format_p(kw.get('p_value', 1.0))}."
        )

    tukey = pd.DataFrame(anova["tukey"])
    if not tukey.empty:
        st.caption("Tukey HSD controls the family-wise error rate for pairwise comparisons.")
        display_tukey = tukey.rename(
            columns={
                "group1": "Segment 1",
                "group2": "Segment 2",
                "meandiff": "Mean Difference",
                "p-adj": "Adjusted p-value",
                "lower": "CI Lower",
                "upper": "CI Upper",
                "reject": "Significant",
            }
        )
        st.dataframe(display_tukey, use_container_width=True, hide_index=True)

    section_title(
        "2. Two-factor ANOVA",
        "Do RFM segment and delivery performance jointly explain review-score differences?",
    )
    anova_table = pd.DataFrame(two_way["anova_table"])
    if not anova_table.empty:
        display_table = anova_table.rename(
            columns={
                "term": "Source",
                "sum_sq": "Sum of Squares",
                "df": "df",
                "F": "F-statistic",
                "PR(>F)": "p-value",
            }
        )
        st.dataframe(display_table, use_container_width=True, hide_index=True)

    interaction_p = two_way["interaction_p_value"]
    if two_way["interaction_significant"]:
        st.success(
            f"Interaction is statistically significant (p={_format_p(interaction_p)}): "
            "the relationship between delivery performance and review score varies by customer segment."
        )
    else:
        st.info(f"No statistically significant interaction detected (p={_format_p(interaction_p)}).")

    section_title(
        "3. Sampling & Sampling Distributions",
        "Compare a simple random sample with a proportionally stratified sample.",
    )
    cols = st.columns(4)
    for c, (label, value, accent) in zip(
        cols,
        [
            ("Population", f"{sampling['population_size']:,}", "blue"),
            ("Sample", f"{sampling['sample_size']:,}", "blue"),
            ("Population Mean", f"{sampling['population_mean']:.2f}", "blue"),
            ("SRS Relative Error", f"{sampling['simple_random_sample']['relative_error']:.2%}", "rose"),
        ],
    ):
        with c:
            metric_card(label, value, accent=accent)

    srs = sampling["simple_random_sample"]
    strat = sampling["stratified_sample"]
    comparison = pd.DataFrame(
        {
            "Estimator": ["Population mean", "Simple random sample", "Weighted stratified sample"],
            "Mean": [sampling["population_mean"], srs["sample_mean"], strat["weighted_mean"]],
        }
    )
    fig = px.bar(comparison, x="Estimator", y="Mean", title="Mean estimate comparison")
    fig.update_layout(xaxis_title=None, yaxis_title="Monetary value")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            "SRS 95% CI",
            f"[{srs['ci_lower']:.2f}, {srs['ci_upper']:.2f}]",
        )
        st.metric("SRS absolute error", f"{srs['absolute_error']:.2f}")
    with c2:
        st.metric("Weighted stratified mean", f"{strat['weighted_mean']:.2f}")
        st.metric("Weighted absolute error", f"{strat['weighted_absolute_error']:.2f}")

    section_title("Stratified allocation", "Proportional allocation across the six RFM customer segments.")
    allocation = pd.DataFrame(
        list(strat["allocation"].items()), columns=["Segment", "Sample Size"]
    )
    fig = px.bar(allocation, x="Segment", y="Sample Size", title="Stratified sample allocation")
    fig.update_layout(xaxis_title=None, yaxis_title="Customers")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Interpretation note: these analyses establish statistical associations and group differences; "
        "they do not by themselves establish causality. Monetary ANOVA uses log1p-transformed monetary value "
        "because customer monetary values are strongly skewed."
    )
