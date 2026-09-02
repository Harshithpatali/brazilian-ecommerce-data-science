from __future__ import annotations

import streamlit as st

# Semantic accent colors — kept in sync with app/components/charts.py and
# app/assets/styles.css. Pick the accent that matches what a metric measures
# (revenue = blue, volume/orders = violet, people = teal, value = amber,
# risk/delay = rose) rather than styling every card the same way.
ACCENTS = {
    "blue": "var(--blue)",
    "violet": "var(--violet)",
    "teal": "var(--teal)",
    "amber": "var(--amber)",
    "rose": "var(--rose)",
}


def hero(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    html = f'<div class="hero"><h1>{title}</h1>{subtitle_html}</div>'
    st.markdown(html, unsafe_allow_html=True)


def metric_card(
    label: str,
    value: str,
    delta: str | None = None,
    help_text: str | None = None,
    accent: str = "blue",
) -> None:
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ''
    help_html = f'<div class="metric-help">{help_text}</div>' if help_text else ''
    accent_color = ACCENTS.get(accent, ACCENTS["blue"])
    html = (
        f'<div class="metric-card" style="--accent-bar: {accent_color};">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'{delta_html}'
        f'{help_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def section_title(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def card_start() -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)


def card_end() -> None:
    st.markdown('</div>', unsafe_allow_html=True)
