from __future__ import annotations

from pathlib import Path

import streamlit as st

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"


@st.cache_data
def _read_css(name: str) -> str:
    path = ASSETS_DIR / name
    return path.read_text(encoding="utf-8")


def inject_theme() -> None:
    """Inject the dashboard's global stylesheet. Call once per page load."""
    css = _read_css("styles.css")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
