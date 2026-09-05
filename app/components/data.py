from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "dashboard"


@st.cache_data(ttl=3600)
def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run: python -m scripts.build_dashboard_data"
        )
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_json(name: str) -> dict:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run: python -m src.statistical_analysis"
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
