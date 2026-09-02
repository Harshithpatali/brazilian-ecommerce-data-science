from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def revenue_line(df: pd.DataFrame) -> go.Figure:
    fig = px.line(df, x="month", y="revenue", markers=True)
    fig.update_layout(xaxis_title=None, yaxis_title="Revenue", margin=dict(l=10, r=10, t=20, b=10))
    return fig


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str | None = None, horizontal: bool = False) -> go.Figure:
    if horizontal:
        fig = px.bar(df, x=y, y=x, orientation="h", title=title)
    else:
        fig = px.bar(df, x=x, y=y, title=title)
    fig.update_layout(margin=dict(l=10, r=10, t=45 if title else 20, b=10))
    return fig


def retention_heatmap(df: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        df,
        labels=dict(x="Months Since Acquisition", y="Cohort Month", color="Retention %"),
        aspect="auto",
        text_auto=".0%",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    return fig
