from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BG = "#0b1220"
TEXT = "#e8eef8"
GRID = "rgba(148,163,184,0.16)"
ACCENT = "#38bdf8"
ACCENT2 = "#a78bfa"
MUTED = "#94a3b8"


def base(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter, Segoe UI, sans-serif"),
        hoverlabel=dict(bgcolor="#111827", font_color=TEXT),
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.05, x=0),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def revenue_trend(df: pd.DataFrame) -> go.Figure:
    d = df.copy()
    d["month"] = pd.to_datetime(d["month"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["month"], y=d["revenue"], mode="lines+markers",
        name="Revenue", line=dict(color=ACCENT, width=3), marker=dict(size=6)
    ))
    fig.update_layout(
        xaxis_title=None, yaxis_title="Revenue (R$)",
        hovermode="x unified",
    )
    return base(fig)


def revenue_and_orders(df: pd.DataFrame) -> go.Figure:
    d = df.copy(); d["month"] = pd.to_datetime(d["month"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["month"], y=d["revenue"], name="Revenue", marker_color=ACCENT2))
    fig.add_trace(go.Scatter(x=d["month"], y=d["orders"], name="Orders", yaxis="y2", mode="lines+markers", line=dict(color=ACCENT, width=2)))
    fig.update_layout(
        yaxis=dict(title="Revenue (R$)"),
        yaxis2=dict(title="Orders", overlaying="y", side="right", showgrid=False),
        hovermode="x unified",
    )
    return base(fig)


def horizontal_bar(df: pd.DataFrame, category: str, value: str, title: str | None = None) -> go.Figure:
    d = df.sort_values(value).copy()
    fig = px.bar(d, x=value, y=category, orientation="h", title=title, text_auto=".2s")
    fig.update_traces(marker_color=ACCENT)
    return base(fig)


def segment_donut(df: pd.DataFrame) -> go.Figure:
    fig = px.pie(df, names="segment", values="customers", hole=0.62)
    fig.update_traces(textposition="inside", textinfo="percent")
    fig.update_layout(showlegend=True)
    return base(fig)


def cohort_heatmap(df: pd.DataFrame) -> go.Figure:
    p = df.pivot_table(index="cohort_month", columns="months_since_first_order", values="retention_rate", aggfunc="first")
    p.index = pd.to_datetime(p.index).strftime("%Y-%m")
    fig = go.Figure(go.Heatmap(
        z=p.values, x=list(p.columns), y=list(p.index), colorscale="Blues",
        zmin=0, zmax=1, hovertemplate="Cohort %{y}<br>Month %{x}<br>Retention %{z:.1%}<extra></extra>"
    ))
    fig.update_layout(xaxis_title="Months since first purchase", yaxis_title="Acquisition cohort")
    return base(fig)


def bucket_bar(df: pd.DataFrame) -> go.Figure:
    fig = px.bar(df, x="delivery_bucket", y="avg_review_score", text_auto=".2f")
    fig.update_traces(marker_color=ACCENT)
    fig.update_layout(xaxis_title=None, yaxis_title="Average review score")
    return base(fig)
