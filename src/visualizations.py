"""
Interactive visualisations (Plotly) for the Streamlit front-end.

Each function returns a `plotly.graph_objects.Figure` ready to be passed
to `st.plotly_chart(...)`. The colour palette matches the notebook.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .optimizers import realized_cvar


STRATEGY_COLORS = {
    "Unhedged": "#e74c3c",
    "Naive_1to1": "#f39c12",
    "MVHR": "#3498db",
    "CVaR": "#2ecc71",
    "MAD": "#9b59b6",
    "Minimax": "#e67e22",
}

STRATEGY_LABELS = {
    "Unhedged": "Unhedged",
    "Naive_1to1": "Naive 1:1",
    "MVHR": "MVHR (52w)",
    "CVaR": "CVaR (LP)",
    "MAD": "MAD (LP)",
    "Minimax": "Minimax (LP)",
}

ALL_STRATEGY_COLS = ["Unhedged", "Naive_1to1", "MVHR", "CVaR", "MAD", "Minimax"]


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------

def plot_eda(
    df: pd.DataFrame,
    physical_cols: List[str],
    ffa_cols: List[str],
) -> go.Figure:
    """Two stacked time-series panels: physical routes on top, FFAs below."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Physical Market — Spot Assessments",
            "Paper Market — Baltic Indices (FFA Benchmarks)",
        ),
    )
    for col in physical_cols:
        fig.add_trace(
            go.Scatter(
                x=df["Date"], y=df[col], mode="lines", name=col,
                legendgroup="phys", legendgrouptitle_text="Physical",
                line={"width": 1.2},
            ),
            row=1, col=1,
        )
    for col in ffa_cols:
        fig.add_trace(
            go.Scatter(
                x=df["Date"], y=df[col], mode="lines", name=col,
                legendgroup="ffa", legendgrouptitle_text="FFA",
                line={"width": 1.8},
            ),
            row=2, col=1,
        )
    fig.update_yaxes(title_text="USD '000 / day", row=1, col=1)
    fig.update_yaxes(title_text="USD '000 / day", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(
        height=700,
        hovermode="x unified",
        legend={"groupclick": "toggleitem"},
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    return fig


# ---------------------------------------------------------------------------
# Backtest result plots
# ---------------------------------------------------------------------------

def plot_pnl_distribution(df_res: pd.DataFrame, route: str) -> go.Figure:
    """Violin + KDE-style distribution of per-voyage P&L by strategy."""
    fig = go.Figure()
    for col in ALL_STRATEGY_COLS:
        fig.add_trace(
            go.Violin(
                y=df_res[col],
                name=STRATEGY_LABELS[col],
                line_color=STRATEGY_COLORS[col],
                fillcolor=STRATEGY_COLORS[col],
                opacity=0.55,
                box_visible=True,
                meanline_visible=True,
                points=False,
            )
        )
    fig.update_layout(
        title=f"P&L Distribution per Voyage — {route} ({len(df_res)} voyages)",
        yaxis_title="Voyage P&L (USD)",
        height=520,
        showlegend=False,
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


def plot_cumulative_pnl(df_res: pd.DataFrame, route: str) -> go.Figure:
    """Cumulative P&L over the backtest horizon for every strategy."""
    fig = go.Figure()
    for col in ALL_STRATEGY_COLS:
        fig.add_trace(
            go.Scatter(
                x=df_res["Date"],
                y=df_res[col].cumsum(),
                mode="lines",
                name=STRATEGY_LABELS[col],
                line={"width": 2, "color": STRATEGY_COLORS[col]},
                hovertemplate="%{x|%Y-%m-%d}<br>Cum P&L: $%{y:,.0f}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_width=1, line_color="black", opacity=0.4)
    fig.update_layout(
        title=f"Cumulative P&L — {route}",
        xaxis_title="Date",
        yaxis_title="Cumulative P&L (USD)",
        height=500,
        hovermode="x unified",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


def plot_advanced_vs_time(df_res: pd.DataFrame) -> go.Figure:
    """Scatter of the 3 advanced models over time, shaded by HMM crisis prob."""
    fig = go.Figure()

    # Crisis shading: add a faint bar chart on a secondary invisible axis
    # so the user can visually tie P&Ls to stress periods.
    fig.add_trace(
        go.Bar(
            x=df_res["Date"],
            y=df_res["Prob_Crisis"],
            name="HMM Crisis Prob.",
            marker_color="rgba(231, 76, 60, 0.25)",
            yaxis="y2",
            hovertemplate="%{x|%Y-%m-%d}<br>P(crisis): %{y:.2f}<extra></extra>",
        )
    )
    for col in ["CVaR", "MAD", "Minimax"]:
        fig.add_trace(
            go.Scatter(
                x=df_res["Date"],
                y=df_res[col],
                mode="markers",
                name=STRATEGY_LABELS[col],
                marker={"color": STRATEGY_COLORS[col], "size": 7, "opacity": 0.75},
                hovertemplate="%{x|%Y-%m-%d}<br>P&L: $%{y:,.0f}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_width=1, line_color="black", opacity=0.4)
    fig.update_layout(
        title="Advanced Models Over Time (bars = HMM crisis probability)",
        xaxis_title="Date",
        yaxis={"title": "Voyage P&L (USD)", "tickprefix": "$", "tickformat": ",.0f"},
        yaxis2={
            "title": "P(crisis)",
            "overlaying": "y",
            "side": "right",
            "range": [0, 1],
            "showgrid": False,
        },
        height=520,
        hovermode="x unified",
        margin={"l": 40, "r": 60, "t": 60, "b": 40},
    )
    return fig


def plot_hedge_allocations(
    df_res: pd.DataFrame, ffa_cols: List[str]
) -> go.Figure:
    """Three stacked-area subplots showing how each optimiser allocates
    weight across the FFA universe over time."""
    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=True,
        subplot_titles=("CVaR Allocation", "MAD Allocation", "Minimax Allocation"),
    )

    palette = px.colors.qualitative.Set2
    prefixes = ["w_cvar_", "w_mad_", "w_minimax_"]

    for col_idx, prefix in enumerate(prefixes, start=1):
        for k, col in enumerate(ffa_cols):
            fig.add_trace(
                go.Scatter(
                    x=df_res["Date"],
                    y=df_res[f"{prefix}{col}"],
                    mode="lines",
                    name=col,
                    legendgroup=col,
                    showlegend=(col_idx == 1),
                    stackgroup=f"stack_{col_idx}",
                    line={"width": 0.5, "color": palette[k % len(palette)]},
                    fillcolor=palette[k % len(palette)],
                    hovertemplate=(
                        f"{col}<br>%{{x|%Y-%m-%d}}<br>w = %{{y:.2f}}<extra></extra>"
                    ),
                ),
                row=1, col=col_idx,
            )
        fig.add_hline(
            y=1.0, line_dash="dash", line_color="black", opacity=0.6,
            row=1, col=col_idx,
        )

    fig.update_yaxes(title_text="Hedge Weight", range=[0, 2.0], row=1, col=1)
    fig.update_layout(
        title="Dynamic Hedge Allocation — 3 Risk Measures Compared",
        height=480,
        margin={"l": 40, "r": 20, "t": 80, "b": 40},
        legend={
            "orientation": "h",
            "yanchor": "bottom", "y": -0.35,
            "xanchor": "center", "x": 0.5,
        },
    )
    return fig


def plot_risk_summary(df_res: pd.DataFrame) -> go.Figure:
    """Three bar charts: tail risk (CVaR), volatility (std), expected P&L."""
    labels = [STRATEGY_LABELS[c] for c in ALL_STRATEGY_COLS]
    colors = [STRATEGY_COLORS[c] for c in ALL_STRATEGY_COLS]

    cvar_vals = [realized_cvar(df_res[c]) for c in ALL_STRATEGY_COLS]
    std_vals = [df_res[c].std() for c in ALL_STRATEGY_COLS]
    mean_vals = [df_res[c].mean() for c in ALL_STRATEGY_COLS]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Tail Risk — Realized CVaR 5%",
            "Volatility — Std Dev",
            "Expected Return — Mean P&L",
        ),
    )
    fig.add_trace(
        go.Bar(
            x=cvar_vals, y=labels, orientation="h",
            marker_color=colors, showlegend=False,
            text=[f"${v:,.0f}" for v in cvar_vals], textposition="auto",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=std_vals, y=labels, orientation="h",
            marker_color=colors, showlegend=False,
            text=[f"${v:,.0f}" for v in std_vals], textposition="auto",
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(
            x=mean_vals, y=labels, orientation="h",
            marker_color=colors, showlegend=False,
            text=[f"${v:,.0f}" for v in mean_vals], textposition="auto",
        ),
        row=1, col=3,
    )
    fig.update_layout(
        height=420,
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
    )
    fig.update_xaxes(tickprefix="$", tickformat=",.0f")
    fig.add_vline(x=0, line_width=1, line_color="black", opacity=0.5, row=1, col=3)
    return fig


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_summary_table(df_res: pd.DataFrame) -> pd.DataFrame:
    """A compact metrics table (same rows as the notebook printout)."""
    rows = {}
    for col in ALL_STRATEGY_COLS:
        s = df_res[col]
        rows[STRATEGY_LABELS[col]] = {
            "Mean P&L ($)": s.mean(),
            "Std Dev ($)": s.std(),
            "Realized CVaR 5% ($)": realized_cvar(s),
            "VaR 5% ($)": s.quantile(0.05),
            "Max Loss ($)": s.min(),
            "Max Gain ($)": s.max(),
            "Median P&L ($)": s.median(),
        }

    # Average hedge ratio row
    hr_map = {
        "Unhedged": 0.0,
        "Naive_1to1": 1.0,
        "MVHR": df_res["MVHR_Beta"].mean() if "MVHR_Beta" in df_res else np.nan,
        "CVaR": df_res["HR_CVaR"].mean() if "HR_CVaR" in df_res else np.nan,
        "MAD": df_res["HR_MAD"].mean() if "HR_MAD" in df_res else np.nan,
        "Minimax": df_res["HR_Minimax"].mean() if "HR_Minimax" in df_res else np.nan,
    }
    for col in ALL_STRATEGY_COLS:
        rows[STRATEGY_LABELS[col]]["Avg Hedge Ratio"] = hr_map[col]

    return pd.DataFrame(rows).T
