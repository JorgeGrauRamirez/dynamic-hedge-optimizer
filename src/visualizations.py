"""
Interactive visualisations (Plotly) for the Streamlit front-end.
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
    "Unhedged":   "#e74c3c",
    "Naive_1to1": "#f39c12",
    "MVHR":       "#3498db",
    "CVaR":       "#2ecc71",
    "MAD":        "#9b59b6",
    "Minimax":    "#e67e22",
}

STRATEGY_LABELS = {
    "Unhedged":   "Unhedged",
    "Naive_1to1": "Naive 1:1",
    "MVHR":       "MVHR (52w)",
    "CVaR":       "CVaR (LP)",
    "MAD":        "MAD (LP)",
    "Minimax":    "Minimax (LP)",
}

ALL_STRATEGY_COLS = ["Unhedged", "Naive_1to1", "MVHR", "CVaR", "MAD", "Minimax"]

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12),
    title_font=dict(size=14, family="Inter, sans-serif"),
)

_GRID_COLOR = "rgba(255,255,255,0.08)"
_LINE_COLOR = "rgba(255,255,255,0.15)"


def _apply_grid(fig: go.Figure, rows: int = 1, cols: int = 1) -> None:
    """Apply a subtle grid that works on dark backgrounds."""
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.update_xaxes(
                showgrid=True, gridcolor=_GRID_COLOR, gridwidth=1,
                zeroline=False, showline=True, linecolor=_LINE_COLOR,
                row=r, col=c,
            )
            fig.update_yaxes(
                showgrid=True, gridcolor=_GRID_COLOR, gridwidth=1,
                zeroline=False, showline=True, linecolor=_LINE_COLOR,
                row=r, col=c,
            )


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def plot_eda(
    df: pd.DataFrame,
    physical_cols: List[str],
    ffa_cols: List[str],
) -> go.Figure:
    """Two stacked time-series panels: physical routes on top, FFAs below."""
    # Compute per-panel stats for annotation
    phys_means = {c: df[c].mean() for c in physical_cols if df[c].notna().any()}
    ffa_means  = {c: df[c].mean() for c in ffa_cols      if df[c].notna().any()}

    palette_phys = px.colors.qualitative.Pastel
    palette_ffa  = px.colors.qualitative.Bold

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            "Physical Market — Spot Assessments (USD '000 / day)",
            "Paper Market — Baltic FFA Benchmark Indices (USD '000 / day)",
        ),
        row_heights=[0.5, 0.5],
    )

    for i, col in enumerate(physical_cols):
        fig.add_trace(
            go.Scatter(
                x=df["Date"], y=df[col], mode="lines", name=col,
                legendgroup="phys", legendgrouptitle_text="Physical Routes",
                line={"width": 1.4, "color": palette_phys[i % len(palette_phys)]},
                hovertemplate=f"<b>{col}</b><br>%{{x|%Y-%m-%d}}<br>${{y:,.1f}}k/day<extra></extra>",
            ),
            row=1, col=1,
        )

    for i, col in enumerate(ffa_cols):
        fig.add_trace(
            go.Scatter(
                x=df["Date"], y=df[col], mode="lines", name=col,
                legendgroup="ffa", legendgrouptitle_text="FFA Indices",
                line={"width": 2.0, "color": palette_ffa[i % len(palette_ffa)]},
                hovertemplate=f"<b>{col}</b><br>%{{x|%Y-%m-%d}}<br>${{y:,.1f}}k/day<extra></extra>",
            ),
            row=2, col=1,
        )

    fig.update_yaxes(title_text="USD '000 / day", row=1, col=1, tickprefix="$")
    fig.update_yaxes(title_text="USD '000 / day", row=2, col=1, tickprefix="$")
    fig.update_xaxes(title_text="Date", row=2, col=1)

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        height=720,
        hovermode="x unified",
        legend={"groupclick": "toggleitem", "bgcolor": "rgba(30,30,30,0.85)",
                "bordercolor": "rgba(255,255,255,0.15)", "borderwidth": 1},
        margin={"l": 50, "r": 30, "t": 70, "b": 40},
    )
    _apply_grid(fig, rows=2)
    return fig


# ---------------------------------------------------------------------------
# Backtest result plots
# ---------------------------------------------------------------------------

def plot_pnl_distribution(df_res: pd.DataFrame, route: str) -> go.Figure:
    """Violin + KDE-style distribution of per-voyage P&L by strategy."""
    fig = go.Figure()

    for col in ALL_STRATEGY_COLS:
        s = df_res[col]
        label = STRATEGY_LABELS[col]
        color = STRATEGY_COLORS[col]
        fig.add_trace(
            go.Violin(
                y=s,
                name=label,
                line_color=color,
                fillcolor=color,
                opacity=0.60,
                box_visible=True,
                meanline_visible=True,
                points=False,
                hoverinfo="y+name",
            )
        )
        # Annotate CVaR and mean below each violin
        mean_val = s.mean()
        cvar_val = realized_cvar(s)
        fig.add_annotation(
            x=label,
            y=s.min() - (s.max() - s.min()) * 0.08,
            text=f"μ=${mean_val:,.0f}<br>CVaR=${cvar_val:,.0f}",
            showarrow=False,
            font={"size": 9, "color": "rgba(180,180,180,0.8)"},
            align="center",
        )

    n = len(df_res)
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(
            text=f"P&L Distribution per Voyage — <span style='color:#2c5364'>{route}</span>  ({n} voyages)",
            font=dict(size=13),
        ),
        yaxis_title="Voyage P&L (USD)",
        height=560,
        showlegend=False,
        margin={"l": 50, "r": 20, "t": 70, "b": 60},
    )
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", showgrid=True, gridcolor=_GRID_COLOR)
    fig.update_xaxes(showgrid=False)
    return fig


def plot_cumulative_pnl(df_res: pd.DataFrame, route: str) -> go.Figure:
    """Cumulative P&L over the backtest horizon for every strategy."""
    fig = go.Figure()

    for col in ALL_STRATEGY_COLS:
        cum = df_res[col].cumsum()
        label = STRATEGY_LABELS[col]
        color = STRATEGY_COLORS[col]
        final_val = cum.iloc[-1]

        fig.add_trace(
            go.Scatter(
                x=df_res["Date"],
                y=cum,
                mode="lines",
                name=label,
                line={"width": 2.2, "color": color},
                hovertemplate=f"<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>Cum P&L: $%{{y:,.0f}}<extra></extra>",
            )
        )
        # End-of-line label
        fig.add_annotation(
            x=df_res["Date"].iloc[-1],
            y=final_val,
            text=f" {label}<br>${ final_val:,.0f}",
            showarrow=False,
            xanchor="left",
            font={"size": 9, "color": color},
        )

    fig.add_hline(y=0, line_width=1.2, line_color="rgba(180,180,180,0.5)", line_dash="dot", opacity=0.7)

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(text=f"Cumulative P&L — <span style='color:#2c5364'>{route}</span>", font=dict(size=13)),
        xaxis_title="Voyage Date",
        yaxis_title="Cumulative P&L (USD)",
        height=520,
        hovermode="x unified",
        legend={"bgcolor": "rgba(30,30,30,0.85)", "bordercolor": "rgba(255,255,255,0.15)", "borderwidth": 1},
        margin={"l": 50, "r": 140, "t": 70, "b": 40},
    )
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    _apply_grid(fig)
    return fig


def plot_advanced_vs_time(df_res: pd.DataFrame) -> go.Figure:
    """Scatter of the 3 advanced models over time, shaded by HMM crisis prob."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df_res["Date"],
            y=df_res["Prob_Crisis"],
            name="HMM Crisis Prob.",
            marker_color="rgba(231, 76, 60, 0.20)",
            marker_line_width=0,
            yaxis="y2",
            hovertemplate="%{x|%Y-%m-%d}<br>P(crisis): %{y:.1%}<extra>HMM Crisis Prob.</extra>",
        )
    )

    for col in ["CVaR", "MAD", "Minimax"]:
        label = STRATEGY_LABELS[col]
        color = STRATEGY_COLORS[col]
        fig.add_trace(
            go.Scatter(
                x=df_res["Date"],
                y=df_res[col],
                mode="markers",
                name=label,
                marker={
                    "color": color,
                    "size": 7,
                    "opacity": 0.80,
                    "line": {"width": 0.5, "color": "white"},
                },
                hovertemplate=f"<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>P&L: $%{{y:,.0f}}<extra></extra>",
            )
        )

    # Annotate average P(crisis)
    avg_crisis = df_res["Prob_Crisis"].mean()
    fig.add_hline(y=0, line_width=1, line_color="rgba(180,180,180,0.5)", line_dash="dot", opacity=0.6)

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(
            text=f"Advanced Models vs Time  ·  Avg HMM crisis prob: <b>{avg_crisis:.1%}</b>",
            font=dict(size=13),
        ),
        xaxis_title="Voyage Date",
        yaxis={
            "title": "Voyage P&L (USD)",
            "tickprefix": "$",
            "tickformat": ",.0f",
            "showgrid": True,
            "gridcolor": "#e5e7eb",
        },
        yaxis2={
            "title": "P(crisis regime)",
            "overlaying": "y",
            "side": "right",
            "range": [0, 1.2],
            "showgrid": False,
            "tickformat": ".0%",
        },
        height=540,
        hovermode="x unified",
        legend={"bgcolor": "rgba(30,30,30,0.85)", "bordercolor": "rgba(255,255,255,0.15)", "borderwidth": 1},
        margin={"l": 50, "r": 70, "t": 70, "b": 40},
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
        horizontal_spacing=0.04,
    )

    palette = px.colors.qualitative.Set2
    prefixes = ["w_cvar_", "w_mad_", "w_minimax_"]
    model_labels = ["CVaR", "MAD", "Minimax"]

    for col_idx, (prefix, model) in enumerate(zip(prefixes, model_labels), start=1):
        # Compute avg total hedge ratio for annotation
        hr_cols = [f"{prefix}{c}" for c in ffa_cols if f"{prefix}{c}" in df_res.columns]
        if hr_cols:
            avg_hr = df_res[hr_cols].sum(axis=1).mean()
            fig.add_annotation(
                text=f"avg HR = {avg_hr:.2f}",
                x=0.5, y=1.06,
                xref=f"x{col_idx} domain",
                yref=f"y{col_idx} domain",
                showarrow=False,
                font={"size": 10, "color": "#6b7280"},
                xanchor="center",
            )

        for k, col in enumerate(ffa_cols):
            w_col = f"{prefix}{col}"
            if w_col not in df_res.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=df_res["Date"],
                    y=df_res[w_col],
                    mode="lines",
                    name=col,
                    legendgroup=col,
                    showlegend=(col_idx == 1),
                    stackgroup=f"stack_{col_idx}",
                    line={"width": 0.5, "color": palette[k % len(palette)]},
                    fillcolor=palette[k % len(palette)],
                    hovertemplate=(
                        f"<b>{col}</b><br>%{{x|%Y-%m-%d}}<br>w = %{{y:.3f}}<extra>{model}</extra>"
                    ),
                ),
                row=1, col=col_idx,
            )
        fig.add_hline(
            y=1.0, line_dash="dash", line_color="rgba(200,200,200,0.5)", opacity=0.5,
            annotation_text="1:1" if col_idx == 1 else "",
            annotation_position="right",
            row=1, col=col_idx,
        )

    fig.update_yaxes(
        title_text="Hedge Weight",
        range=[0, 2.0],
        row=1, col=1,
        tickformat=".1f",
        showgrid=True,
        gridcolor="#e5e7eb",
    )
    for c in [2, 3]:
        fig.update_yaxes(showgrid=True, gridcolor=_GRID_COLOR, row=1, col=c)

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(text="Dynamic Hedge Allocation — 3 Risk Measures Compared", font=dict(size=13)),
        height=500,
        margin={"l": 50, "r": 20, "t": 90, "b": 60},
        legend={
            "orientation": "h",
            "yanchor": "bottom", "y": -0.30,
            "xanchor": "center", "x": 0.5,
            "bgcolor": "rgba(30,30,30,0.85)",
            "bordercolor": "#e5e7eb",
            "borderwidth": 1,
        },
    )
    return fig


def plot_risk_summary(df_res: pd.DataFrame) -> go.Figure:
    """Three bar charts: tail risk (CVaR), volatility (std), expected P&L."""
    labels = [STRATEGY_LABELS[c] for c in ALL_STRATEGY_COLS]
    colors = [STRATEGY_COLORS[c] for c in ALL_STRATEGY_COLS]

    cvar_vals = [realized_cvar(df_res[c]) for c in ALL_STRATEGY_COLS]
    std_vals  = [df_res[c].std()          for c in ALL_STRATEGY_COLS]
    mean_vals = [df_res[c].mean()         for c in ALL_STRATEGY_COLS]

    # Identify best in each panel (for annotation)
    best_cvar_idx  = int(np.argmax(cvar_vals))   # least negative (closest to 0)
    best_std_idx   = int(np.argmin(std_vals))
    best_mean_idx  = int(np.argmax(mean_vals))

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Tail Risk — CVaR 5%  (↑ closer to 0 = better)",
            "Volatility — Std Dev  (↓ lower = better)",
            "Expected Return — Mean P&L  (↑ higher = better)",
        ),
        horizontal_spacing=0.06,
    )

    def _bar(vals, best_idx, col):
        bar_colors = [
            c if i != best_idx else "#065f46"   # highlight best in dark green
            for i, c in enumerate(colors)
        ]
        fig.add_trace(
            go.Bar(
                x=vals, y=labels, orientation="h",
                marker_color=bar_colors,
                marker_line_width=0,
                showlegend=False,
                text=[f"${v:,.0f}" for v in vals],
                textposition="outside",
                textfont={"size": 10},
                hovertemplate="%{y}: $%{x:,.0f}<extra></extra>",
            ),
            row=1, col=col,
        )

    _bar(cvar_vals, best_cvar_idx, 1)
    _bar(std_vals,  best_std_idx,  2)
    _bar(mean_vals, best_mean_idx, 3)

    fig.add_vline(x=0, line_width=1, line_color="rgba(200,200,200,0.5)", opacity=0.5, row=1, col=3)

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        height=440,
        margin={"l": 100, "r": 80, "t": 80, "b": 40},
    )
    for c in [1, 2, 3]:
        fig.update_xaxes(tickprefix="$", tickformat=",.0f", row=1, col=c,
                         showgrid=True, gridcolor=_GRID_COLOR)
        fig.update_yaxes(showgrid=False, row=1, col=c)
    return fig


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_summary_table(df_res: pd.DataFrame) -> pd.DataFrame:
    """A compact metrics table."""
    rows = {}
    for col in ALL_STRATEGY_COLS:
        s = df_res[col]
        rows[STRATEGY_LABELS[col]] = {
            "Mean P&L ($)":        s.mean(),
            "Std Dev ($)":         s.std(),
            "Realized CVaR 5% ($)": realized_cvar(s),
            "VaR 5% ($)":          s.quantile(0.05),
            "Max Loss ($)":        s.min(),
            "Max Gain ($)":        s.max(),
            "Median P&L ($)":      s.median(),
        }

    hr_map = {
        "Unhedged":   0.0,
        "Naive_1to1": 1.0,
        "MVHR":       df_res["MVHR_Beta"].mean()  if "MVHR_Beta" in df_res else np.nan,
        "CVaR":       df_res["HR_CVaR"].mean()    if "HR_CVaR"   in df_res else np.nan,
        "MAD":        df_res["HR_MAD"].mean()     if "HR_MAD"    in df_res else np.nan,
        "Minimax":    df_res["HR_Minimax"].mean() if "HR_Minimax" in df_res else np.nan,
    }
    for col in ALL_STRATEGY_COLS:
        rows[STRATEGY_LABELS[col]]["Avg Hedge Ratio"] = hr_map[col]

    return pd.DataFrame(rows).T
