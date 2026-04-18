"""
Dynamic Hedge & Fix Optimiser — Streamlit front-end.

Run locally with:
    streamlit run app.py

The sidebar exposes every meaningful parameter from the notebook
(route, anchor, FFA universe, calibration windows, Monte Carlo size,
hedge bounds, CVaR alpha, etc.). The main panel has three tabs:
    1. Overview — data description + exploratory time-series.
    2. Backtest Results — metrics table, P&L distribution, cumulative P&L.
    3. Deep Dive — hedge allocations, regime shading, full results table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import (
    build_working_dataset,
    get_recommended_ffa_universe,
    load_data,
    split_columns,
)
from src.backtesting import BacktestConfig, run_backtest
from src.optimizers import realized_cvar
from src.visualizations import (
    ALL_STRATEGY_COLS,
    STRATEGY_LABELS,
    build_summary_table,
    plot_advanced_vs_time,
    plot_cumulative_pnl,
    plot_eda,
    plot_hedge_allocations,
    plot_pnl_distribution,
    plot_risk_summary,
)


# ---------------------------------------------------------------------------
# Page config + CSS tweaks
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dynamic Hedge & Fix Optimiser",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        [data-testid="stMetricValue"] { font-size: 1.4rem; }
        h1, h2, h3 { color: #1f3b57; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "Merged_Shipping_Data.csv"


@st.cache_data(show_spinner="Loading shipping data...")
def _cached_load(path: str) -> pd.DataFrame:
    return load_data(path)


@st.cache_data(show_spinner=False)
def _run_backtest_cached(
    df_working: pd.DataFrame, cfg_dict: dict
) -> pd.DataFrame:
    cfg = BacktestConfig(**cfg_dict)
    return run_backtest(df_working, cfg)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("⚓ Dynamic Hedge & Fix Optimiser")
st.caption(
    "Optimisation-based risk management for hedging in dry-bulk freight "
    "markets · Master Thesis, Jorge (S243346) · Navi Merchants"
)


# ---------------------------------------------------------------------------
# Data ingestion — either the bundled CSV or a user upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Data")
    uploaded = st.file_uploader(
        "Upload your own merged dataset (optional)",
        type=["csv"],
        help=(
            "If left empty, the app uses the bundled `Merged_Shipping_Data.csv`. "
            "Your file must have a `Date` column plus physical and FFA columns, "
            "already scaled to thousands."
        ),
    )

if uploaded is not None:
    df = load_data(uploaded)
    data_source = "uploaded file"
elif DEFAULT_DATA_PATH.exists():
    df = _cached_load(str(DEFAULT_DATA_PATH))
    data_source = "bundled dataset"
else:
    st.error(
        "No data found. Please place `Merged_Shipping_Data.csv` inside the "
        "`data/` folder or upload a file from the sidebar."
    )
    st.stop()

physical_cols, ffa_cols_all = split_columns(df)
recommended_ffa = get_recommended_ffa_universe(df)


# ---------------------------------------------------------------------------
# Sidebar — all tunable parameters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("2. Route & instruments")
    target_physical_route = st.selectbox(
        "Physical route (exposure)",
        options=physical_cols,
        index=physical_cols.index("cont trip spore/jpn rge (Smax)")
        if "cont trip spore/jpn rge (Smax)" in physical_cols
        else 0,
        help="The physical freight assessment you are exposed to.",
    )
    anchor_index_col = st.selectbox(
        "Anchor FFA (correlated benchmark)",
        options=recommended_ffa,
        index=recommended_ffa.index("Smx Avg 10TC")
        if "Smx Avg 10TC" in recommended_ffa
        else 0,
        help=(
            "The FFA most correlated with your physical route — used as the "
            "anchor in the OU log-spread model and for the Naive 1:1 benchmark."
        ),
    )
    ffa_columns = st.multiselect(
        "FFA universe (hedging instruments)",
        options=recommended_ffa,
        default=recommended_ffa,
        help=(
            "Set of FFAs available to the optimiser. The anchor must be "
            "included. `Smx Avg 11TC` is excluded by default because it only "
            "has data from 2023-05."
        ),
    )
    if anchor_index_col not in ffa_columns:
        ffa_columns = [anchor_index_col] + [c for c in ffa_columns if c != anchor_index_col]

    st.header("3. Voyage & backtest")
    voyage_weeks = st.slider("Voyage length (weeks)", 1, 12, 5)
    n_backtests = st.slider(
        "Number of voyages to backtest", 20, 300, 150, step=10
    )
    non_overlapping = st.checkbox(
        "Non-overlapping voyages",
        value=False,
        help="If unchecked, uses a rolling 1-week step (overlapping voyages).",
    )
    assumed_volume = st.number_input(
        "Assumed voyage volume (MT-days or days)",
        min_value=1.0,
        max_value=10_000.0,
        value=65.0,
        step=5.0,
        help="Scales P&L from $/day to absolute $ per voyage.",
    )

    st.header("4. Calibration & regime")
    calibration_weeks = st.slider(
        "Advanced-model calibration window (weeks)",
        4, 52, 8,
        help="Lookback used to calibrate the OU log-spread and GBM volatilities.",
    )
    mvhr_calibration_weeks = st.slider(
        "MVHR calibration window (weeks)", 8, 104, 52
    )
    stress_corr = st.slider(
        "Stress correlation (crisis regime)",
        0.50, 1.00, 0.90, step=0.05,
        help=(
            "Correlation assumed inside the high-volatility regime. Blended "
            "with the historical matrix using the HMM crisis probability."
        ),
    )

    st.header("5. Monte Carlo & optimiser")
    n_sims = st.select_slider(
        "Number of Monte Carlo scenarios",
        options=[1_000, 2_500, 5_000, 7_500, 10_000, 15_000, 20_000],
        value=10_000,
    )
    cvar_alpha = st.slider(
        "CVaR tail level α", 0.01, 0.20, 0.05, step=0.01,
        help="CVaR is computed on the worst α fraction of scenarios.",
    )
    hedge_upper_bound = st.slider(
        "Upper bound on each hedge weight", 0.5, 3.0, 1.5, step=0.1,
        help="Maximum weight on any single FFA instrument.",
    )
    random_seed = st.number_input("Random seed", value=42, step=1)

    st.divider()
    run_btn = st.button("Run backtest", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Build the working dataset (cheap, always up-to-date)
# ---------------------------------------------------------------------------
if not ffa_columns:
    st.warning("Select at least one FFA instrument from the sidebar.")
    st.stop()

df_working = build_working_dataset(df, target_physical_route, ffa_columns)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_backtest, tab_deepdive = st.tabs(
    ["📊 Overview", "🎯 Backtest Results", "🔬 Deep Dive"]
)


# ---------- TAB 1: OVERVIEW ----------
with tab_overview:
    st.subheader("Dataset summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data source", data_source.title())
    c2.metric("Rows (weekly)", f"{len(df):,}")
    c3.metric("Date range", f"{df['Date'].min():%Y-%m-%d}")
    c4.metric("…through", f"{df['Date'].max():%Y-%m-%d}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Physical routes", len(physical_cols))
    c2.metric("FFA indices", len(ffa_cols_all))
    c3.metric("Working rows (after filtering)", f"{len(df_working):,}")

    st.markdown("#### Selected route & anchor")
    st.write(
        f"**Physical exposure:** `{target_physical_route}`  "
        f"· **Anchor FFA:** `{anchor_index_col}`  "
        f"· **FFA universe:** {', '.join(f'`{c}`' for c in ffa_columns)}"
    )

    st.markdown("#### Exploratory time series")
    st.plotly_chart(
        plot_eda(df, physical_cols, ffa_cols_all),
        use_container_width=True,
    )

    with st.expander("View raw data (last 50 rows)"):
        st.dataframe(df.tail(50), use_container_width=True, height=350)


# ---------- TAB 2: BACKTEST RESULTS ----------
with tab_backtest:
    st.subheader("Backtest results")

    if run_btn or "df_results" in st.session_state:
        if run_btn:
            cfg_dict = dict(
                target_physical_route=target_physical_route,
                anchor_index_col=anchor_index_col,
                ffa_columns=ffa_columns,
                voyage_weeks=voyage_weeks,
                n_backtests=n_backtests,
                assumed_volume=assumed_volume,
                stress_corr=stress_corr,
                mvhr_calibration_weeks=mvhr_calibration_weeks,
                calibration_weeks=calibration_weeks,
                non_overlapping=non_overlapping,
                n_sims=int(n_sims),
                cvar_alpha=cvar_alpha,
                hedge_upper_bound=hedge_upper_bound,
                random_seed=int(random_seed),
            )
            progress = st.progress(0.0, text="Starting backtest...")
            cfg = BacktestConfig(**cfg_dict)

            def _cb(frac: float, msg: str) -> None:
                progress.progress(min(max(frac, 0.0), 1.0), text=msg)

            try:
                df_results = run_backtest(df_working, cfg, progress_cb=_cb)
            except ValueError as e:
                progress.empty()
                st.error(str(e))
                st.stop()

            progress.empty()
            st.session_state["df_results"] = df_results
            st.session_state["cfg"] = cfg_dict

        df_results: pd.DataFrame = st.session_state["df_results"]
        cfg_used: dict = st.session_state["cfg"]

        if df_results.empty:
            st.warning(
                "The backtest produced no valid voyages. Try a different route "
                "or a shorter calibration window."
            )
            st.stop()

        st.success(
            f"Backtest complete — {len(df_results)} voyages evaluated on "
            f"`{cfg_used['target_physical_route']}`."
        )

        # -- Headline metrics (CVaR strategy as the reference advanced model) --
        ref_col = "CVaR"
        unhedged_cvar = realized_cvar(df_results["Unhedged"])
        ref_cvar = realized_cvar(df_results[ref_col])
        cvar_reduction = (
            100 * (ref_cvar - unhedged_cvar) / abs(unhedged_cvar)
            if unhedged_cvar != 0 and not np.isnan(unhedged_cvar) else np.nan
        )
        unhedged_std = df_results["Unhedged"].std()
        ref_std = df_results[ref_col].std()
        std_reduction = (
            100 * (unhedged_std - ref_std) / unhedged_std
            if unhedged_std > 0 else np.nan
        )
        best_mean_col = max(ALL_STRATEGY_COLS, key=lambda c: df_results[c].mean())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Voyages backtested", f"{len(df_results):,}")
        c2.metric(
            f"CVaR 5% — {ref_col}",
            f"${ref_cvar:,.0f}",
            f"{cvar_reduction:+.1f}% vs Unhedged" if not np.isnan(cvar_reduction) else None,
            delta_color="normal",
        )
        c3.metric(
            f"Std Dev — {ref_col}",
            f"${ref_std:,.0f}",
            f"{-std_reduction:+.1f}% vs Unhedged" if not np.isnan(std_reduction) else None,
            delta_color="inverse",
        )
        c4.metric(
            "Highest mean P&L",
            STRATEGY_LABELS[best_mean_col],
            f"${df_results[best_mean_col].mean():,.0f}",
        )

        st.markdown("#### Performance metrics by strategy")
        summary = build_summary_table(df_results)
        st.dataframe(
            summary.style.format("{:,.2f}"),
            use_container_width=True,
        )
        csv_summary = summary.to_csv().encode("utf-8")
        st.download_button(
            "⬇️ Download summary table (CSV)",
            csv_summary,
            file_name="summary_metrics.csv",
            mime="text/csv",
        )

        st.markdown("#### P&L distribution")
        st.plotly_chart(
            plot_pnl_distribution(df_results, cfg_used["target_physical_route"]),
            use_container_width=True,
        )

        st.markdown("#### Cumulative P&L")
        st.plotly_chart(
            plot_cumulative_pnl(df_results, cfg_used["target_physical_route"]),
            use_container_width=True,
        )

        st.markdown("#### Risk / return comparison")
        st.plotly_chart(plot_risk_summary(df_results), use_container_width=True)
    else:
        st.info(
            "Adjust parameters in the sidebar and click **Run backtest** to "
            "generate results."
        )


# ---------- TAB 3: DEEP DIVE ----------
with tab_deepdive:
    st.subheader("Deep dive")

    if "df_results" not in st.session_state:
        st.info("Run a backtest first from the **Backtest Results** tab.")
    else:
        df_results = st.session_state["df_results"]
        cfg_used = st.session_state["cfg"]

        st.markdown("#### Advanced models over time + HMM crisis probability")
        st.plotly_chart(plot_advanced_vs_time(df_results), use_container_width=True)

        st.markdown("#### Dynamic hedge allocation by risk measure")
        st.plotly_chart(
            plot_hedge_allocations(df_results, cfg_used["ffa_columns"]),
            use_container_width=True,
        )

        st.markdown("#### Per-voyage results table")
        display_cols = (
            ["Date"] + ALL_STRATEGY_COLS +
            ["Prob_Crisis", "MVHR_Beta", "MVHR_FFA",
             "HR_CVaR", "HR_MAD", "HR_Minimax"]
        )
        st.dataframe(
            df_results[display_cols].style.format({
                **{c: "{:,.0f}" for c in ALL_STRATEGY_COLS},
                "Prob_Crisis": "{:.2%}",
                "MVHR_Beta": "{:.2f}",
                "HR_CVaR": "{:.2f}",
                "HR_MAD": "{:.2f}",
                "HR_Minimax": "{:.2f}",
            }),
            use_container_width=True,
            height=420,
        )

        csv_full = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download full results (CSV)",
            csv_full,
            file_name="backtest_full_results.csv",
            mime="text/csv",
        )

        with st.expander("Configuration used"):
            st.json(cfg_used)
