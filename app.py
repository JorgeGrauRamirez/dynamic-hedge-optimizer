"""
Dynamic Hedge & Fix Optimiser — Streamlit front-end.

Run locally with:
    streamlit run app.py
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
# Page config + CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dynamic Hedge Optimiser",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }

        /* ── Hero banner — always dark, explicit colours ── */
        .hero-banner {
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
            border-radius: 16px;
            padding: 2.2rem 2.5rem;
            margin-bottom: 1.5rem;
            color: #ffffff;
            box-shadow: 0 8px 32px rgba(0,0,0,0.40);
        }
        .hero-banner h1 {
            margin: 0 0 0.4rem 0;
            font-size: 2rem;
            font-weight: 700;
            color: #ffffff !important;
            letter-spacing: -0.5px;
        }
        .hero-banner .subtitle {
            font-size: 0.92rem;
            color: #a8c8e8;
            margin: 0;
            line-height: 1.6;
        }
        .hero-banner .badge {
            display: inline-block;
            background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.25);
            border-radius: 20px;
            padding: 0.2rem 0.75rem;
            font-size: 0.75rem;
            margin-right: 0.4rem;
            margin-top: 0.7rem;
            color: #cde8ff;
        }

        /* ── Section headers — inherit colour so they work in dark & light ── */
        .section-header {
            font-size: 1.05rem;
            font-weight: 600;
            border-left: 4px solid #4a9eca;
            padding-left: 0.75rem;
            margin: 1.4rem 0 0.8rem 0;
            color: inherit;
        }

        /* ── Metric labels ── */
        [data-testid="stMetricLabel"] {
            font-size: 0.78rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
            font-weight: 600;
        }

        /* ── Info boxes — semi-transparent so they blend with any theme ── */
        .method-box {
            background: rgba(59, 130, 246, 0.08);
            border: 1px solid rgba(59, 130, 246, 0.25);
            border-left: 4px solid #3b82f6;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.8rem;
            font-size: 0.875rem;
            line-height: 1.65;
            color: inherit;
        }
        .method-box strong { color: #60a5fa; }
        .method-box code {
            background: rgba(96, 165, 250, 0.15);
            border-radius: 3px;
            padding: 0.1rem 0.35rem;
            font-size: 0.82rem;
            color: #93c5fd;
        }

        .insight-box {
            background: rgba(16, 185, 129, 0.07);
            border: 1px solid rgba(16, 185, 129, 0.25);
            border-radius: 10px;
            padding: 1rem 1.25rem;
            margin: 0.6rem 0;
            font-size: 0.875rem;
            color: inherit;
        }
        .insight-box .label {
            font-weight: 600;
            color: #34d399;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.3rem;
        }

        /* ── Tag pills ── */
        .tag {
            display: inline-block;
            background: rgba(96, 165, 250, 0.18);
            color: #93c5fd;
            border: 1px solid rgba(96, 165, 250, 0.30);
            border-radius: 12px;
            padding: 0.15rem 0.6rem;
            font-size: 0.75rem;
            font-weight: 500;
            margin: 0.1rem 0.15rem;
        }

        /* ── Sidebar section cards ── */
        .sidebar-section {
            border-radius: 10px;
            border: 1px solid rgba(74, 158, 202, 0.25);
            padding: 0.85rem 0.9rem 0.6rem 0.9rem;
            margin-bottom: 0.75rem;
            background: rgba(74, 158, 202, 0.05);
        }
        .sidebar-section-title {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #4a9eca;
            margin-bottom: 0.55rem;
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }

        /* ── Run status banner ── */
        .run-status {
            background: rgba(234, 179, 8, 0.12);
            border: 1px solid rgba(234, 179, 8, 0.35);
            border-radius: 8px;
            padding: 0.6rem 0.85rem;
            font-size: 0.8rem;
            color: #fbbf24;
            margin-top: 0.5rem;
            text-align: center;
            animation: pulse 1.5s ease-in-out infinite;
        }
        .run-done {
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.35);
            border-radius: 8px;
            padding: 0.6rem 0.85rem;
            font-size: 0.8rem;
            color: #34d399;
            margin-top: 0.5rem;
            text-align: center;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50%       { opacity: 0.55; }
        }

        /* ── Divider ── */
        .fancy-divider {
            height: 2px;
            background: linear-gradient(90deg, #2c5364 0%, #3b82f6 50%, transparent 100%);
            border: none;
            margin: 1.5rem 0;
            border-radius: 2px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h1>⚓ Dynamic Hedge & Fix Optimiser</h1>
        <p class="subtitle">
            Optimisation-based risk management for dry-bulk freight markets —
            combining HMM regime detection, stochastic simulation, and three
            linear-programming risk models (CVaR · MAD · Minimax) to generate
            dynamic hedge allocations across the FFA universe.
        </p>
        <span class="badge">Master Thesis · Jorge S243346</span>
        <span class="badge">Navi Merchants</span>
        <span class="badge">Python · Streamlit · Plotly</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "Merged_Shipping_Data.csv"


@st.cache_data(show_spinner="Loading shipping data…")
def _cached_load(path: str) -> pd.DataFrame:
    return load_data(path)


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # ── Section 1: Data ──────────────────────────────────────────────────
    st.markdown(
        '<div class="sidebar-section">'
        '<div class="sidebar-section-title">📂 &nbsp;1 · Data</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload custom dataset (optional)",
        type=["csv"],
        help=(
            "If left empty, the app uses the bundled `Merged_Shipping_Data.csv`. "
            "Your file must have a `Date` column plus physical and FFA columns "
            "scaled to USD thousands/day."
        ),
    )
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded is not None:
    df = load_data(uploaded)
    data_source = "uploaded file"
elif DEFAULT_DATA_PATH.exists():
    df = _cached_load(str(DEFAULT_DATA_PATH))
    data_source = "bundled dataset"
else:
    st.error(
        "No data found. Place `Merged_Shipping_Data.csv` inside `data/` "
        "or upload a file from the sidebar."
    )
    st.stop()

physical_cols, ffa_cols_all = split_columns(df)
recommended_ffa = get_recommended_ffa_universe(df)


# ---------------------------------------------------------------------------
# Sidebar — parameters
# ---------------------------------------------------------------------------
with st.sidebar:

    # ── Section 2: Route & Instruments ──────────────────────────────────
    st.markdown(
        '<div class="sidebar-section">'
        '<div class="sidebar-section-title">🗺️ &nbsp;2 · Route & Instruments</div>',
        unsafe_allow_html=True,
    )
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
            "included. `Smx Avg 11TC` is excluded by default (data from 2023-05 only)."
        ),
    )
    if anchor_index_col not in ffa_columns:
        ffa_columns = [anchor_index_col] + [c for c in ffa_columns if c != anchor_index_col]
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 3: Voyage & Backtest ────────────────────────────────────
    st.markdown(
        '<div class="sidebar-section">'
        '<div class="sidebar-section-title">🚢 &nbsp;3 · Voyage & Backtest</div>',
        unsafe_allow_html=True,
    )
    voyage_weeks = st.slider("Voyage length (weeks)", 1, 12, 5)
    n_backtests = st.slider("Number of voyages to backtest", 20, 300, 150, step=10)
    non_overlapping = st.checkbox(
        "Non-overlapping voyages",
        value=False,
        help="If unchecked, uses a rolling 1-week step (overlapping voyages).",
    )
    assumed_volume = st.number_input(
        "Assumed voyage volume (MT-days)",
        min_value=1.0,
        max_value=10_000.0,
        value=65.0,
        step=5.0,
        help="Scales P&L from $/day to absolute $ per voyage.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 4: Calibration & Regime ─────────────────────────────────
    st.markdown(
        '<div class="sidebar-section">'
        '<div class="sidebar-section-title">📡 &nbsp;4 · Calibration & Regime</div>',
        unsafe_allow_html=True,
    )
    calibration_weeks = st.slider(
        "Advanced-model calibration window (weeks)",
        4, 52, 8,
        help="Lookback used to calibrate the OU log-spread and GBM volatilities.",
    )
    mvhr_calibration_weeks = st.slider("MVHR calibration window (weeks)", 8, 104, 52)
    stress_corr = st.slider(
        "Stress correlation (crisis regime)",
        0.50, 1.00, 0.90, step=0.05,
        help=(
            "Correlation assumed inside the high-volatility regime. Blended "
            "with the historical matrix using the HMM crisis probability."
        ),
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 5: Monte Carlo & Optimiser ──────────────────────────────
    st.markdown(
        '<div class="sidebar-section">'
        '<div class="sidebar-section-title">🎲 &nbsp;5 · Monte Carlo & Optimiser</div>',
        unsafe_allow_html=True,
    )
    n_sims = st.select_slider(
        "Monte Carlo scenarios",
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
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Run button + status ──────────────────────────────────────────────
    run_btn = st.button("▶  Run Backtest", type="primary", use_container_width=True)
    _sidebar_status = st.empty()   # placeholder for running / done feedback

    if "df_results" in st.session_state and not run_btn:
        n_done = len(st.session_state["df_results"])
        _sidebar_status.markdown(
            f'<div class="run-done">✓ Last run: {n_done} voyages complete</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Guard: at least one FFA
# ---------------------------------------------------------------------------
if not ffa_columns:
    st.warning("Select at least one FFA instrument from the sidebar.")
    st.stop()

df_working = build_working_dataset(df, target_physical_route, ffa_columns)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_backtest, tab_deepdive, tab_methodology = st.tabs(
    ["📊  Overview", "🎯  Backtest Results", "🔬  Deep Dive", "📚  Methodology"]
)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
with tab_overview:

    # Dataset KPIs
    st.markdown('<p class="section-header">Dataset Summary</p>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Data source", data_source.title())
    c2.metric("Weekly observations", f"{len(df):,}")
    c3.metric("From", f"{df['Date'].min():%b %Y}")
    c4.metric("To", f"{df['Date'].max():%b %Y}")
    c5.metric("Clean rows (selected route)", f"{len(df_working):,}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<p class="section-header">Selected Configuration</p>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="method-box">
                <strong>Physical exposure:</strong><br>
                &nbsp;&nbsp;<code>{target_physical_route}</code><br><br>
                <strong>Anchor FFA</strong> (used in OU spread model + Naive 1:1 benchmark):<br>
                &nbsp;&nbsp;<code>{anchor_index_col}</code><br><br>
                <strong>FFA universe</strong> ({len(ffa_columns)} instruments available to the LP optimiser):<br>
                &nbsp;&nbsp;{'&nbsp;'.join(f'<span class="tag">{c}</span>' for c in ffa_columns)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown('<p class="section-header">Market Coverage</p>', unsafe_allow_html=True)
        # Coverage stats per column
        total_rows = len(df)
        coverage_physical = {c: df[c].notna().sum() / total_rows * 100 for c in physical_cols}
        coverage_ffa = {c: df[c].notna().sum() / total_rows * 100 for c in ffa_cols_all}
        avg_phys = np.mean(list(coverage_physical.values()))
        avg_ffa = np.mean(list(coverage_ffa.values()))
        date_span = (df["Date"].max() - df["Date"].min()).days / 365.25

        st.markdown(
            f"""
            <div class="method-box">
                <strong>Physical routes available:</strong> {len(physical_cols)}<br>
                <strong>FFA indices available:</strong> {len(ffa_cols_all)}<br>
                <strong>Time span:</strong> {date_span:.1f} years of weekly data<br>
                <strong>Avg physical data fill:</strong> {avg_phys:.0f}%<br>
                <strong>Avg FFA data fill:</strong> {avg_ffa:.0f}%<br><br>
                <em>Data is forward-filled (ffill) within the working dataset
                to handle missing weeks; rows with persistent NaNs are dropped.</em>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # EDA chart
    st.markdown('<p class="section-header">Exploratory Time Series — Spot vs FFA Markets</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="insight-box">
            <div class="label">📖 How to read this chart</div>
            The top panel shows the raw spot assessments for all 10 physical routes
            (USD '000/day). The bottom panel shows the 6 Baltic FFA benchmark indices.
            Notice how FFA prices tend to lead or converge towards physical prices —
            this co-movement is the foundation of the hedging relationship exploited
            by the optimiser.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        plot_eda(df, physical_cols, ffa_cols_all),
        use_container_width=True,
    )

    with st.expander("📋 Raw data preview (last 50 rows)"):
        st.dataframe(df.tail(50), use_container_width=True, height=350)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — BACKTEST RESULTS
# ═══════════════════════════════════════════════════════════════════════════
with tab_backtest:

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
            _sidebar_status.markdown(
                '<div class="run-status">⏳ Running backtest…</div>',
                unsafe_allow_html=True,
            )
            progress = st.progress(0.0, text="Starting backtest…")
            cfg = BacktestConfig(**cfg_dict)

            def _cb(frac: float, msg: str) -> None:
                progress.progress(min(max(frac, 0.0), 1.0), text=msg)

            try:
                df_results = run_backtest(df_working, cfg, progress_cb=_cb)
            except ValueError as e:
                progress.empty()
                _sidebar_status.markdown(
                    '<div class="run-status">❌ Backtest failed — see error above</div>',
                    unsafe_allow_html=True,
                )
                st.error(str(e))
                st.stop()

            progress.empty()
            st.session_state["df_results"] = df_results
            st.session_state["cfg"] = cfg_dict
            n_done = len(df_results)
            _sidebar_status.markdown(
                f'<div class="run-done">✓ Done — {n_done} voyages</div>',
                unsafe_allow_html=True,
            )

        df_results: pd.DataFrame = st.session_state["df_results"]
        cfg_used: dict = st.session_state["cfg"]

        if df_results.empty:
            st.warning(
                "The backtest produced no valid voyages. Try a different route "
                "or a shorter calibration window."
            )
            st.stop()

        st.success(
            f"✅ Backtest complete — **{len(df_results)} voyages** evaluated on "
            f"`{cfg_used['target_physical_route']}`."
        )

        # -- Headline KPIs --
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
        worst_mean_col = min(ALL_STRATEGY_COLS, key=lambda c: realized_cvar(df_results[c]))

        st.markdown('<p class="section-header">Headline Metrics (CVaR strategy as reference)</p>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Voyages backtested", f"{len(df_results):,}")
        c2.metric(
            "CVaR 5% — CVaR model",
            f"${ref_cvar:,.0f}",
            f"{cvar_reduction:+.1f}% vs Unhedged" if not np.isnan(cvar_reduction) else None,
            delta_color="normal",
        )
        c3.metric(
            "Std Dev — CVaR model",
            f"${ref_std:,.0f}",
            f"{-std_reduction:+.1f}% vs Unhedged" if not np.isnan(std_reduction) else None,
            delta_color="inverse",
        )
        c4.metric(
            "Best mean P&L",
            STRATEGY_LABELS[best_mean_col],
            f"${df_results[best_mean_col].mean():,.0f}",
        )
        c5.metric(
            "Best tail-risk model",
            STRATEGY_LABELS[worst_mean_col],
            f"CVaR ${realized_cvar(df_results[worst_mean_col]):,.0f}",
        )

        # Contextual interpretation
        best_cvar_label = STRATEGY_LABELS[worst_mean_col]
        st.markdown(
            f"""
            <div class="insight-box">
                <div class="label">📊 Result Interpretation</div>
                The <strong>CVaR model</strong> reduced tail risk by <strong>{abs(cvar_reduction):.1f}%</strong>
                and volatility by <strong>{std_reduction:.1f}%</strong> vs holding an unhedged position.
                The best performing strategy by mean P&L was <strong>{STRATEGY_LABELS[best_mean_col]}</strong>,
                and the strongest tail-risk reducer was <strong>{best_cvar_label}</strong>.
                All three LP optimisers are evaluated out-of-sample — weights are solved on
                Monte Carlo scenarios at voyage inception, then evaluated on realized market moves.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Performance table
        st.markdown('<p class="section-header">Performance Metrics by Strategy</p>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="method-box">
                Six strategies are compared: <strong>Unhedged</strong> (baseline),
                <strong>Naive 1:1</strong> (short one contract on the anchor FFA),
                <strong>MVHR</strong> (minimum-variance hedge ratio via 52-week OLS), and three
                LP-optimised strategies — <strong>CVaR</strong>, <strong>MAD</strong>, and <strong>Minimax</strong>.
                All P&L figures are in USD and assume the configured voyage volume.
            </div>
            """,
            unsafe_allow_html=True,
        )
        summary = build_summary_table(df_results)
        st.dataframe(
            summary.style.format("{:,.2f}"),
            use_container_width=True,
        )
        st.download_button(
            "⬇️  Download summary table (CSV)",
            summary.to_csv().encode("utf-8"),
            file_name="summary_metrics.csv",
            mime="text/csv",
        )

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

        # P&L Distribution
        st.markdown('<p class="section-header">P&L Distribution per Voyage</p>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="insight-box">
                <div class="label">📖 How to read</div>
                Each violin shows the full empirical distribution of realized voyage P&L for
                one strategy. The <strong>white dot</strong> is the median, the <strong>thick bar</strong> is the IQR,
                and the outer shape is a kernel-density estimate. Wider tails = higher risk.
                A narrow, right-shifted violin is ideal.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            plot_pnl_distribution(df_results, cfg_used["target_physical_route"]),
            use_container_width=True,
        )

        # Cumulative P&L
        st.markdown('<p class="section-header">Cumulative P&L Over the Backtest Window</p>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="insight-box">
                <div class="label">📖 How to read</div>
                Cumulative sum of per-voyage P&L sorted by voyage date.
                A steeper upward slope indicates consistently positive outcomes.
                Drawdowns (temporary dips) reveal sensitivity to market regimes.
                All strategies start from zero — absolute spread between them
                widens as the backtest progresses.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            plot_cumulative_pnl(df_results, cfg_used["target_physical_route"]),
            use_container_width=True,
        )

        # Risk/Return bar chart
        st.markdown('<p class="section-header">Risk / Return Comparison</p>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="insight-box">
                <div class="label">📖 How to read</div>
                Three panels side-by-side: <strong>Tail Risk</strong> (CVaR 5% — lower is better
                for risk management), <strong>Volatility</strong> (std dev — lower is more stable),
                and <strong>Expected Return</strong> (mean P&L — higher is better).
                The ideal strategy sits in the bottom-left of the first two panels
                and the top of the third.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(plot_risk_summary(df_results), use_container_width=True)

    else:
        st.markdown('<p class="section-header">No results yet</p>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="method-box">
                Configure the parameters in the sidebar and click <strong>▶ Run Backtest</strong>
                to generate strategy comparisons. The backtest will evaluate all six strategies
                across the selected number of historical voyages.
                <br><br>
                <strong>Typical runtime:</strong> ~30 seconds for 150 voyages with 10 000 MC scenarios.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════
with tab_deepdive:

    if "df_results" not in st.session_state:
        st.info("Run a backtest first from the **Backtest Results** tab.")
    else:
        df_results = st.session_state["df_results"]
        cfg_used = st.session_state["cfg"]

        # Advanced models vs time
        st.markdown('<p class="section-header">Advanced Models Over Time + HMM Crisis Probability</p>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="insight-box">
                <div class="label">📖 How to read</div>
                Each dot is the realized P&L for one voyage under one of the three LP-optimised
                strategies. The <strong>red bars</strong> in the background show the HMM-estimated
                probability of being in a <em>crisis / high-volatility</em> regime at voyage inception.
                When crisis probability is high, the optimiser shifts to a more conservative,
                stress-correlated covariance matrix — watch how P&L dispersion changes in those
                periods.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(plot_advanced_vs_time(df_results), use_container_width=True)

        # Hedge allocations
        st.markdown('<p class="section-header">Dynamic Hedge Allocation by Risk Measure</p>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="insight-box">
                <div class="label">📖 How to read</div>
                Each stacked-area panel shows how the LP optimiser allocates hedge weight across
                the FFA universe over time. The <strong>dashed line at 1.0</strong> represents a
                full 1:1 hedge on a single instrument. Weights above 1 indicate <em>over-hedging</em>
                (cross-hedging via correlated instruments); weights at 0 mean the optimiser chose
                to leave that instrument unhedged. Colour represents each FFA contract.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            plot_hedge_allocations(df_results, cfg_used["ffa_columns"]),
            use_container_width=True,
        )

        # Per-voyage table
        st.markdown('<p class="section-header">Per-Voyage Results Table</p>', unsafe_allow_html=True)
        display_cols = (
            ["Date"] + ALL_STRATEGY_COLS +
            ["Prob_Crisis", "MVHR_Beta", "MVHR_FFA", "HR_CVaR", "HR_MAD", "HR_Minimax"]
        )
        st.dataframe(
            df_results[display_cols].style.format({
                **{c: "${:,.0f}" for c in ALL_STRATEGY_COLS},
                "Prob_Crisis": "{:.2%}",
                "MVHR_Beta": "{:.2f}",
                "HR_CVaR": "{:.2f}",
                "HR_MAD": "{:.2f}",
                "HR_Minimax": "{:.2f}",
            }),
            use_container_width=True,
            height=420,
        )
        st.download_button(
            "⬇️  Download full results (CSV)",
            df_results.to_csv(index=False).encode("utf-8"),
            file_name="backtest_full_results.csv",
            mime="text/csv",
        )

        with st.expander("⚙️  Configuration used for this backtest"):
            st.json(cfg_used)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════
with tab_methodology:

    st.markdown('<p class="section-header">Overview — What This Tool Does</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="method-box">
            This tool helps a dry-bulk freight operator decide <em>how much</em> and
            <em>which</em> Forward Freight Agreements (FFAs) to trade in order to
            reduce the financial risk of their physical freight exposure.
            Rather than using a single static hedge ratio, the system dynamically
            recalibrates before every voyage using three complementary risk measures
            and a market-regime detector.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Step 1 — Regime Detection (HMM)</p>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="method-box">
                A <strong>2-state Markov-switching model</strong> (Hamilton, 1989) is fitted
                on the weekly log-returns of the anchor FFA index using all available
                history up to the voyage start date.<br><br>
                The model identifies two latent regimes:
                <ul>
                    <li><strong>Normal</strong> — low volatility, typical freight market conditions</li>
                    <li><strong>Crisis</strong> — high volatility, stress market (e.g. COVID-19, supply shocks)</li>
                </ul>
                The <em>smoothed posterior probability</em> of being in the crisis regime,
                <code>P(crisis)</code>, is used to blend the historical correlation matrix
                with a stress correlation matrix:<br><br>
                <code>Σ_blend = (1 − P_crisis) · Σ_hist + P_crisis · Σ_stress</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<p class="section-header">Step 2 — Spread Calibration (OU Process)</p>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="method-box">
                The log-spread between the physical route and the anchor FFA is modelled
                as an <strong>Ornstein-Uhlenbeck (mean-reverting) process</strong>:<br><br>
                <code>dS_t = θ(μ − S_t) dt + σ_OU dW_t</code><br><br>
                where <code>S_t = log(Physical_t / Anchor_t)</code>.
                Parameters are estimated via OLS regression of <code>S_t</code> on <code>S_{t-1}</code>
                over the calibration window (default 8 weeks):<br>
                <ul>
                    <li><code>θ</code> — mean-reversion speed</li>
                    <li><code>μ</code> — long-run spread equilibrium</li>
                    <li><code>σ_OU</code> — spread diffusion</li>
                </ul>
                This captures the economic intuition that physical and paper prices
                are cointegrated and should not diverge permanently.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown('<p class="section-header">Step 3 — Monte Carlo Simulation (GBM + OU)</p>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="method-box">
                <strong>N scenarios</strong> (default 10 000) of FFA prices and physical prices are simulated
                over the voyage horizon using daily time steps:<br><br>
                <strong>FFA paths</strong> — Geometric Brownian Motion with correlated noise:
                <code>log F_{t+dt} = log F_t − ½σ²dt + σ√dt · ε</code><br><br>
                Correlation across FFAs is drawn from the blended matrix <code>Σ_blend</code>
                and decomposed via <strong>Cholesky factorisation</strong> to ensure
                consistent joint scenarios.<br><br>
                <strong>Physical price paths</strong> — reconstructed from the FFA anchor plus
                the simulated OU spread:
                <code>Physical_t = Anchor_t · exp(S_t)</code><br><br>
                Scenario P&Ls are computed as:
                <code>PnL_sim = (Physical_end − Physical_0) × Volume</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<p class="section-header">Step 4 — LP Optimisation (3 Risk Measures)</p>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="method-box">
                Given the N simulated scenarios, three separate <strong>linear programs</strong>
                are solved to determine the hedge weight vector <code>w</code> over the FFA universe:<br><br>
                <strong>CVaR (Rockafellar & Uryasev, 2000):</strong><br>
                Minimises the expected loss in the worst α% of scenarios.
                <code>min γ + 1/(Nα) Σ u_i</code> subject to linearised tail constraints.<br><br>
                <strong>MAD (Konno & Yamazaki, 1991):</strong><br>
                Minimises the Mean Absolute Deviation of the hedged P&L.
                <code>min 1/N Σ d_i</code> using de-meaned scenario trick to linearise.<br><br>
                <strong>Minimax (Young, 1998):</strong><br>
                Maximises the worst-case voyage P&L across all scenarios.
                <code>min −y</code> subject to <code>w·ΔFFA + y ≤ ΔPhys ∀i</code>.<br><br>
                All weights are bounded: <code>0 ≤ w_j ≤ w_upper</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<p class="section-header">Step 5 — Out-of-Sample Evaluation & Benchmarks</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
            <div class="method-box">
                Optimised weights are <strong>locked in at voyage inception</strong> and evaluated
                on the <em>realized</em> price moves observed over the actual voyage window.
                This ensures there is <strong>no lookahead bias</strong>.<br><br>
                The realized P&L for each strategy is:
                <code>PnL = (ΔPhys − w · ΔFFA) × Volume</code><br><br>
                The backtest rolls through up to <strong>300 historical voyages</strong>,
                each separated by 1 week (or voyage-length if non-overlapping is selected).
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div class="method-box">
                <strong>Three benchmarks are included for comparison:</strong><br><br>
                <strong>Unhedged</strong> — full physical exposure, no FFA position.<br><br>
                <strong>Naive 1:1</strong> — short exactly 1 contract on the anchor FFA.
                This is the simplest possible hedge and a common industry starting point.<br><br>
                <strong>MVHR</strong> — Minimum Variance Hedge Ratio. An OLS regression of
                weekly physical price changes on FFA changes over the past 52 weeks.
                The best single-FFA beta is used. This is the standard econometric benchmark
                from the academic hedging literature.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<p class="section-header">Data & Instruments</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="method-box">
            <strong>Physical routes</strong> — 10 weekly spot assessments for Handysize and
            Supramax dry-bulk vessels on major trade lanes (ECSA, USG, Continent → Far East;
            Canakkale → India). Prices are in <strong>USD '000/day</strong>.<br><br>
            <strong>FFA indices</strong> — 6 Baltic Exchange benchmark indices:
            Cape 5TC, Panamax 5TC, Panamax 4TC, Supramax 10TC / 11TC, Handysize 7TC.
            These are the liquid paper instruments available for hedging.<br><br>
            <strong>Dataset:</strong> Weekly frequency, 2018–present (~400 observations).
            The <code>Smx Avg 11TC</code> index is excluded from the default universe
            because it only started in May 2023 (insufficient history for calibration).
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="section-header">References</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="method-box">
            <strong>Hamilton, J.D. (1989).</strong> A New Approach to the Economic Analysis of
            Nonstationary Time Series and the Business Cycle. <em>Econometrica</em>, 57(2), 357–384.<br><br>
            <strong>Rockafellar, R.T. & Uryasev, S. (2000).</strong> Optimization of Conditional
            Value-at-Risk. <em>Journal of Risk</em>, 2(3), 21–41.<br><br>
            <strong>Konno, H. & Yamazaki, H. (1991).</strong> Mean-Absolute Deviation Portfolio
            Optimization Model and Its Applications to Tokyo Stock Market.
            <em>Management Science</em>, 37(5), 519–531.<br><br>
            <strong>Young, M.R. (1998).</strong> A Minimax Portfolio Selection Rule with Linear
            Programming Solution. <em>Management Science</em>, 44(5), 673–683.<br><br>
            <strong>Alizadeh, A.H. & Nomikos, N.K. (2009).</strong> Shipping Derivatives and Risk
            Management. Palgrave Macmillan.
        </div>
        """,
        unsafe_allow_html=True,
    )
