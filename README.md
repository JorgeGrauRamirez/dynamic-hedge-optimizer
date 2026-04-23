# ⚓ Dynamic Hedge & Fix Optimiser

**Optimisation-Based Risk Management for Hedging in Dry-Bulk Freight Markets**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dynamic-hedge-optimizer-jg9.streamlit.app)

A quantitative decision-support system that helps a dry-bulk shipping operator decide *how much* and *which* Forward Freight Agreements (FFAs) to trade in order to reduce financial exposure from volatile freight rates. The system dynamically recalibrates at every voyage origin using three complementary LP-based risk models and a market-regime detector, then evaluates all strategies out-of-sample across hundreds of historical voyages.

---

## What problem does it solve?

Dry-bulk freight rates are among the most volatile commodity prices in the world. A ship operator with physical exposure to freight rates (e.g. a Supramax vessel trading on a specific lane) can hedge that exposure by taking short positions in Baltic Exchange FFAs — standardised paper contracts settled against the same indices. The challenge is determining *which* FFAs to trade and *how large* the position should be.

This tool models the joint dynamics of physical and FFA prices, simulates thousands of future scenarios, and solves three independent linear programs to find the hedge weight vector that minimises different risk objectives. All weights are locked in at voyage inception and evaluated on realised market moves — no lookahead.

---

## Application structure

The app has five tabs:

| Tab | Contents |
|-----|----------|
| **📊 Overview** | Dataset summary, selected route/anchor/FFA universe, time-series chart of all physical routes and FFA indices, raw data preview |
| **🎯 Backtest Results** | Headline KPIs, full performance table, P&L violin distributions, cumulative P&L over time, risk/return bar comparison across all six strategies |
| **🔬 Deep Dive** | LP strategy P&L scattered over time with HMM crisis probability overlay, dynamic hedge allocation stacked-area charts, per-voyage results table, downloadable CSVs |
| **🧭 Live Hedge** | One-click calibration on the most recent available data — returns recommended hedge weights for the next voyage (CVaR, MAD, Minimax, MVHR) |
| **📚 Methodology** | Step-by-step explanation of every modelling stage with mathematical formulations |

All parameters are exposed in the sidebar — route, anchor, FFA universe, voyage length, calibration windows, stress correlation, number of Monte Carlo scenarios, CVaR α, weight bounds, and random seed.

---

## Modelling pipeline

Each voyage (backtest or live) runs through the following stages:

### 1 — Regime detection (Hidden Markov Model)

A **2-state Markov-switching model** (Hamilton, 1989) is fitted on weekly log-returns of the anchor FFA index using all available history up to the voyage start date. It identifies two latent states — a low-volatility *normal* regime and a high-volatility *crisis* regime (e.g. COVID-19, supply shocks). The smoothed posterior crisis probability `P(crisis)` is extracted and used in the next step.

### 2 — Blended correlation matrix

The historical FFA–FFA return correlation matrix is blended with a stress matrix (all off-diagonal elements set to the `stress_corr` parameter):

```
Σ_blend = (1 − P_crisis) · Σ_hist  +  P_crisis · Σ_stress
```

This makes the optimiser more conservative when the model detects elevated market stress. If the resulting matrix is not positive semi-definite (due to numerical rounding), it is projected to the nearest PSD matrix before Cholesky decomposition.

### 3 — Spread calibration (Ornstein–Uhlenbeck)

The log-spread between the physical route and the anchor FFA is modelled as a mean-reverting **Ornstein–Uhlenbeck process**:

```
dS_t = θ(μ − S_t) dt + σ_OU dW_t
```

where `S_t = log(Physical_t / Anchor_t)`. Parameters θ, μ, σ_OU are estimated via OLS on the calibration window (default 8 weeks). If the AR(1) coefficient falls outside (0, 1) — meaning the spread is non-stationary or explosive — the voyage is skipped.

### 4 — Monte Carlo simulation

`N` (default 10 000) joint scenarios are simulated over the voyage horizon using daily time steps:

- **FFA paths** — Geometric Brownian Motion with correlated Cholesky-decomposed noise:
  `log F_{t+dt} = log F_t − ½σ²dt + σ√dt · L·ε`
- **Physical price paths** — reconstructed from the simulated anchor path plus the simulated OU spread:
  `Physical_t = Anchor_t · exp(S_t)`

Scenario P&Ls for the physical leg and each FFA instrument are computed as `(Price_end − Price_0) × Volume`.

### 5 — LP optimisation (three risk measures)

Three separate **linear programs** are solved on the same set of N scenarios, each minimising a different risk objective over the hedge weight vector `w`:

| Method | Objective | Reference |
|--------|-----------|-----------|
| **CVaR** | Expected loss in worst α% of scenarios | Rockafellar & Uryasev (2000) |
| **MAD** | Mean absolute deviation of hedged P&L | Konno & Yamazaki (1991) |
| **Minimax** | Worst-case loss across all scenarios | Young (1998) |

All weights are bounded: `0 ≤ wⱼ ≤ w_upper`. The HiGHS backend (via `scipy.optimize.linprog`) is used with sparse constraint matrices.

### 6 — Benchmarks

Three standard benchmarks are evaluated alongside the LP strategies:

- **Unhedged** — full physical exposure, no FFA position
- **Naive 1:1** — short exactly one contract on the anchor FFA
- **MVHR** — Minimum Variance Hedge Ratio: OLS regression of weekly physical price changes on FFA changes over the past 52 weeks; the best single-FFA beta is used

### 7 — Out-of-sample evaluation

Weights from step 5 are locked at voyage inception. Realized P&L is computed from actual market prices observed at the end of each voyage window:

```
PnL = (ΔPhysical − w · ΔFFA) × Volume
```

The backtest rolls through up to 300 historical origin dates (1-week or voyage-length step).

---

## Live hedge tab

The **🧭 Live Hedge** tab runs the exact same pipeline anchored at the most recent valid row of the dataset. Because trailing rows of the physical route can have missing or statistically degenerate data (e.g. AR coefficient ≥ 1), the engine searches backwards up to 10 weeks to find the last anchor point where all OU and GBM parameters are well-defined. A warning is displayed if calibration fell back to an earlier date.

The output is a table of recommended hedge weights per FFA instrument, under each of the three LP models plus the MVHR single-FFA ratio.

---

## Repository structure

```
dynamic-hedge-optimizer/
├── app.py                        # Streamlit front-end — 5 tabs + sidebar
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # CSV ingestion, column detection, working-set build
│   ├── optimizers.py             # CVaR / MAD / Minimax LP solvers (HiGHS backend)
│   ├── backtesting.py            # HMM + OU + GBM + MC + out-of-sample evaluation loop
│   ├── live_hedge.py             # Live hedge engine (most-recent-data calibration)
│   └── visualizations.py        # All Plotly charts
├── data/
│   └── Merged_Shipping_Data.csv  # Weekly merged dataset (physical routes + FFA indices)
├── .streamlit/
│   └── config.toml               # Theme and server config
├── requirements.txt              # Pinned Python dependencies
└── README.md
```

### Module responsibilities

**`data_loader.py`** — reads the CSV, auto-detects physical vs FFA columns by matching against a hard-coded FFA list, forward-fills gaps, and builds the working dataset for a selected route/instrument combination.

**`optimizers.py`** — three standalone LP functions (`optimise_cvar`, `optimise_mad`, `optimise_minimax`) plus `realized_cvar` for post-hoc evaluation. Uses `scipy.sparse` matrices for efficiency at N = 10 000.

**`backtesting.py`** — defines `BacktestConfig` (dataclass with all parameters), implements the HMM regime detector, OU/GBM calibration, Cholesky-correlated Monte Carlo, and the rolling backtest loop `run_backtest`.

**`live_hedge.py`** — `compute_live_hedge` runs the same pipeline on the most recent data. `_try_calibrate` returns `None` on any invalid parameter combination; the outer function retries up to 10 earlier dates so that trailing data gaps do not block the recommendation.

**`visualizations.py`** — all Plotly charts: EDA time series, P&L violin/distribution, cumulative P&L, risk/return bars, hedge allocation stacked areas, advanced model scatter with HMM overlay.

---

## Data schema

`Merged_Shipping_Data.csv` is a weekly-frequency file with a `Date` column followed by physical route assessment columns and Baltic FFA index columns. All price columns are in **USD '000 / day**. The FFA universe is auto-detected from this list:

```
Cape Avg 5TC, Pmx Avg 5TC, Pmx Avg 4TC, Smx Avg 11TC, Smx Avg 10TC, Handysize Avg 7TC
```

Any other column (besides `Date`) is treated as a physical route. A custom dataset with the same schema can be uploaded via the sidebar file uploader.

---

## Configurable parameters

| Group | Parameter | Range | Default | Description |
|-------|-----------|-------|---------|-------------|
| Route | Physical route | any physical column | `cont trip spore/jpn rge (Smax)` | The freight exposure being hedged |
| Route | Anchor FFA | any FFA | `Smx Avg 10TC` | Used in OU spread model and Naive 1:1 benchmark |
| Route | FFA universe | multiselect | all except `Smx Avg 11TC` | Instruments available to the LP optimiser |
| Voyage | Voyage length | 1–12 weeks | 5 | Horizon of each simulated/realised voyage |
| Voyage | Number of voyages | 20–300 | 150 | Historical origin dates to evaluate |
| Voyage | Non-overlapping | on/off | off | Rolling 1-week step if off |
| Voyage | Assumed volume | > 0 MT-days | 65.0 | Scales P&L from $/day to $/voyage |
| Calibration | Advanced window | 4–52 weeks | 8 | OU + GBM lookback |
| Calibration | MVHR window | 8–104 weeks | 52 | Benchmark OLS lookback |
| Regime | Stress correlation | 0.50–1.00 | 0.90 | Crisis-regime correlation level |
| Monte Carlo | N scenarios | 1 000–20 000 | 10 000 | Simulated paths per voyage |
| Optimiser | CVaR α | 0.01–0.20 | 0.05 | Tail level for CVaR objective |
| Optimiser | Hedge upper bound | 0.5–3.0 | 1.5 | Max weight on any single FFA |
| Optimiser | Random seed | any int | 42 | For reproducibility |

---

## Running locally

```bash
git clone https://github.com/<user>/dynamic-hedge-optimizer.git
cd dynamic-hedge-optimizer
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. No API keys or secrets are required — the bundled CSV is loaded directly from the `data/` folder.

---

## Dependencies

```
streamlit  >=1.32, <2.0
pandas     >=2.0,  <3.0
numpy      >=1.24, <3.0
scipy      >=1.11, <2.0
statsmodels>=0.14, <1.0
plotly     >=5.18, <6.0
```

---

## References

- Hamilton, J.D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle.* Econometrica, 57(2), 357–384.
- Rockafellar, R.T. & Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk.* Journal of Risk, 2(3), 21–41.
- Konno, H. & Yamazaki, H. (1991). *Mean-Absolute Deviation Portfolio Optimization Model and Its Applications to Tokyo Stock Market.* Management Science, 37(5), 519–531.
- Young, M.R. (1998). *A Minimax Portfolio Selection Rule with Linear Programming Solution.* Management Science, 44(5), 673–683.
- Alizadeh, A.H. & Nomikos, N.K. (2009). *Shipping Derivatives and Risk Management.* Palgrave Macmillan.

---

## License

This repository is part of a Master's thesis at the Technical University of Denmark (DTU). Reuse with attribution.
