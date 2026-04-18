# ⚓ Dynamic Hedge & Fix Optimiser

**Optimisation-Based Risk Management for Hedging in Freight Markets**
*Master Thesis · Jorge (S243346) · Navi Merchants*

A quantitative decision-support system for dry-bulk shipping that bridges
physical freight exposure and financial Forward Freight Agreements (FFAs).
The app benchmarks three LP-based risk measures (CVaR, MAD, Minimax)
against three standard benchmarks (Unhedged, Naive 1:1, MVHR) across
hundreds of backtested voyages.

Live demo (once deployed): `[https://<your-app>.streamlit.app](https://dynamic-hedge-optimizer-jg9.streamlit.app/)`

---

## What the app does

1. **Loads** the merged weekly dataset (physical route assessments + Baltic
   FFA indices, all scaled to USD '000 / day).
2. **Detects** the volatility regime with a 2-state Markov-switching model
   on the anchor index and blends historical and stress correlations by
   the smoothed crisis probability.
3. **Calibrates** per origin date:
   - an Ornstein–Uhlenbeck process on the log-spread (physical − anchor),
   - a multivariate Geometric Brownian Motion on FFA log-returns.
4. **Simulates** 10 000 joint FFA + physical paths per voyage (daily steps).
5. **Optimises** three hedge allocations on the same scenarios:
   - **CVaR** (Rockafellar & Uryasev, 2000)
   - **MAD**  (Konno & Yamazaki, 1991)
   - **Minimax** (Young, 1998)
6. **Evaluates** everything out-of-sample against the realised voyage prices
   and compares against the Unhedged, Naive 1:1 and 52-week MVHR benchmarks.

All parameters are exposed in the sidebar — route, anchor, FFA universe,
voyage length, calibration windows, stress correlation, number of Monte
Carlo scenarios, CVaR α, hedge weight bounds, and random seed.

---

## Repository structure

```
dynamic-hedge-optimizer/
├── app.py                       # Streamlit front-end (3 tabs + sidebar)
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # CSV ingestion, column split, working-set build
│   ├── optimizers.py            # CVaR / MAD / Minimax LPs (HiGHS backend)
│   ├── backtesting.py           # HMM + OU + GBM + MC + out-of-sample eval
│   └── visualizations.py        # Interactive Plotly charts
├── data/
│   └── Merged_Shipping_Data.csv # Weekly merged dataset (physical + FFAs)
├── .streamlit/
│   └── config.toml              # Theme + server config
├── requirements.txt             # Pinned dependencies
├── .gitignore
└── README.md
```

---

## Running locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-user>/dynamic-hedge-optimizer.git
cd dynamic-hedge-optimizer

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Deploying to Streamlit Community Cloud (free)

1. Push this repo to GitHub (see the section below).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   your GitHub account.
3. Click **New app** → select your repo → branch `main` → main file path
   `app.py` → **Deploy**.
4. In ~3 minutes you will get a public URL of the form
   `https://<your-app-name>.streamlit.app` that you can include in your
   thesis and share during your defence.

No secrets or extra configuration are needed — the bundled CSV is read
directly from the repo. If you want to swap in a different dataset, use
the **Upload** widget in the sidebar.

---

## Pushing to GitHub (first time)

```bash
cd dynamic-hedge-optimizer
git init
git add .
git commit -m "Initial commit: Dynamic Hedge & Fix Optimiser"

# Create a new empty repo at https://github.com/new
# (e.g. named "dynamic-hedge-optimizer"), then:
git remote add origin https://github.com/<your-user>/dynamic-hedge-optimizer.git
git branch -M main
git push -u origin main
```

---

## Configurable parameters (sidebar)

| Group | Parameter | Range / options | Default | Notes |
|---|---|---|---|---|
| Data | Uploaded CSV | any `.csv` | bundled file | Same schema as `Merged_Shipping_Data.csv`. |
| Route | Physical route | any physical column | `cont trip spore/jpn rge (Smax)` | Exposure being hedged. |
| Route | Anchor FFA | FFAs with enough history | `Smx Avg 10TC` | Used for OU log-spread + Naive 1:1. |
| Route | FFA universe | multiselect of FFAs | all except `Smx Avg 11TC` | Instruments available to the optimiser. |
| Voyage | Voyage weeks | 1 – 12 | 5 | Horizon of each simulated/realised voyage. |
| Voyage | Number of voyages | 20 – 300 | 150 | How many historical origin dates to evaluate. |
| Voyage | Non-overlapping | on / off | off | If off, rolling 1-week step. |
| Voyage | Assumed volume | > 0 | 65.0 | Scales P&L from $/day to $/voyage. |
| Calibration | Advanced window | 4 – 52 weeks | 8 | OU + GBM lookback. |
| Calibration | MVHR window | 8 – 104 weeks | 52 | Benchmark OLS lookback. |
| Regime | Stress correlation | 0.50 – 1.00 | 0.90 | Crisis-regime correlation level. |
| Monte Carlo | N scenarios | 1 000 – 20 000 | 10 000 | Per voyage. |
| Optimiser | CVaR α | 0.01 – 0.20 | 0.05 | Tail level for the CVaR objective. |
| Optimiser | Hedge upper bound | 0.5 – 3.0 | 1.5 | Max weight on any single FFA. |
| Optimiser | Random seed | any int | 42 | For reproducibility. |

---

## Tabs in the app

1. **📊 Overview** — dataset metrics, selected route/anchor, stacked time
   series of every physical route and FFA, raw data preview.
2. **🎯 Backtest Results** — run the backtest, headline metrics (CVaR 5 %,
   Std Dev, best mean P&L), full summary table, P&L distribution (violin),
   cumulative P&L over time, risk/return horizontal-bar comparison.
3. **🔬 Deep Dive** — advanced models scattered over time with HMM crisis
   probability as shaded bars, dynamic hedge allocations by risk measure
   (stacked areas), per-voyage results table, downloadable CSVs, and the
   exact configuration used for the run.

---

## Data schema

`Merged_Shipping_Data.csv` is a weekly-frequency file with one `Date`
column plus physical route columns and Baltic FFA index columns, all
already scaled to USD '000 / day. The FFA universe is auto-detected from
this hard-coded list:

```
Cape Avg 5TC, Pmx Avg 5TC, Pmx Avg 4TC,
Smx Avg 11TC, Smx Avg 10TC, Handysize Avg 7TC
```

Anything else (that is not `Date`) is treated as a physical route.

---

## References

- Rockafellar, R. T., & Uryasev, S. (2000). *Optimization of Conditional
  Value-at-Risk.* Journal of Risk, 2(3), 21–41.
- Konno, H., & Yamazaki, H. (1991). *Mean-Absolute Deviation Portfolio
  Optimization Model and Its Applications to Tokyo Stock Market.*
  Management Science, 37(5), 519–531.
- Young, M. R. (1998). *A Minimax Portfolio Selection Rule with Linear
  Programming Solution.* Management Science, 44(5), 673–683.
- Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle.* Econometrica,
  57(2), 357–384.

---

## License

This repository is part of a Master's thesis. Reuse with attribution.
