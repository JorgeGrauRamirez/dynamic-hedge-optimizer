"""
Live hedge recommendation engine.

Calibrates the full model pipeline on the most recent available data and
returns recommended hedge weights for the next voyage.  Uses the same
regime detection → OU/GBM calibration → Monte Carlo → LP optimisation as
the backtest, but anchored at the last valid row of the working dataset.

Key differences vs. the backtest loop:
- Works backwards from the last row to find a valid calibration window
  (the most recent physical data may have trailing NaNs after ffill is
  applied, so we search for the latest row where all parameters are finite).
- Returns a structured dict with a human-readable ``error`` key instead of
  raising exceptions, so the Streamlit UI can display a friendly message.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import sqrtm

from .optimizers import optimise_cvar, optimise_mad, optimise_minimax
from .backtesting import BacktestConfig


def _fit_markov_regime(anchor_rets: pd.Series) -> float:
    try:
        mod = sm.tsa.MarkovRegression(
            anchor_rets, k_regimes=2, trend="c", switching_variance=True
        )
        res = mod.fit(disp=False)
        sigma_mask = res.params.index.str.contains(r"sigma2\[0\]")
        vol_0 = res.params[sigma_mask].values[0]
        sigma_mask = res.params.index.str.contains(r"sigma2\[1\]")
        vol_1 = res.params[sigma_mask].values[0]
        crisis_regime = 1 if vol_1 > vol_0 else 0
        return float(res.smoothed_marginal_probabilities[crisis_regime].iloc[-1])
    except Exception:
        return 0.0


def _nearest_psd(corr: np.ndarray) -> np.ndarray:
    eigenvalues = np.linalg.eigvalsh(corr)
    if eigenvalues.min() > 0:
        return corr
    fixed = np.real(sqrtm(corr @ corr))
    d = np.sqrt(np.diag(fixed))
    return fixed / np.outer(d, d)


def _try_calibrate(
    df_working: pd.DataFrame,
    t0_idx: int,
    cfg: BacktestConfig,
) -> Optional[dict]:
    """Attempt calibration at a single anchor index t0_idx.

    Returns a result dict on success, or None if any parameter is invalid.
    """
    ffa_cols = list(cfg.ffa_columns)
    target = cfg.target_physical_route
    anchor = cfg.anchor_index_col
    anchor_idx = ffa_cols.index(anchor)

    df_train_adv = df_working.iloc[t0_idx - cfg.calibration_weeks : t0_idx + 1].copy()
    df_train_mvhr = df_working.iloc[t0_idx - cfg.mvhr_calibration_weeks : t0_idx + 1].copy()

    # OU spread calibration
    log_phys = np.log(df_train_adv[target])
    log_anch = np.log(df_train_adv[anchor])
    spread = log_phys - log_anch

    y_s = spread.iloc[1:].values
    x_s = spread.iloc[:-1].values
    if len(y_s) < 4:
        return None

    ols_spread = sm.OLS(y_s, sm.add_constant(x_s)).fit()
    beta_s = ols_spread.params[1]
    sigma_s = float(np.std(ols_spread.resid, ddof=2))

    if not (0 < beta_s < 1):
        return None

    theta = -np.log(beta_s)
    mu_ou = ols_spread.params[0] / (1 - beta_s)
    sigma_ou = sigma_s * np.sqrt(-2 * np.log(beta_s) / (1 - beta_s ** 2))

    if any(np.isnan(v) for v in [theta, mu_ou, sigma_ou]):
        return None

    # GBM calibration
    ffa_rets = np.log(df_train_adv[ffa_cols] / df_train_adv[ffa_cols].shift(1)).dropna()
    if len(ffa_rets) < 4:
        return None
    sigma_gbm = ffa_rets.std()

    # Regime detection (uses all available history up to t0)
    hmm_lookback = min(250, t0_idx)
    df_hmm = df_working.iloc[t0_idx - hmm_lookback : t0_idx + 1]
    anchor_rets_hmm = (
        np.log(df_hmm[anchor] / df_hmm[anchor].shift(1))
        .dropna()
        .reset_index(drop=True)
    )
    prob_crisis = _fit_markov_regime(anchor_rets_hmm)

    # Blended correlation
    corr_hist = ffa_rets.corr().values
    corr_stress = np.full((len(ffa_cols), len(ffa_cols)), cfg.stress_corr)
    np.fill_diagonal(corr_stress, 1.0)
    corr_matrix = (1 - prob_crisis) * corr_hist + prob_crisis * corr_stress

    if np.isnan(corr_matrix).any():
        return None

    P0_f = df_train_adv[ffa_cols].iloc[-1].values
    P0_s = spread.iloc[-1]
    P0_p_spot = df_train_adv[target].iloc[-1]
    as_of_date = df_train_adv["Date"].iloc[-1]

    if P0_p_spot <= 0 or np.isnan(P0_p_spot) or np.any(P0_f <= 0):
        return None

    corr_matrix = _nearest_psd(corr_matrix)
    try:
        L_corr = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        return None

    # Monte Carlo
    np.random.seed(cfg.random_seed)
    steps = cfg.voyage_weeks * 7
    dt = 1.0 / 7.0
    Z_m = np.random.normal(0, 1, (steps, cfg.n_sims, len(ffa_cols)))
    Z_s = np.random.normal(0, 1, (steps, cfg.n_sims))

    log_f_t = np.tile(np.log(P0_f), (cfg.n_sims, 1))
    s_t = np.full(cfg.n_sims, P0_s)
    sg = sigma_gbm.values
    for step in range(steps):
        eps = Z_m[step].dot(L_corr.T)
        log_f_t += -0.5 * sg ** 2 * dt + sg * np.sqrt(dt) * eps
        s_t += theta * (mu_ou - s_t) * dt + sigma_ou * np.sqrt(dt) * Z_s[step]

    S_f_sim = np.exp(log_f_t)
    S_p_sim = S_f_sim[:, anchor_idx] * np.exp(s_t)

    M = len(ffa_cols)
    N = cfg.n_sims
    phys_pnl = (S_p_sim - P0_p_spot) * cfg.assumed_volume
    hedge_pnl = (S_f_sim - P0_f) * cfg.assumed_volume

    w_cvar = optimise_cvar(phys_pnl, hedge_pnl, M, N, alpha=cfg.cvar_alpha, w_upper=cfg.hedge_upper_bound)
    w_mad = optimise_mad(phys_pnl, hedge_pnl, M, N, w_upper=cfg.hedge_upper_bound)
    w_minimax = optimise_minimax(phys_pnl, hedge_pnl, M, N, w_upper=cfg.hedge_upper_bound)

    # MVHR
    phys_changes = df_train_mvhr[target].diff().dropna()
    best_beta, best_r2, best_idx = 0.0, -1.0, 0
    for k, col in enumerate(ffa_cols):
        ffa_changes = df_train_mvhr[col].diff().dropna()
        aligned = pd.concat([phys_changes, ffa_changes], axis=1).dropna()
        if len(aligned) < 3:
            continue
        mod = sm.OLS(
            aligned.iloc[:, 0].values,
            sm.add_constant(aligned.iloc[:, 1].values),
        ).fit()
        if mod.rsquared > best_r2:
            best_r2 = mod.rsquared
            best_beta = float(np.clip(mod.params[1], 0.0, 2.0))
            best_idx = k

    return {
        "error": None,
        "as_of_date": as_of_date,
        "prob_crisis": prob_crisis,
        "regime_label": "Crisis / High-Vol" if prob_crisis >= 0.5 else "Normal / Low-Vol",
        "ffa_columns": ffa_cols,
        "w_cvar": w_cvar,
        "w_mad": w_mad,
        "w_minimax": w_minimax,
        "mvhr_beta": best_beta,
        "mvhr_ffa": ffa_cols[best_idx],
    }


def compute_live_hedge(
    df_working: pd.DataFrame,
    cfg: BacktestConfig,
) -> dict:
    """Return live hedge weights calibrated on the most recent valid data.

    Tries up to 10 look-back positions (each one week earlier) so that
    trailing missing data in the physical route does not block calibration.
    """
    ffa_cols = list(cfg.ffa_columns)
    min_history = max(cfg.calibration_weeks, cfg.mvhr_calibration_weeks)
    n = len(df_working)

    if n < min_history + 1:
        return {"error": f"Not enough history ({n} rows). Need at least {min_history + 1}."}

    # Try from the most recent row backwards up to 10 weeks
    max_attempts = min(10, n - min_history)
    last_error = "Calibration failed for all recent data points."

    for offset in range(max_attempts):
        t0_idx = n - 1 - offset
        if t0_idx < min_history:
            break
        result = _try_calibrate(df_working, t0_idx, cfg)
        if result is not None:
            if offset > 0:
                result["note"] = (
                    f"Most recent row was invalid; calibrated on data "
                    f"{offset} week(s) earlier ({result['as_of_date'].strftime('%d %b %Y')})."
                )
            return result

    return {"error": last_error}
