"""
Backtesting engine for the Dynamic Hedge & Fix Optimiser.

For each historical origin date t0 we:

1. Detect the volatility regime using a 2-state Markov-switching model on
   the anchor index (HMM crisis probability).
2. Calibrate an OU process on the log-spread (physical - anchor) and a
   multivariate GBM on FFA log-returns, blending the historical
   correlation matrix with a stress correlation proportional to the
   crisis probability.
3. Run a Monte Carlo simulation of joint FFA + physical paths over
   ``voyage_weeks``.
4. Solve three LP hedge allocations (CVaR, MAD, Minimax) on the simulated
   P&Ls, and compute two benchmarks (Naive 1:1 and 52-week MVHR).
5. Evaluate every strategy out-of-sample against the realised prices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import sqrtm

from .optimizers import optimise_cvar, optimise_mad, optimise_minimax


@dataclass
class BacktestConfig:
    """All user-configurable parameters for a backtest run."""

    target_physical_route: str
    anchor_index_col: str
    ffa_columns: List[str]

    voyage_weeks: int = 5
    n_backtests: int = 150
    assumed_volume: float = 65.0
    stress_corr: float = 0.90
    mvhr_calibration_weeks: int = 52
    calibration_weeks: int = 8
    non_overlapping: bool = False

    n_sims: int = 10000
    cvar_alpha: float = 0.05
    hedge_upper_bound: float = 1.5
    random_seed: int = 42


def _fit_markov_regime(anchor_rets: pd.Series) -> float:
    """Return the smoothed probability of being in the high-volatility
    regime at the most recent observation."""
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
    """Project a correlation matrix to the nearest PSD matrix if needed."""
    eigenvalues = np.linalg.eigvalsh(corr)
    if eigenvalues.min() > 0:
        return corr
    fixed = np.real(sqrtm(corr @ corr))
    d = np.sqrt(np.diag(fixed))
    return fixed / np.outer(d, d)


def run_backtest(
    df_working: pd.DataFrame,
    cfg: BacktestConfig,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> pd.DataFrame:
    """Run the full backtest.

    Parameters
    ----------
    df_working : DataFrame
        Already filtered to the target route + chosen FFA universe and
        with no missing values. Must contain a `Date` column.
    cfg : BacktestConfig
        All simulation / calibration parameters.
    progress_cb : callable, optional
        Called as ``progress_cb(fraction, message)`` every iteration to
        drive a UI progress bar. ``fraction`` is in ``[0, 1]``.
    """
    np.random.seed(cfg.random_seed)

    ffa_cols = list(cfg.ffa_columns)
    target = cfg.target_physical_route
    anchor = cfg.anchor_index_col
    anchor_idx = ffa_cols.index(anchor)

    step_size = cfg.voyage_weeks if cfg.non_overlapping else 1
    min_history = max(cfg.calibration_weeks, cfg.mvhr_calibration_weeks)
    last_possible_start = len(df_working) - cfg.voyage_weeks - 15
    if last_possible_start < min_history:
        raise ValueError(
            "Not enough history for this configuration. Try a shorter "
            "calibration window or fewer voyage weeks."
        )

    results_log: List[dict] = []

    for i in range(cfg.n_backtests):
        t0_idx = last_possible_start - i * step_size
        if t0_idx < min_history:
            break

        if progress_cb is not None:
            progress_cb(
                (i + 1) / cfg.n_backtests,
                f"Backtesting voyage {i + 1}/{cfg.n_backtests}",
            )

        # --- A. Temporal splits ---
        df_train_adv = df_working.iloc[
            t0_idx - cfg.calibration_weeks : t0_idx + 1
        ].copy()
        df_train_mvhr = df_working.iloc[
            t0_idx - cfg.mvhr_calibration_weeks : t0_idx + 1
        ].copy()
        df_voyage = df_working.iloc[t0_idx : t0_idx + cfg.voyage_weeks + 1].copy()
        t0_date = df_train_adv["Date"].iloc[-1]

        # --- B. Regime detection (Markov switching) ---
        hmm_lookback = min(250, t0_idx)
        df_hmm = df_working.iloc[t0_idx - hmm_lookback : t0_idx + 1]
        anchor_rets_hmm = (
            np.log(df_hmm[anchor] / df_hmm[anchor].shift(1)).dropna()
        )
        prob_crisis = _fit_markov_regime(anchor_rets_hmm)

        # --- C. Calibration (OU on log-spread + GBM on FFAs) ---
        log_phys = np.log(df_train_adv[target])
        log_anch = np.log(df_train_adv[anchor])
        spread = log_phys - log_anch

        y_s = spread.iloc[1:].values
        x_s = spread.iloc[:-1].values
        ols_spread = sm.OLS(y_s, sm.add_constant(x_s)).fit()
        beta_s = ols_spread.params[1]
        sigma_s = float(np.std(ols_spread.resid, ddof=2))

        # OU parameter transforms
        theta = -np.log(beta_s) if beta_s > 0 else np.nan
        mu_ou = (
            ols_spread.params[0] / (1 - beta_s) if beta_s < 1 else np.nan
        )
        sigma_ou = (
            sigma_s * np.sqrt(-2 * np.log(beta_s) / (1 - beta_s ** 2))
            if 0 < beta_s < 1
            else np.nan
        )

        ffa_rets_adv = (
            np.log(df_train_adv[ffa_cols] / df_train_adv[ffa_cols].shift(1))
            .dropna()
        )
        sigma_gbm = ffa_rets_adv.std()

        # Correlation blending (historical <-> stress)
        corr_hist = ffa_rets_adv.corr().values
        corr_stress = np.full((len(ffa_cols), len(ffa_cols)), cfg.stress_corr)
        np.fill_diagonal(corr_stress, 1.0)
        corr_matrix = (
            (1 - prob_crisis) * corr_hist + prob_crisis * corr_stress
        )

        # Spot prices at t0
        P0_f = df_train_adv[ffa_cols].iloc[-1].values
        P0_s = spread.iloc[-1]
        P0_p_spot = df_train_adv[target].iloc[-1]

        # Diagnostics — skip voyage if anything is ill-defined
        if (
            P0_p_spot <= 0
            or np.isnan(P0_p_spot)
            or np.isnan(beta_s)
            or not (0 < beta_s < 1)
            or np.isnan(theta)
            or np.isnan(sigma_ou)
            or np.isnan(mu_ou)
            or np.isnan(np.sum(corr_matrix))
        ):
            continue

        corr_matrix = _nearest_psd(corr_matrix)
        L_corr = np.linalg.cholesky(corr_matrix)

        # --- D. Monte Carlo paths (daily steps) ---
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

        # --- E. Scenario P&Ls ---
        M = len(ffa_cols)
        N = cfg.n_sims
        phys_pnl = (S_p_sim - P0_p_spot) * cfg.assumed_volume
        hedge_pnl = (S_f_sim - P0_f) * cfg.assumed_volume

        # --- F. Optimisers ---
        w_cvar = optimise_cvar(
            phys_pnl, hedge_pnl, M, N,
            alpha=cfg.cvar_alpha, w_upper=cfg.hedge_upper_bound,
        )
        w_mad = optimise_mad(
            phys_pnl, hedge_pnl, M, N, w_upper=cfg.hedge_upper_bound,
        )
        w_minimax = optimise_minimax(
            phys_pnl, hedge_pnl, M, N, w_upper=cfg.hedge_upper_bound,
        )

        # --- G. MVHR benchmark (best single-FFA OLS beta on weekly changes) ---
        phys_changes = df_train_mvhr[target].diff().dropna()
        best_beta = 0.0
        best_r2 = -1.0
        best_idx = 0
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

        # --- H. Out-of-sample evaluation ---
        P_start_p = df_voyage[target].iloc[0]
        P_end_p = df_voyage[target].iloc[-1]
        P_start_f = df_voyage[ffa_cols].iloc[0].values
        P_end_f = df_voyage[ffa_cols].iloc[-1].values

        delta_phys = P_end_p - P_start_p
        delta_ffas = P_end_f - P_start_f

        pnl_unhedged = delta_phys * cfg.assumed_volume
        pnl_naive = (delta_phys - delta_ffas[anchor_idx]) * cfg.assumed_volume
        pnl_mvhr = (delta_phys - best_beta * delta_ffas[best_idx]) * cfg.assumed_volume
        pnl_cvar = (delta_phys - np.dot(delta_ffas, w_cvar)) * cfg.assumed_volume
        pnl_mad = (delta_phys - np.dot(delta_ffas, w_mad)) * cfg.assumed_volume
        pnl_minimax = (delta_phys - np.dot(delta_ffas, w_minimax)) * cfg.assumed_volume

        entry = {
            "Date": t0_date,
            "Unhedged": pnl_unhedged,
            "Naive_1to1": pnl_naive,
            "MVHR": pnl_mvhr,
            "CVaR": pnl_cvar,
            "MAD": pnl_mad,
            "Minimax": pnl_minimax,
            "Prob_Crisis": prob_crisis,
            "MVHR_Beta": best_beta,
            "MVHR_FFA": ffa_cols[best_idx],
            "HR_CVaR": float(np.sum(w_cvar)),
            "HR_MAD": float(np.sum(w_mad)),
            "HR_Minimax": float(np.sum(w_minimax)),
        }
        for k, col in enumerate(ffa_cols):
            entry[f"w_cvar_{col}"] = w_cvar[k]
            entry[f"w_mad_{col}"] = w_mad[k]
            entry[f"w_minimax_{col}"] = w_minimax[k]

        results_log.append(entry)

    df_res = (
        pd.DataFrame(results_log)
        .sort_values("Date")
        .reset_index(drop=True)
    )
    return df_res
