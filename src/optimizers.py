"""
Linear-programming optimisers for hedge allocation.

Three risk measures are implemented, each as a separate LP that shares the
same inputs (simulated P&L scenarios) and returns a vector of hedge weights
`w` of length M (number of FFA instruments):

* CVaR — Rockafellar & Uryasev (2000)
* MAD  — Konno & Yamazaki (1991)
* Minimax — Young (1998)

All three are solved with `scipy.optimize.linprog` using the HiGHS backend
and sparse constraint matrices so they scale to thousands of scenarios.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


def optimise_cvar(
    phys_pnl: np.ndarray,
    hedge_pnl: np.ndarray,
    M: int,
    N: int,
    alpha: float = 0.05,
    w_upper: float = 1.5,
) -> np.ndarray:
    """CVaR minimisation (Rockafellar & Uryasev, 2000).

    Minimises the average loss in the worst ``alpha`` fraction of scenarios.

    Decision variables: ``[w_1..w_M, gamma, u_1..u_N]``.

        min  gamma + (1 / (N * alpha)) * sum_i u_i
        s.t. sum_j w_j * DeltaFFA_{j,i} - gamma - u_i <= DeltaPhys_i  for all i
             u_i >= 0,  0 <= w_j <= w_upper,  gamma free.
    """
    c = np.zeros(M + 1 + N)
    c[M] = 1.0
    c[M + 1 :] = 1.0 / (N * alpha)

    A_ub = sp.hstack(
        [
            sp.csr_matrix(hedge_pnl),
            sp.csr_matrix(np.full((N, 1), -1.0)),
            -sp.eye(N, format="csr"),
        ],
        format="csr",
    )
    b_ub = phys_pnl

    bounds = [(0.0, w_upper)] * M + [(None, None)] + [(0.0, None)] * N

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return res.x[:M] if res.success else np.zeros(M)


def optimise_mad(
    phys_pnl: np.ndarray,
    hedge_pnl: np.ndarray,
    M: int,
    N: int,
    w_upper: float = 1.5,
) -> np.ndarray:
    """MAD minimisation (Konno & Yamazaki, 1991).

    Minimises the mean absolute deviation of portfolio P&L from its mean,
    penalising both upside and downside deviations equally.

    Uses the de-meaned scenario trick so the LP is linear in ``w`` and
    ``d``:

        min  (1 / N) * sum_i d_i
        s.t.  sum_j w_j * A_{i,j} - d_i <=  b_i
             -sum_j w_j * A_{i,j} - d_i <= -b_i
             d_i >= 0,  0 <= w_j <= w_upper

    where ``A = Hedge - mean(Hedge)`` and ``b = Phys - mean(Phys)``.
    """
    hedge_dm = hedge_pnl - hedge_pnl.mean(axis=0)
    phys_dm = phys_pnl - phys_pnl.mean()

    c = np.zeros(M + N)
    c[M:] = 1.0 / N

    A1 = sp.hstack(
        [sp.csr_matrix(hedge_dm), -sp.eye(N, format="csr")], format="csr"
    )
    A2 = sp.hstack(
        [sp.csr_matrix(-hedge_dm), -sp.eye(N, format="csr")], format="csr"
    )
    A_ub = sp.vstack([A1, A2], format="csr")
    b_ub = np.concatenate([phys_dm, -phys_dm])

    bounds = [(0.0, w_upper)] * M + [(0.0, None)] * N

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return res.x[:M] if res.success else np.zeros(M)


def optimise_minimax(
    phys_pnl: np.ndarray,
    hedge_pnl: np.ndarray,
    M: int,
    N: int,
    w_upper: float = 1.5,
) -> np.ndarray:
    """Minimax optimisation (Young, 1998).

    Maximises the worst-case portfolio P&L across all scenarios, i.e.
    minimises the worst-case loss.

    Decision variables: ``[w_1..w_M, y]``, rewritten as a minimisation:

        min  -y
        s.t. sum_j w_j * DeltaFFA_{j,i} + y <= DeltaPhys_i   for all i
             0 <= w_j <= w_upper,  y free.
    """
    c = np.zeros(M + 1)
    c[M] = -1.0

    A_ub = sp.hstack(
        [sp.csr_matrix(hedge_pnl), sp.csr_matrix(np.ones((N, 1)))],
        format="csr",
    )
    b_ub = phys_pnl

    bounds = [(0.0, w_upper)] * M + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return res.x[:M] if res.success else np.zeros(M)


def realized_cvar(series, alpha: float = 0.05) -> float:
    """Empirical CVaR at the ``alpha`` tail of a realised P&L series."""
    threshold = series.quantile(alpha)
    tail = series[series <= threshold]
    return float(tail.mean()) if len(tail) > 0 else float("nan")
