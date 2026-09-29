"""Performance metrics with honest uncertainty (block bootstrap, Wilson intervals)."""

from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def sharpe(daily: pd.Series | np.ndarray) -> float:
    """Annualised Sharpe with a zero risk-free rate (cash earns nothing in this backtest)."""
    x = np.asarray(daily, dtype=float)
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(ANN)) if sd > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def _block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=k)
    return ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]


def bootstrap_ci(
    x: np.ndarray, stat, y: np.ndarray | None = None, block: int = 21, n: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% circular block-bootstrap interval. With `y`, resamples the paired difference
    stat(x) - stat(y) using identical indices for both series."""
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        idx = _block_indices(len(x), block, rng)
        out[i] = stat(x[idx]) - (stat(y[idx]) if y is not None else 0.0)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def summarize(equity: pd.Series, trades: pd.DataFrame) -> dict:
    daily = equity.pct_change().dropna()
    years = len(daily) / ANN
    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "ann_return": float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0,
        "ann_vol": float(daily.std(ddof=1) * np.sqrt(ANN)),
        "sharpe": sharpe(daily),
        "max_drawdown": max_drawdown(equity),
        "avg_turnover_per_rebalance": float(trades["turnover"].mean()) if len(trades) else 0.0,
        "total_cost_bps_of_equity": float(trades["cost"].sum() * 1e4) if len(trades) else 0.0,
        "n_rebalances": int(len(trades)),
    }
