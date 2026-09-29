"""Deterministic risk manager. Hard limits live here, in code, and always win over the LLM."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskLimits:
    max_position: float = 0.15  # single-name cap
    max_gross: float = 1.0  # long-only, no leverage
    target_vol: float = 0.15  # annualised; only ever scales exposure down
    vol_lookback: int = 60
    max_asset_vol: float = 0.70  # veto names whose trailing vol is above this
    dd_derisk: float = 0.10  # drawdown from peak that halves exposure
    dd_halt: float = 0.20  # drawdown from peak that vetoes everything (once)


@dataclass
class RiskDecision:
    weights: dict[str, float]
    vetoed: list[str] = field(default_factory=list)
    halted: bool = False
    actions: list[str] = field(default_factory=list)


class RiskManager:
    """Stateful only in one respect: after a full halt it re-enters at half exposure,
    otherwise a strategy sitting in cash could never recover the drawdown and would stay
    halted forever."""

    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()
        self._halted_last = False

    def apply(
        self, proposed: dict[str, float], returns: pd.DataFrame, drawdown: float
    ) -> RiskDecision:
        """`returns`: daily returns up to the decision date. `drawdown`: current drawdown
        of the strategy as a non-negative fraction (0.12 = 12% below peak)."""
        lim = self.limits
        d = RiskDecision(weights={k: max(0.0, float(v)) for k, v in proposed.items()})
        w = d.weights
        if d.weights != proposed:
            d.actions.append("clipped negative weights to zero (long-only)")

        vol = returns.iloc[-lim.vol_lookback :].std() * np.sqrt(252)
        for k in list(w):
            if w[k] > 0 and vol.get(k, 0.0) > lim.max_asset_vol:
                d.vetoed.append(k)
                d.actions.append(f"veto {k}: trailing vol {vol[k]:.0%} > {lim.max_asset_vol:.0%}")
                w[k] = 0.0

        for k in w:
            if w[k] > lim.max_position:
                d.actions.append(f"cap {k}: {w[k]:.1%} -> {lim.max_position:.1%}")
                w[k] = lim.max_position

        gross = sum(w.values())
        if gross > lim.max_gross:
            d.actions.append(f"scale gross exposure {gross:.1%} -> {lim.max_gross:.1%}")
            w = {k: v * lim.max_gross / gross for k, v in w.items()}

        if sum(w.values()) > 0:
            cov = returns.iloc[-lim.vol_lookback :].cov() * 252
            vec = np.array([w.get(c, 0.0) for c in cov.columns])
            pvol = float(np.sqrt(vec @ cov.values @ vec))
            if pvol > lim.target_vol:
                s = lim.target_vol / pvol
                d.actions.append(f"vol target: portfolio vol {pvol:.1%} -> {lim.target_vol:.1%}")
                w = {k: v * s for k, v in w.items()}

        if self._halted_last:
            d.actions.append("re-entering after halt at half exposure")
            w = {k: v * 0.5 for k, v in w.items()}
        elif drawdown >= lim.dd_halt:
            d.halted = True
            d.actions.append(
                f"drawdown stop: {drawdown:.1%} >= {lim.dd_halt:.0%}, veto all, go to cash"
            )
            w = {k: 0.0 for k in w}
        elif drawdown >= lim.dd_derisk:
            d.actions.append(f"drawdown {drawdown:.1%} >= {lim.dd_derisk:.0%}: halve exposure")
            w = {k: v * 0.5 for k, v in w.items()}
        self._halted_last = d.halted
        d.weights = w
        return d
