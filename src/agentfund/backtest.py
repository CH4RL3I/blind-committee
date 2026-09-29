"""Portfolio accounting and the walk-forward driver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

Target = dict[str, float] | None  # ticker -> weight; None means "do not trade"


class Portfolio:
    """Long-only portfolio with cash (zero return), drifting weights and proportional costs."""

    def __init__(self, tickers: list[str], cost_bps: float):
        self.tickers, self.cost = tickers, cost_bps / 1e4
        self.w = np.zeros(len(tickers))
        self.equity, self.peak = 1.0, 1.0
        self.curve: list[tuple[pd.Timestamp, float]] = []
        self.trades: list[dict] = []

    @property
    def drawdown(self) -> float:
        return 1.0 - self.equity / self.peak

    def step(self, day: pd.Timestamp, rets: np.ndarray) -> None:
        growth = 1.0 + float(self.w @ rets)
        self.w = self.w * (1.0 + rets) / growth
        self.equity *= growth
        self.peak = max(self.peak, self.equity)
        self.curve.append((day, self.equity))

    def rebalance(self, day: pd.Timestamp, target: dict[str, float]) -> None:
        tgt = np.array([target.get(t, 0.0) for t in self.tickers])
        if tgt.sum() > 1 + 1e-9 or (tgt < -1e-12).any():
            raise ValueError("targets must be long-only with gross <= 1")
        traded = float(np.abs(tgt - self.w).sum())  # buys + sells
        self.equity *= 1.0 - traded * self.cost
        self.trades.append(
            {
                "date": day,
                "turnover": traded / 2,
                "cost": traded * self.cost,
                **dict(zip(self.tickers, tgt, strict=True)),
            }
        )
        self.w = tgt
        if self.curve:  # the cost lands on the day's closing equity
            self.curve[-1] = (self.curve[-1][0], self.equity)


@dataclass
class Result:
    name: str
    equity: pd.Series
    trades: pd.DataFrame
    extra: dict = field(default_factory=dict)

    @property
    def daily(self) -> pd.Series:
        return self.equity.pct_change().dropna()


def run_strategy(
    name: str,
    prices: pd.DataFrame,
    decision_dates: list[pd.Timestamp],
    decide: Callable[[pd.Timestamp, Portfolio], Target],
    cost_bps: float = 10.0,
) -> Result:
    """Decide at the close of each decision date, trade at the NEXT trading day's close.

    The equity curve starts at 1.0 on the first decision date (cash until the first trade).
    """
    rets = prices.pct_change().fillna(0.0)
    pf = Portfolio(list(prices.columns), cost_bps)
    days = prices.index
    pos = {d: days.get_loc(d) for d in decision_dates}
    cursor = pos[decision_dates[0]]
    pf.curve.append((days[cursor], 1.0))

    def advance(to: int) -> None:
        nonlocal cursor
        while cursor < to:
            cursor += 1
            pf.step(days[cursor], rets.iloc[cursor].to_numpy())

    for d in decision_dates:
        i = pos[d]
        advance(i)
        target = decide(d, pf)
        if i + 1 >= len(days):
            break
        advance(i + 1)
        if target is not None:
            pf.rebalance(days[i + 1], target)
    advance(len(days) - 1)
    eq = pd.Series(dict(pf.curve), name=name)
    return Result(name, eq, pd.DataFrame(pf.trades))


def month_end_decision_dates(index: pd.DatetimeIndex, start: str, end: str) -> list[pd.Timestamp]:
    """Last trading day of each calendar month in [start, end], with a later day to trade on."""
    s = pd.Series(index, index=index).loc[start:end]
    last = s.groupby([s.index.year, s.index.month]).max()
    return [d for d in last if d != index[-1]]
