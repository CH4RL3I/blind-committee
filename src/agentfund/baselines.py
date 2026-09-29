"""Simple rules the agents must beat to justify their existence."""

from __future__ import annotations

import pandas as pd

from .backtest import Portfolio, Target
from .features import technical_features


def equal_weight(tickers: list[str]):
    w = {t: 1 / len(tickers) for t in tickers}
    return lambda d, pf: dict(w)


def buy_and_hold(ticker: str):
    state = {"done": False}

    def decide(d: pd.Timestamp, pf: Portfolio) -> Target:
        if state["done"]:
            return None
        state["done"] = True
        return {ticker: 1.0}

    return decide


def momentum(prices: pd.DataFrame, top_k: int = 4):
    """12-1 month momentum: hold the top_k names equal-weight, rebalanced monthly."""

    def decide(d: pd.Timestamp, pf: Portfolio) -> Target:
        score = {t: technical_features(prices[t], d)["momentum_12m_ex_1m"] for t in prices.columns}
        top = sorted(score, key=score.get, reverse=True)[:top_k]
        return {t: 1 / top_k for t in top}

    return decide
