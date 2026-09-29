"""Feature extraction. Every function takes `asof` and may only look at data before or at it."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .data.sec import Fact, point_in_time

REVENUE = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
]
NET_INCOME = ["NetIncomeLoss", "ProfitLoss"]
OP_INCOME = ["OperatingIncomeLoss"]
OCF = ["NetCashProvidedByUsedInOperatingActivities"]
EQUITY = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
]
LIABILITIES = ["Liabilities"]


# ---------------------------------------------------------------- technical


def technical_features(close: pd.Series, asof: pd.Timestamp) -> dict[str, float | None]:
    """Price features from closes up to and including `asof`. Ratios only, no price levels."""
    c = close.loc[:asof].dropna()
    if len(c) < 260:
        raise ValueError("need at least 260 trading days of history")

    def ret(n: int) -> float:
        return float(c.iloc[-1] / c.iloc[-1 - n] - 1)

    daily = c.pct_change().dropna()
    delta = c.diff().dropna().iloc[-14:]
    gain, loss = delta.clip(lower=0).mean(), (-delta.clip(upper=0)).mean()
    rsi = 100.0 if loss == 0 else float(100 - 100 / (1 + gain / loss))
    last126 = c.iloc[-126:]
    out = {
        "return_1m": ret(21),
        "return_3m": ret(63),
        "return_6m": ret(126),
        "momentum_12m_ex_1m": float(c.iloc[-22] / c.iloc[-253] - 1),
        "volatility_60d_annualised": float(daily.iloc[-60:].std() * np.sqrt(252)),
        "distance_from_52w_high": float(c.iloc[-1] / c.iloc[-252:].max() - 1),
        "price_vs_200d_average": float(c.iloc[-1] / c.iloc[-200:].mean() - 1),
        "rsi_14": rsi,
        "max_drawdown_6m": float((last126 / last126.cummax() - 1).min()),
    }
    return {k: round(v, 3) for k, v in out.items()}


# ------------------------------------------------------------- fundamentals


def _series(facts: list[Fact], concepts: list[str], lo: int, hi: int) -> dict[date, float]:
    """{period end: value} for duration facts of length lo..hi days, preferring earlier concepts."""
    best: dict[tuple, tuple[int, float]] = {}
    for f in facts:
        if f.concept not in concepts or f.days is None or not lo <= f.days <= hi:
            continue
        rank = concepts.index(f.concept)
        key = (f.start, f.end)
        if key not in best or rank < best[key][0]:
            best[key] = (rank, f.val)
    return dict(sorted(((k[1], v[1]) for k, v in best.items())))


def _instant(facts: list[Fact], concepts: list[str]) -> tuple[date, float] | None:
    rows = [
        (f.end, f.val, concepts.index(f.concept))
        for f in facts
        if f.concept in concepts and f.start is None
    ]
    if not rows:
        return None
    end = max(r[0] for r in rows)
    val = min((r for r in rows if r[0] == end), key=lambda r: r[2])[1]
    return end, val


def _prior_year(series: dict[date, float], end: date) -> float | None:
    for e, v in series.items():
        if abs((end - e).days - 365) <= 20:
            return v
    return None


def _growth(cur: float | None, prev: float | None) -> float | None:
    if cur is None or prev is None or prev <= 0:
        return None
    return round(cur / prev - 1, 3)


def _ratio(a: float | None, b: float | None) -> float | None:
    return None if a is None or not b or b <= 0 else round(a / b, 3)


def fundamental_features(facts_json: dict, asof: date) -> tuple[dict, dict]:
    """(ratios, levels) built only from facts filed strictly before `asof`.

    `ratios` are safe to show an anonymised model; `levels` (absolute USD, absolute dates)
    are only shown in the raw, non-anonymised condition of the leakage probe.
    """
    all_facts = point_in_time(
        facts_json, REVENUE + NET_INCOME + OP_INCOME + OCF + EQUITY + LIABILITIES, asof
    )
    ann = {
        k: _series(all_facts, c, 350, 380)
        for k, c in [("rev", REVENUE), ("ni", NET_INCOME), ("op", OP_INCOME), ("ocf", OCF)]
    }
    qtr = {k: _series(all_facts, c, 80, 100) for k, c in [("rev", REVENUE), ("ni", NET_INCOME)]}

    def last(s: dict[date, float]) -> tuple[date | None, float | None]:
        return (None, None) if not s else list(s.items())[-1]

    a_end, a_rev = last(ann["rev"])
    a_ni = ann["ni"].get(a_end) if a_end else None
    a_op = ann["op"].get(a_end) if a_end else None
    a_ocf = ann["ocf"].get(a_end) if a_end else None
    q_end, q_rev = last(qtr["rev"])
    q_ni = qtr["ni"].get(q_end) if q_end else None
    eq, li = _instant(all_facts, EQUITY), _instant(all_facts, LIABILITIES)
    latest_filed = max((f.filed for f in all_facts), default=None)

    ratios = {
        "revenue_growth_yoy_annual": _growth(
            a_rev, _prior_year(ann["rev"], a_end) if a_end else None
        ),
        "revenue_growth_yoy_latest_quarter": _growth(
            q_rev, _prior_year(qtr["rev"], q_end) if q_end else None
        ),
        "net_income_growth_yoy_latest_quarter": _growth(
            q_ni, _prior_year(qtr["ni"], q_end) if q_end else None
        ),
        "net_margin_annual": _ratio(a_ni, a_rev),
        "operating_margin_annual": _ratio(a_op, a_rev),
        "operating_cash_flow_margin_annual": _ratio(a_ocf, a_rev),
        "return_on_equity_annual": _ratio(a_ni, eq[1] if eq else None),
        "liabilities_to_equity": _ratio(li[1] if li else None, eq[1] if eq else None),
        "days_since_latest_filing": (asof - latest_filed).days if latest_filed else None,
        "days_since_latest_quarter_end": (asof - q_end).days if q_end else None,
    }
    levels = {
        "annual_revenue_usd_bn": None if a_rev is None else round(a_rev / 1e9, 1),
        "annual_net_income_usd_bn": None if a_ni is None else round(a_ni / 1e9, 1),
        "latest_quarter_end": None if q_end is None else q_end.isoformat(),
        "latest_filing_date": None if latest_filed is None else latest_filed.isoformat(),
    }
    return ratios, levels
