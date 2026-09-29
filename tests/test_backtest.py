import numpy as np
import pandas as pd
import pytest

from agentfund import metrics
from agentfund.backtest import month_end_decision_dates, run_strategy


def dates(prices, n=6):
    return list(prices.index[250 : 250 + 20 * n : 20])


def test_buy_and_hold_no_cost_matches_price_ratio(prices):
    ds = dates(prices)
    calls = {"n": 0}

    def decide(d, pf):
        calls["n"] += 1
        return {"A": 1.0} if calls["n"] == 1 else None

    res = run_strategy("bh", prices[["A"]], ds, decide, cost_bps=0)
    buy_day = prices.index[prices.index.get_loc(ds[0]) + 1]  # executed next close
    expected = prices["A"].iloc[-1] / prices["A"].loc[buy_day]
    assert res.equity.iloc[-1] == pytest.approx(expected)


def test_costs_equal_turnover_times_bps(prices):
    ds = dates(prices, 3)
    res = run_strategy("ew", prices, ds, lambda d, pf: {"A": 0.5, "B": 0.5}, cost_bps=10)
    first = res.trades.iloc[0]
    assert first["turnover"] == pytest.approx(
        0.5
    )  # one-way turnover is half of the traded notional
    assert first["cost"] == pytest.approx(1.0 * 10 / 1e4)  # sum|dw| = 1.0 traded
    # later rebalances only trade the drift, so cost is much smaller
    assert res.trades["cost"].iloc[1] < first["cost"]
    free = run_strategy("ew0", prices, ds, lambda d, pf: {"A": 0.5, "B": 0.5}, cost_bps=0)
    assert res.equity.iloc[-1] < free.equity.iloc[-1]


def test_decision_uses_next_day_execution_and_cash_earns_zero(prices):
    ds = dates(prices, 2)
    res = run_strategy("cash", prices, ds, lambda d, pf: {}, cost_bps=10)
    assert (res.equity == 1.0).all()
    assert res.trades["turnover"].sum() == 0


def test_rejects_leverage(prices):
    with pytest.raises(ValueError):
        run_strategy("x", prices, dates(prices, 2), lambda d, pf: {"A": 0.8, "B": 0.8})


def test_drawdown_is_exposed_to_decide(prices):
    seen = []
    run_strategy(
        "x", prices, dates(prices, 4), lambda d, pf: seen.append(pf.drawdown) or {"A": 1.0}
    )
    assert len(seen) == 4 and all(0 <= s < 1 for s in seen)


def test_month_end_dates_exclude_final_day():
    idx = pd.bdate_range("2024-01-01", "2024-03-29")
    ds = month_end_decision_dates(idx, "2024-01-01", "2024-03-29")
    assert [d.strftime("%m-%d") for d in ds] == ["01-31", "02-29"]


def test_metrics_known_values():
    eq = pd.Series([1.0, 1.1, 0.99, 1.2])
    assert metrics.max_drawdown(eq) == pytest.approx(0.99 / 1.1 - 1)
    lo, hi = metrics.wilson(5, 10)
    assert lo < 0.5 < hi and 0.2 < lo and hi < 0.8
    x = np.random.default_rng(0).normal(0.001, 0.01, 500)
    lo, hi = metrics.bootstrap_ci(x, metrics.sharpe)
    assert lo < metrics.sharpe(x) < hi
