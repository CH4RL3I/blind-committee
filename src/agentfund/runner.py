"""End-to-end backtest: agents, ablations and baselines on one shared price panel."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import baselines, metrics
from .anonymize import Anonymizer
from .audit import AuditLog
from .backtest import Result, month_end_decision_dates, run_strategy
from .config import BENCHMARK, MODEL_CUTOFF, UNIVERSE
from .data.prices import load_prices
from .data.sec import company_facts
from .llm import Backend
from .pipeline import Fund, FundConfig
from .schemas import STANCE_SIGN

AGENT_CONFIGS = [
    FundConfig("full", log_packets=True),
    FundConfig("no_fundamentals", analysts=("technical",)),
    FundConfig("no_technical", analysts=("fundamentals",)),
    FundConfig("no_risk_manager", risk=False),
    FundConfig("rule_based_pm", pm="rule"),
]


def _forward_hits(views: dict, prices: pd.DataFrame, dates: list[pd.Timestamp]) -> dict:
    """Stance hit rates: did bullish/bearish calls beat / miss the universe average over the
    holding period (next trading day after the decision to the next decision's execution)?"""
    days = prices.index
    out: dict[str, dict] = {}
    for name in ("fundamentals", "technical"):
        rows = []
        for i, d in enumerate(dates):
            e0 = days[days.get_loc(d) + 1]
            e1 = (
                days[min(days.get_loc(dates[i + 1]) + 1, len(days) - 1)]
                if i + 1 < len(dates)
                else days[-1]
            )
            fwd = prices.loc[e1] / prices.loc[e0] - 1
            excess = fwd - fwd.mean()
            for t in prices.columns:
                v = views[(d, t, name)]["view"]
                rows.append((STANCE_SIGN[v["stance"]], float(excess[t]), float(fwd[t])))
        arr = np.array(rows)
        act = arr[arr[:, 0] != 0]
        k_ex = int(((act[:, 0] * act[:, 1]) > 0).sum())
        n = len(act)
        lo, hi = metrics.wilson(k_ex, n)
        out[name] = {
            "n_views": len(arr),
            "n_non_neutral": n,
            "share_bullish": float((arr[:, 0] > 0).mean()),
            "share_bearish": float((arr[:, 0] < 0).mean()),
            "hit_rate_vs_universe": k_ex / n if n else None,
            "ci95": [lo, hi],
        }
    return out


def run_backtest(
    start: str,
    end: str,
    backend: Backend,
    out: Path,
    anonymise: bool = True,
    cost_bps: float = 10.0,
    log=print,
) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    prices_all = load_prices(
        "2022-06-01", (pd.Timestamp(end) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    )
    prices_all = prices_all.loc[:end]
    uni = prices_all[list(UNIVERSE)]
    facts = {t: company_facts(t) for t in UNIVERSE}
    anon = Anonymizer(UNIVERSE, seed=0) if anonymise else None
    dates = month_end_decision_dates(prices_all.index, start, end)
    log(
        f"{len(dates)} rebalances, {len(UNIVERSE)} assets, "
        f"first {dates[0].date()} last {dates[-1].date()}"
    )

    results: list[Result] = []
    px = prices_all[[*UNIVERSE, BENCHMARK]]
    results.append(
        run_strategy("equal_weight", uni, dates, baselines.equal_weight(list(uni)), cost_bps)
    )
    results.append(
        run_strategy(
            "spy_buy_hold", px[[BENCHMARK]], dates, baselines.buy_and_hold(BENCHMARK), cost_bps
        )
    )
    results.append(run_strategy("momentum_top4", uni, dates, baselines.momentum(uni), cost_bps))

    view_cache: dict = {}
    fund_stats = {}
    for cfg in AGENT_CONFIGS:
        audit = AuditLog(out / f"audit_{cfg.name}.jsonl")
        fund = Fund(cfg, backend, uni, facts, anon, audit, view_cache=view_cache)
        log(f"running agent config: {cfg.name} ({time.time() - t0:.0f}s elapsed)")
        res = run_strategy(cfg.name, uni, dates, fund.decide, cost_bps)
        results.append(res)
        fund_stats[cfg.name] = fund.stats.__dict__.copy()

    eq = pd.concat({r.name: r.equity for r in results}, axis=1)
    eq.to_csv(out / "equity.csv")
    for r in results:
        r.trades.to_csv(out / f"trades_{r.name}.csv", index=False)

    ew = results[0].daily.to_numpy()
    summary = {}
    for r in results:
        s = metrics.summarize(r.equity, r.trades)
        x = r.daily.reindex(results[0].daily.index).fillna(0).to_numpy()
        s["sharpe_ci95"] = metrics.bootstrap_ci(x, metrics.sharpe)
        if r.name != "equal_weight":
            s["sharpe_diff_vs_equal_weight_ci95"] = metrics.bootstrap_ci(x, metrics.sharpe, ew)
        summary[r.name] = s

    hits = _forward_hits(view_cache, uni, dates)
    meta = {
        "start": start,
        "end": end,
        "n_rebalances": len(dates),
        "n_assets": len(UNIVERSE),
        "anonymised": anonymise,
        "cost_bps_per_side": cost_bps,
        "backend": backend.name,
        "assumed_model_cutoff": MODEL_CUTOFF.isoformat(),
        "runtime_seconds": round(time.time() - t0),
        "n_trading_days": int(len(eq)),
        "fund_stats": fund_stats,
    }
    (out / "summary.json").write_text(
        json.dumps({"meta": meta, "metrics": summary, "stance_hits": hits}, indent=1)
    )
    log(f"done in {time.time() - t0:.0f}s")
    return {"meta": meta, "metrics": summary, "stance_hits": hits}
