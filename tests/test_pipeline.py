import json

import pandas as pd
from conftest import ScriptedBackend, view_json

from agentfund.anonymize import Anonymizer
from agentfund.audit import AuditLog
from agentfund.backtest import run_strategy
from agentfund.pipeline import Fund, FundConfig
from agentfund.pm import rule_pm
from agentfund.schemas import AnalystView

NAMES = {"A": "Acme Corporation", "B": "Beta Inc.", "C": "Gamma Co."}
EMPTY = {"facts": {"us-gaap": {}}}


def make_fund(prices, backend, tmp_path, cfg=None):
    anon = Anonymizer(NAMES)
    facts = {t: EMPTY for t in NAMES}
    return Fund(
        cfg or FundConfig(), backend, prices, facts, anon, AuditLog(tmp_path / "a.jsonl")
    ), anon


def test_risk_manager_overrides_all_in_pm_and_audit_is_written(prices, tmp_path):
    holder = {}

    def fn(system, prompt):
        if "portfolio manager" in system:
            aid = holder["anon"].to_id["A"]
            return json.dumps({"positions": [{"id": aid, "weight": 1.0}], "rationale": "all in"})
        return view_json()

    backend = ScriptedBackend(fn=fn)
    fund, anon = make_fund(prices, backend, tmp_path)
    holder["anon"] = anon
    d = prices.index[300]
    w = fund.decide(d, type("PF", (), {"drawdown": 0.0, "equity": 1.0})())
    assert w["A"] <= 0.15 + 1e-9  # PM asked for 100%; code says 15%
    rec = json.loads((tmp_path / "a.jsonl").read_text().splitlines()[0])
    assert rec["pm"]["proposed_weights"]["A"] == 1.0
    assert rec["final_weights"]["A"] <= 0.15 + 1e-9
    assert any("cap A" in a for a in rec["risk"]["actions"])
    assert set(rec["analysts"]["A"]) == {"fundamentals", "technical"}


def test_pm_never_sees_tickers_or_names(prices, tmp_path):
    backend = ScriptedBackend(
        fn=lambda s, p: (
            view_json()
            if "analyst" in s
            else json.dumps({"positions": [{"id": "ASSET_01", "weight": 0.1}], "rationale": "x"})
        )
    )
    fund, _ = make_fund(prices, backend, tmp_path)
    fund.decide(prices.index[300], type("PF", (), {"drawdown": 0.0, "equity": 1.0})())
    for system, prompt in backend.calls:
        blob = system + prompt
        assert "Acme" not in blob and "Beta Inc" not in blob and '"A"' not in blob


def test_invalid_pm_falls_back_to_rule_based_pm(prices, tmp_path):
    backend = ScriptedBackend(fn=lambda s, p: view_json() if "analyst" in s else "garbage")
    fund, _ = make_fund(prices, backend, tmp_path)
    w = fund.decide(prices.index[300], type("PF", (), {"drawdown": 0.0, "equity": 1.0})())
    assert fund.stats.pm_fallbacks == 1 and sum(w.values()) > 0


def test_rule_pm_scores_and_full_pipeline_backtest(prices, tmp_path):
    views = {
        "X": {"f": AnalystView(stance="bullish", confidence=0.8, reasoning="")},
        "Y": {"f": AnalystView(stance="bearish", confidence=0.9, reasoning="")},
    }
    assert rule_pm(views) == {"X": 1.0}
    fund, _ = make_fund(
        prices,
        ScriptedBackend(fn=lambda s, p: view_json()),
        tmp_path,
        FundConfig("rule", pm="rule"),
    )
    ds = list(prices.index[300:360:20])
    res = run_strategy("rule", prices, ds, fund.decide, 10)
    assert pd.notna(res.equity.iloc[-1]) and len(res.trades) == 3
    assert len((tmp_path / "a.jsonl").read_text().splitlines()) == 3
