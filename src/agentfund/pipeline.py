"""Orchestration: analysts -> portfolio manager -> risk manager -> audit log."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from .agents import ANALYSTS
from .anonymize import Anonymizer
from .audit import AuditLog
from .backtest import Portfolio
from .llm import Backend, LLMError
from .packets import fundamentals_packet, technical_packet
from .pm import llm_pm, rule_pm
from .risk import RiskLimits, RiskManager
from .schemas import AnalystView


@dataclass
class Stats:
    analyst_calls: int = 0
    analyst_retries: int = 0
    analyst_failures: int = 0
    pm_calls: int = 0
    pm_retries: int = 0
    pm_fallbacks: int = 0


@dataclass
class FundConfig:
    name: str = "full"
    analysts: tuple[str, ...] = ("fundamentals", "technical")
    pm: Literal["llm", "rule"] = "llm"
    risk: bool = True
    log_packets: bool = False


@dataclass
class Fund:
    cfg: FundConfig
    backend: Backend
    prices: pd.DataFrame  # universe closes (no benchmark column needed)
    facts: dict[str, dict]
    anon: Anonymizer | None
    audit: AuditLog
    view_cache: dict = field(default_factory=dict)  # shared across configs: analyst work is reused
    limits: RiskLimits = field(default_factory=RiskLimits)
    stats: Stats = field(default_factory=Stats)

    def __post_init__(self):
        self.risk = RiskManager(self.limits)

    def _ident(self, ticker: str) -> str:
        return self.anon.to_id[ticker] if self.anon else ticker

    def _analyse(self, d: pd.Timestamp, ticker: str, name: str) -> dict:
        key = (d, ticker, name)
        if key in self.view_cache:
            return self.view_cache[key]
        asof = d.date()
        if name == "technical":
            packet, feats = technical_packet(ticker, asof, self.prices[ticker], self.anon)
        else:
            packet, feats = fundamentals_packet(ticker, asof, self.facts[ticker], self.anon)
        self.stats.analyst_calls += 1
        try:
            view, attempts = ANALYSTS[name].analyse(self.backend, packet)
            self.stats.analyst_retries += attempts - 1
            rec = {
                "view": view.model_dump(),
                "attempts": attempts,
                "packet": packet,
                "features": feats,
            }
        except LLMError as e:
            self.stats.analyst_failures += 1
            neutral = AnalystView(stance="neutral", confidence=0.0, reasoning="analyst failed")
            rec = {
                "view": neutral.model_dump(),
                "attempts": 0,
                "error": str(e),
                "packet": packet,
                "features": feats,
            }
        self.view_cache[key] = rec
        return rec

    def decide(self, d: pd.Timestamp, pf: Portfolio) -> dict[str, float]:
        tickers = list(self.prices.columns)
        recs = {t: {a: self._analyse(d, t, a) for a in self.cfg.analysts} for t in tickers}
        views = {
            self._ident(t): {a: AnalystView(**r["view"]) for a, r in recs[t].items()}
            for t in tickers
        }
        back = {self._ident(t): t for t in tickers}

        pm_raw, fallback = None, False
        if self.cfg.pm == "llm":
            self.stats.pm_calls += 1
            try:
                w_by_id, dec, attempts = llm_pm(self.backend, views)
                self.stats.pm_retries += attempts - 1
                pm_raw = dec.model_dump()
            except (LLMError, ValueError):
                self.stats.pm_fallbacks += 1
                fallback, w_by_id = True, rule_pm(views)
        else:
            w_by_id = rule_pm(views)
        proposed = {back[i]: w for i, w in w_by_id.items()}
        # Hard constraint even without a risk manager: long-only, no leverage.
        total = sum(proposed.values())
        if not self.cfg.risk and total > 1:
            proposed = {t: w / total for t, w in proposed.items()}

        actions, vetoed = [], []
        if self.cfg.risk:
            rets = self.prices.pct_change().loc[:d].dropna(how="all")
            rd = self.risk.apply(proposed, rets, pf.drawdown)
            final, actions, vetoed = rd.weights, rd.actions, rd.vetoed
        else:
            final = proposed

        self.audit.write(
            {
                "config": self.cfg.name,
                "decision_date": d.date().isoformat(),
                "equity": pf.equity,
                "drawdown": pf.drawdown,
                "id_map": self.anon.to_id if self.anon else None,
                "analysts": {
                    t: {
                        a: {k: v for k, v in r.items() if self.cfg.log_packets or k != "packet"}
                        for a, r in recs[t].items()
                    }
                    for t in tickers
                },
                "pm": {"raw": pm_raw, "fallback_to_rule": fallback, "proposed_weights": proposed},
                "risk": {"enabled": self.cfg.risk, "vetoed": vetoed, "actions": actions},
                "final_weights": {t: round(w, 6) for t, w in final.items() if w > 0},
            }
        )
        return final
