"""The analysts. Each turns a packet into a validated `AnalystView` via the LLM."""

from __future__ import annotations

from dataclasses import dataclass

from .llm import Backend, structured
from .schemas import AnalystView

FUNDAMENTALS_SYSTEM = (
    "You are a fundamentals analyst at an investment fund. You see reported financial ratios "
    "for one anonymous company, built only from filings public at the decision date. Judge "
    "growth, profitability, cash generation and balance-sheet leverage. Give a stance for the "
    "next month (bullish, neutral or bearish), a confidence between 0 and 1, and reasoning in "
    "at most two short sentences. Use only the data shown. Be sceptical: neutral is a valid "
    "answer when the evidence is mixed."
)
TECHNICAL_SYSTEM = (
    "You are a technical analyst at an investment fund. You see price features for one "
    "anonymous asset, computed only from data up to the decision date. Judge trend, momentum, "
    "volatility and mean-reversion risk. Give a stance for the next month (bullish, neutral or "
    "bearish), a confidence between 0 and 1, and reasoning in at most two short sentences. "
    "Use only the data shown. Be sceptical: neutral is a valid answer when signals conflict."
)


@dataclass(frozen=True)
class Analyst:
    name: str
    system: str

    def analyse(self, backend: Backend, packet: str) -> tuple[AnalystView, int]:
        return structured(backend, AnalystView, self.system, packet, max_tokens=200)


FUNDAMENTALS = Analyst("fundamentals", FUNDAMENTALS_SYSTEM)
TECHNICAL = Analyst("technical", TECHNICAL_SYSTEM)
ANALYSTS = {a.name: a for a in (FUNDAMENTALS, TECHNICAL)}
