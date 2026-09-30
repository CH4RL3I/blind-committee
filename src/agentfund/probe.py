"""Leakage probe: can the model recall who and when, and what happened next?

For sample (asset, date) pairs the same evidence packet is shown in two conditions, raw
(ticker, name, absolute date, price and dollar levels) and anonymised. The model is asked to
name the company and year and to state whether the stock rose or fell over the next month.

Reading the result:
* Identification in the anonymised condition measures whether the packet still gives the
  asset away (it should be near chance for a small model, and is the direct leak channel).
* Direction accuracy compared across raw/anonymised and pre/post training cutoff measures
  recall of outcomes. Accuracy that is high before the cutoff, drops after it, and drops
  again under anonymisation would be the signature of memorisation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from . import metrics
from .anonymize import Anonymizer
from .backtest import month_end_decision_dates
from .config import MODEL_CUTOFF, UNIVERSE
from .data.prices import load_prices
from .data.sec import company_facts
from .llm import Backend, LLMError, structured
from .packets import fundamentals_packet, technical_packet
from .schemas import ProbeAnswer

SYSTEM = (
    "You are given a data packet describing one company at one point in time. Using your own "
    "memory of financial markets, answer three questions: the company's stock ticker, the "
    "calendar year of the decision date, and whether the stock rose or fell over the month "
    "after the decision date. If you cannot tell, give your best guess anyway."
)
PRE_DATES = ["2021-04", "2021-10", "2022-04", "2022-10", "2023-04", "2023-10"]
POST_DATES = ["2024-04", "2024-10", "2025-04", "2025-10", "2026-03", "2026-07"]
ALIASES = {"GOOGL": {"GOOG", "GOOGL"}}


def _pick(dates: list[pd.Timestamp], months: list[str]) -> list[pd.Timestamp]:
    by_month = {d.strftime("%Y-%m"): d for d in dates}
    return [by_month[m] for m in months if m in by_month]


def run_probe(backend: Backend, out: Path, log=print, seed: int = 0) -> dict:
    t0 = time.time()
    px = load_prices("2019-06-01", "2026-09-15")[list(UNIVERSE)]
    facts = {t: company_facts(t) for t in UNIVERSE}
    anon = Anonymizer(UNIVERSE, seed=seed)
    dates = month_end_decision_dates(px.index, "2021-01-01", "2026-08-31")
    samples = [(d, "pre") for d in _pick(dates, PRE_DATES)] + [
        (d, "post") for d in _pick(dates, POST_DATES)
    ]
    rows = []
    for d, window in samples:
        i = px.index.get_loc(d)
        fwd = px.iloc[i + 22] / px.iloc[i + 1] - 1
        for t in UNIVERSE:
            for mode, a in (("raw", None), ("anonymised", anon)):
                tp, _ = technical_packet(t, d.date(), px[t], a)
                fp, _ = fundamentals_packet(t, d.date(), facts[t], a)
                try:
                    ans, _ = structured(
                        backend, ProbeAnswer, SYSTEM, f"{fp}\n\n{tp}", max_tokens=120
                    )
                    rec = ans.model_dump()
                except LLMError:
                    rec = {
                        "ticker_guess": "",
                        "year_guess": 0,
                        "next_month_direction": "up",
                        "confidence": 0.0,
                        "failed": True,
                    }
                guess = rec["ticker_guess"].strip().upper().replace("$", "")
                rows.append(
                    {
                        "ticker": t,
                        "date": d.date().isoformat(),
                        "window": window,
                        "mode": mode,
                        "id_correct": guess in ALIASES.get(t, {t}),
                        "year_correct": rec["year_guess"] == d.year,
                        "went_up": bool(fwd[t] > 0),
                        "direction_correct": (rec["next_month_direction"] == "up")
                        == bool(fwd[t] > 0),
                        "predicted_up": rec["next_month_direction"] == "up",
                        "failed": bool(rec.get("failed")),
                        "answer": rec,
                    }
                )
        log(f"probe {d.date()} ({window}) done, {time.time() - t0:.0f}s")

    df = pd.DataFrame(rows)
    table = []
    for (mode, window), g in df.groupby(["mode", "window"]):
        n = len(g)
        row = {"mode": mode, "window": window, "n": n, "failed": int(g.failed.sum())}
        for col in ("id_correct", "year_correct", "direction_correct"):
            k = int(g[col].sum())
            row[col] = {"rate": k / n, "k": k, "ci95": metrics.wilson(k, n)}
        row["base_rate_up"] = float(g.went_up.mean())
        row["share_predicted_up"] = float(g.predicted_up.mean())
        table.append(row)
    result = {
        "assumed_cutoff": MODEL_CUTOFF.isoformat(),
        "backend": backend.name,
        "n_samples_per_cell": len(UNIVERSE) * len(_pick(dates, POST_DATES)),
        "table": table,
        "runtime_seconds": round(time.time() - t0),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    (out.parent / "probe_samples.jsonl").write_text(
        "\n".join(json.dumps(r, default=str) for r in rows) + "\n"
    )
    return result
