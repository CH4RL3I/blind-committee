"""Figures and tables from saved results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

LABELS = {
    "full": "Agents (full)",
    "equal_weight": "Equal weight",
    "spy_buy_hold": "SPY buy & hold",
    "momentum_top4": "Momentum top-4",
    "no_fundamentals": "Agents without fundamentals",
    "no_technical": "Agents without technical",
    "no_risk_manager": "Agents without risk manager",
    "rule_based_pm": "Rule-based PM (same views)",
}
COLORS = {
    "full": "#1f4e9c",
    "equal_weight": "#d98a1c",
    "spy_buy_hold": "#4a4a4a",
    "momentum_top4": "#2a9d6f",
    "no_fundamentals": "#7f9fd0",
    "no_technical": "#a8b8d8",
    "no_risk_manager": "#c1487a",
    "rule_based_pm": "#8a63c9",
}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def equity_figure(run: Path, out: Path) -> None:
    eq = pd.read_csv(run / "equity.csv", index_col=0, parse_dates=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    groups = [
        ("Agents vs baselines", ["full", "equal_weight", "spy_buy_hold", "momentum_top4"]),
        (
            "Ablations (agent variants)",
            ["full", "no_fundamentals", "no_technical", "no_risk_manager", "rule_based_pm"],
        ),
    ]
    for ax, (title, cols) in zip(axes, groups, strict=True):
        for c in cols:
            ax.plot(
                eq.index, eq[c], label=LABELS[c], color=COLORS[c], lw=2.2 if c == "full" else 1.4
            )
        ax.set_title(title, loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        _style(ax)
    axes[0].set_ylabel("Growth of 1.0 (net of 10 bps per side)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def probe_figure(probe: Path, out: Path) -> None:
    res = json.loads(probe.read_text())
    cells = {(r["mode"], r["window"]): r for r in res["table"]}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    order = [("anonymised", "pre"), ("anonymised", "post"), ("raw", "pre"), ("raw", "post")]
    names = [
        "anonymised\nbefore cutoff",
        "anonymised\nafter cutoff",
        "raw\nbefore cutoff",
        "raw\nafter cutoff",
    ]
    cols = ["#1f4e9c", "#7f9fd0", "#c1487a", "#e3a3bf"]
    for ax, key, title in [
        (axes[0], "id_correct", "Ticker recalled (open-ended guess)"),
        (axes[1], "direction_correct", "Next-month direction correct"),
    ]:
        for i, o in enumerate(order):
            r = cells[o][key]
            lo, hi = r["ci95"]
            ax.bar(i, r["rate"], color=cols[i])
            ax.errorbar(
                i,
                r["rate"],
                yerr=[[r["rate"] - lo], [hi - r["rate"]]],
                color="black",
                capsize=3,
                lw=1,
            )
        ax.set_xticks(range(4), names, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(title, loc="left", fontsize=11)
        _style(ax)
    axes[1].axhline(0.5, color="black", ls=":", lw=1)
    axes[1].text(3.45, 0.51, "coin flip", ha="right", fontsize=8)
    axes[0].set_ylabel("share of samples (95% Wilson interval)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _pct(x: float) -> str:
    return f"{x:+.1%}"


def results_table(run: Path) -> str:
    s = json.loads((run / "summary.json").read_text())["metrics"]
    lines = [
        "| Strategy | Total return | Ann. return | Ann. vol | Sharpe (95% CI) | Max drawdown "
        "| Turnover / rebalance |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in [
        "full",
        "no_fundamentals",
        "no_technical",
        "no_risk_manager",
        "rule_based_pm",
        "equal_weight",
        "spy_buy_hold",
        "momentum_top4",
    ]:
        m = s[k]
        lo, hi = m["sharpe_ci95"]
        lines.append(
            f"| {LABELS[k]} | {_pct(m['total_return'])} | {_pct(m['ann_return'])} | "
            f"{m['ann_vol']:.1%} | {m['sharpe']:.2f} [{lo:.2f}, {hi:.2f}] | "
            f"{m['max_drawdown']:.1%} | {m['avg_turnover_per_rebalance']:.1%} |"
        )
    return "\n".join(lines)


def probe_table(probe: Path) -> str:
    res = json.loads(probe.read_text())
    lines = [
        "| Condition | Window | n | Ticker recalled | Year recalled | Direction correct "
        "(95% CI) | Up-rate (base) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(res["table"], key=lambda r: (r["mode"], r["window"] != "pre")):
        d = r["direction_correct"]
        lines.append(
            f"| {r['mode']} | {'before cutoff' if r['window'] == 'pre' else 'after cutoff'} | "
            f"{r['n']} | {r['id_correct']['rate']:.1%} | {r['year_correct']['rate']:.1%} | "
            f"{d['rate']:.1%} [{d['ci95'][0]:.1%}, {d['ci95'][1]:.1%}] | {r['base_rate_up']:.1%} |"
        )
    return "\n".join(lines)


def example_trace(run: Path, date: str | None = None) -> str:
    """Render one decision from the full-config audit log as readable markdown."""
    recs = [json.loads(line) for line in (run / "audit_full.jsonl").read_text().splitlines()]
    rec = next((r for r in recs if r["decision_date"] == date), None) if date else None
    if rec is None:
        rec = max(recs, key=lambda r: len(r["final_weights"]))
        rec = recs[len(recs) // 2] if not rec["final_weights"] else rec
    ticker = (
        max(rec["final_weights"], key=rec["final_weights"].get)
        if rec["final_weights"]
        else next(iter(rec["analysts"]))
    )
    aid = rec["id_map"][ticker] if rec["id_map"] else ticker
    a = rec["analysts"][ticker]
    L = [
        f"**Decision {rec['decision_date']}** (strategy equity {rec['equity']:.3f}, drawdown "
        f"{rec['drawdown']:.1%}). Asset shown to the models as `{aid}` (revealed here: {ticker}).",
        "",
    ]
    for name in ("fundamentals", "technical"):
        v = a[name]["view"]
        feats = ", ".join(
            f"{k}={val}" for k, val in list(a[name]["features"].items())[:5] if val is not None
        )
        L += [
            f"- **{name.capitalize()} analyst** sees: {feats}, ...",
            f"  - view: **{v['stance']}**, confidence {v['confidence']:.2f}: {v['reasoning']}",
        ]
    pm = rec["pm"]
    prop = pm["proposed_weights"].get(ticker, 0.0)
    L += [
        "",
        f"- **Portfolio manager** proposes {prop:.1%} for this asset"
        + (f": {pm['raw']['rationale']}" if pm["raw"] else " (fell back to the rule-based PM)"),
    ]
    acts = rec["risk"]["actions"] or ["no constraint binding"]
    L += [
        f"- **Risk manager (code)**: {'; '.join(acts)}",
        f"- **Final weight** for {aid}: {rec['final_weights'].get(ticker, 0.0):.1%}; "
        f"portfolio holds {len(rec['final_weights'])} names, "
        f"gross {sum(rec['final_weights'].values()):.1%}",
    ]
    return "\n".join(L)
