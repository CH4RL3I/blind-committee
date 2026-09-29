"""Portfolio manager: weighs analyst views into target weights (LLM), or a rule-based twin."""

from __future__ import annotations

from .llm import Backend, structured
from .schemas import STANCE_SIGN, AnalystView, PMDecision

PM_SYSTEM = (
    "You are the portfolio manager of a long-only equity fund. You receive analyst views on "
    "anonymous assets: a stance, a confidence and reasoning from each analyst. Weigh them, "
    "resolve disagreements, and output a target weight between 0 and 1 for each asset you "
    "want to hold (list only assets with a positive weight). Weights should sum to at most 1; "
    "unallocated weight stays in cash. Guidance: no single asset above 0.15, prefer "
    "diversification, size by conviction, hold cash when views are weak. A separate risk "
    "manager enforces limits after you."
)


def format_views(views: dict[str, dict[str, AnalystView]]) -> str:
    """views[asset_id][analyst_name] -> compact text table for the PM prompt."""
    lines = []
    for asset in sorted(views):
        lines.append(f"{asset}:")
        for name, v in views[asset].items():
            lines.append(f"  {name}: {v.stance} (confidence {v.confidence:.2f}) - {v.reasoning}")
    return "\n".join(lines)


def llm_pm(
    backend: Backend, views: dict[str, dict[str, AnalystView]]
) -> tuple[dict[str, float], PMDecision, int]:
    """Returns (weights by asset id, raw decision, attempts). Unknown ids are rejected
    through the retry loop; missing ids mean zero weight."""
    known = set(views)
    prompt = (
        f"Analyst views:\n{format_views(views)}\n\nValid asset ids: {', '.join(sorted(known))}."
    )
    for _ in range(3):
        dec, attempts = structured(backend, PMDecision, PM_SYSTEM, prompt, max_tokens=400)
        if all(p.id in known for p in dec.positions):
            w: dict[str, float] = {}
            for p in dec.positions:
                w[p.id] = w.get(p.id, 0.0) + p.weight
            return w, dec, attempts
        prompt += f"\nYour previous reply used ids outside {sorted(known)}. Use only valid ids."
    raise ValueError("PM kept returning unknown asset ids")


def rule_pm(views: dict[str, dict[str, AnalystView]]) -> dict[str, float]:
    """Mechanical aggregation of the same views: score = mean(sign * confidence), long the
    positive scores in proportion. The comparison that isolates what the LLM PM adds."""
    score = {
        a: sum(STANCE_SIGN[v.stance] * v.confidence for v in vs.values()) / len(vs)
        for a, vs in views.items()
    }
    pos = {a: s for a, s in score.items() if s > 0}
    tot = sum(pos.values())
    return {a: s / tot for a, s in pos.items()} if tot else {}
