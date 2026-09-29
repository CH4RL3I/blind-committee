"""Static configuration: universe, paths, and the assumed model training cutoff."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

# Large, liquid US names across sectors. Company names are used only by the anonymiser
# (to scrub them) and by the raw (non-anonymised) probe condition.
UNIVERSE: dict[str, str] = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com, Inc.",
    "GOOGL": "Alphabet Inc.",
    "META": "Meta Platforms, Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "CVX": "Chevron Corporation",
    "JNJ": "Johnson & Johnson",
    "PG": "The Procter & Gamble Company",
    "KO": "The Coca-Cola Company",
    "HD": "The Home Depot, Inc.",
}
BENCHMARK = "SPY"

# Meta documents a December 2023 knowledge cutoff for Llama 3.2. Decision dates from
# 2024-01-31 onwards produce forward returns that lie entirely after that date.
MODEL_CUTOFF = date(2023, 12, 31)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


def root() -> Path:
    """Working root for data/, results/ and docs/. Override with AGENTFUND_ROOT."""
    return Path(os.environ.get("AGENTFUND_ROOT", Path.cwd()))


def cache_dir() -> Path:
    d = root() / "data" / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d
