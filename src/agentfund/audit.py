"""Append-only JSONL audit trail: one record per rebalance decision."""

from __future__ import annotations

import json
from pathlib import Path


class AuditLog:
    def __init__(self, path: Path | None):
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")

    def write(self, record: dict) -> None:
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")
