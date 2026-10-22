"""Nachvollziehbare Ablage von Datenabrufen, Validierungen und Backtests in SQLite."""

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS source_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    exchange TEXT NOT NULL,
    action TEXT NOT NULL,          -- probe | download | synthetic
    status TEXT NOT NULL,          -- ok | failed
    details TEXT NOT NULL          -- JSON
);
CREATE TABLE IF NOT EXISTS validation_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    pair TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    rows INTEGER NOT NULL,
    ok INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,
    report TEXT NOT NULL           -- JSON
);
CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    strategy TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timerange TEXT,
    data_source TEXT NOT NULL,     -- z. B. synthetic | binance
    synthetic INTEGER NOT NULL,
    data_fingerprints TEXT NOT NULL,   -- JSON {pair: sha256}
    config_sha256 TEXT NOT NULL,
    cost_model TEXT NOT NULL,      -- JSON
    freqtrade_version TEXT NOT NULL,
    result_file TEXT,
    metrics TEXT NOT NULL          -- JSON
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Registry:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as con:
            con.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    def _insert(self, table: str, row: dict[str, Any]) -> int:
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with closing(self._connect()) as con, con:
            cur = con.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(row.values()))
            return int(cur.lastrowid)

    def log_source_event(self, exchange: str, action: str, status: str, details: dict) -> int:
        return self._insert("source_events", {
            "created_at": _now(), "exchange": exchange, "action": action, "status": status,
            "details": json.dumps(details, default=str)})

    def add_validation(self, report: dict[str, Any]) -> int:
        return self._insert("validation_reports", {
            "created_at": _now(), "source": report["source"], "pair": report["pair"],
            "timeframe": report["timeframe"], "rows": report["rows"], "ok": int(report["ok"]),
            "fingerprint": report["fingerprint"], "report": json.dumps(report, default=str)})

    def add_backtest(self, **row: Any) -> int:
        row = dict(row)
        for key in ("data_fingerprints", "cost_model", "metrics"):
            row[key] = json.dumps(row[key], default=str, sort_keys=True)
        row["synthetic"] = int(row["synthetic"])
        return self._insert("backtest_runs", {"created_at": _now(), **row})

    def fetch(self, table: str, limit: int = 20) -> list[dict[str, Any]]:
        if table not in ("source_events", "validation_reports", "backtest_runs"):
            raise ValueError(table)
        with closing(self._connect()) as con:
            rows = con.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
