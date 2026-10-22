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
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,          -- backtest | paper
    run_ref TEXT,                  -- Ergebnisdatei / Lauf-ID
    bot TEXT, strategy TEXT NOT NULL, model_identifier TEXT, model_version_id INTEGER,
    pair TEXT NOT NULL,
    decided_at TEXT NOT NULL,      -- Schluss der Signalkerze (UTC)
    action TEXT NOT NULL,          -- enter_long | exit_long | hold
    proba_up REAL, do_predict INTEGER,
    explanation TEXT,              -- Feature-Einfluss
    market TEXT,                   -- JSON: Preis, Volatilität, Marktphase …
    features TEXT,                 -- JSON: genutzte Featurewerte (Auswahl)
    outcome TEXT,                  -- JSON: Gewinn, Gebühren, Ausstiegsgrund, Dauer
    risk TEXT,                     -- JSON: Stoploss, ATR %, Positionsgröße
    model_error TEXT               -- z. B. false_positive
);
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    bot TEXT NOT NULL,
    version INTEGER NOT NULL,
    identifier TEXT NOT NULL,
    freqaimodel TEXT NOT NULL,
    config TEXT NOT NULL,          -- JSON: vollständige FreqAI-Overlay-Konfiguration
    config_sha256 TEXT NOT NULL,
    data_fingerprints TEXT NOT NULL,
    eval_timerange TEXT NOT NULL,
    metrics TEXT NOT NULL,         -- JSON
    resources TEXT NOT NULL,       -- JSON
    criteria TEXT,                 -- JSON: Ergebnis der Freigabeprüfung
    status TEXT NOT NULL,          -- candidate | passed | failed | approved | active | retired
    approved_by TEXT,
    UNIQUE (bot, version)
);
CREATE TABLE IF NOT EXISTS model_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    model_version_id INTEGER NOT NULL,
    event TEXT NOT NULL,
    actor TEXT NOT NULL,
    details TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS strategy_approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    event TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ai_filter_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    n_trades INTEGER NOT NULL,
    n_holdout INTEGER NOT NULL,
    metrics TEXT NOT NULL,
    importances TEXT NOT NULL,
    model_file TEXT NOT NULL
);
"""

TABLES = ("source_events", "validation_reports", "backtest_runs", "decisions", "model_versions", "model_events",
          "strategy_approvals", "ai_filter_versions")


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

    def insert(self, table: str, row: dict[str, Any]) -> int:
        if table not in TABLES:
            raise ValueError(table)
        return self._insert(table, {"created_at": _now(), **row})

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with closing(self._connect()) as con:
            return [dict(r) for r in con.execute(sql, params).fetchall()]

    def execute(self, sql: str, params: tuple = ()) -> None:
        with closing(self._connect()) as con, con:
            con.execute(sql, params)

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
        if table not in TABLES:
            raise ValueError(table)
        with closing(self._connect()) as con:
            rows = con.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
