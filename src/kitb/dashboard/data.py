"""Datenzugriff fürs Dashboard (ohne Streamlit, dadurch testbar). Nur lesend – außer Not-Stopp."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pandas as pd

from kitb import killswitch, paths
from kitb.config import load_project_config
from kitb.data.storage import load_ohlcv_raw
from kitb.data.timeframes import to_timedelta


def data_sources(user_data: Path | None = None) -> list[str]:
    d = (user_data or paths.USER_DATA) / "data"
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


def candles(source: str, pair: str, timeframe: str, last_n: int = 500, user_data: Path | None = None) -> pd.DataFrame:
    df = load_ohlcv_raw((user_data or paths.USER_DATA) / "data" / source, pair, timeframe)
    return df.tail(last_n).reset_index(drop=True)


def price_status(df: pd.DataFrame, timeframe: str, now: pd.Timestamp | None = None) -> dict[str, Any]:
    if df.empty:
        return {"last": None, "age_candles": None, "stale": True}
    now = now or pd.Timestamp.now(tz="UTC")
    last_close_time = df["date"].iloc[-1] + to_timedelta(timeframe)
    age = (now - last_close_time) / to_timedelta(timeframe)
    stale = age > load_project_config()["validation"]["max_staleness_candles"]
    return {"last": float(df["close"].iloc[-1]), "last_candle_close": str(last_close_time),
            "age_candles": round(float(age), 1), "stale": bool(stale)}


def _query(db: Path, sql: str, params: tuple = ()) -> pd.DataFrame:
    if not db.exists():
        return pd.DataFrame()
    with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as con:
        try:
            return pd.read_sql_query(sql, con, params=params)
        except Exception:  # Tabelle fehlt (älterer Stand) -> leer anzeigen statt abstürzen
            return pd.DataFrame()


def registry_table(table: str, limit: int = 200, db: Path | None = None) -> pd.DataFrame:
    allowed = {"source_events", "validation_reports", "backtest_runs", "decisions", "model_versions", "model_events"}
    if table not in allowed:
        raise ValueError(table)
    return _query(db or paths.REGISTRY_DB, f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,))


def latest_data_quality(db: Path | None = None) -> pd.DataFrame:
    return _query(db or paths.REGISTRY_DB,
                  "SELECT source, pair, timeframe, rows, ok, created_at, report FROM validation_reports v "
                  "WHERE id = (SELECT MAX(id) FROM validation_reports w WHERE w.source = v.source "
                  "AND w.pair = v.pair AND w.timeframe = v.timeframe) ORDER BY source, pair, timeframe")


def errors(limit: int = 50, db: Path | None = None) -> pd.DataFrame:
    return _query(db or paths.REGISTRY_DB,
                  "SELECT created_at, exchange, action, status, details FROM source_events "
                  "WHERE status NOT IN ('ok') ORDER BY id DESC LIMIT ?", (limit,))


def reports(kind: str, user_data: Path | None = None) -> list[Path]:
    d = (user_data or paths.USER_DATA) / "reports"
    prefix = {"compare": "compare-", "ml": "ml-compare-"}[kind]
    return sorted((p for p in d.glob(f"{prefix}*") if p.is_dir()), reverse=True) if d.exists() else []


def paper_trades(bot: str, user_data: Path | None = None) -> pd.DataFrame:
    """Trades eines Paper-Bots aus der Freqtrade-Dry-Run-Datenbank (Phase 7)."""
    db = (user_data or paths.USER_DATA) / "paper" / bot / "tradesv3.dryrun.sqlite"
    return _query(db, "SELECT id, pair, is_open, open_date, close_date, open_rate, close_rate, amount, "
                      "stake_amount, close_profit, close_profit_abs, realized_profit, exit_reason, "
                      "fee_open, fee_close FROM trades ORDER BY id")


def paper_summary(bot: str, starting_balance: float, user_data: Path | None = None) -> dict[str, Any]:
    t = paper_trades(bot, user_data)
    if t.empty:
        return {"bot": bot, "trades": 0, "open": 0, "realized_pnl": 0.0, "equity": starting_balance}
    closed = t[t["is_open"] == 0]
    pnl = float(closed["close_profit_abs"].fillna(0).sum())
    return {"bot": bot, "trades": int(len(t)), "open": int((t["is_open"] == 1).sum()),
            "closed": int(len(closed)), "realized_pnl": round(pnl, 2), "equity": round(starting_balance + pnl, 2),
            "winrate": round(float((closed["close_profit_abs"] > 0).mean()), 3) if len(closed) else None}


def risk_state(user_data: Path | None = None) -> dict[str, Any]:
    p = (user_data or paths.USER_DATA) / "paper" / "risk_state.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (OSError, ValueError):
        return {"error": "risk_state.json unlesbar"}


def system_status(user_data: Path | None = None) -> dict[str, Any]:
    from kitb.safety import ALLOWED_RUN_MODES

    project = load_project_config()
    return {"kill_switch": killswitch.info(user_data), "allowed_run_modes": sorted(ALLOWED_RUN_MODES),
            "live_trading": "gesperrt", "agents_enabled": project["agents"]["enabled"],
            "data_sources": data_sources(user_data)}
