"""Gemeinsames Gedächtnis beider KIs (Burak, 23.10.2026).

Jede KI notiert beim Einstieg Marktlage, Begründung und Art (Signal-/Lern-Trade) in ihre **eigene** Datei
(``snapshots/<ki>.jsonl``, nur anhängen – nie zwei Schreiber pro Datei). Ein einziger Schreib-Dienst (``ai-memory``,
``kitb memory loop``) führt jede Minute diese Notizen mit den abgeschlossenen Trades aus den Freqtrade-Datenbanken
zusammen und schreibt sie in ``shared.sqlite``. Damit kann gleichzeitiges Schreiben die Datenbank nicht beschädigen;
die Quellen (Freqtrade-DBs, Notizdateien) bleiben bei Neustarts erhalten, die Zusammenführung ist idempotent.
Beide KIs lesen das gesamte Gedächtnis (auch die Trades und Verluste der anderen) nur lesend.
"""

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kitb import market_flow, paths

KIS = {"ki_trader": "KI 1 (Muster)", "ki_reward": "KI 2 (Belohnung)", "bot_a": "KI-Bot A", "bot_b": "KI-Bot B"}
PRICE_FEATURES = ["ret_1h", "ret_4h", "ret_24h", "rsi_14", "natr_14", "dist_ema50", "dist_ema200", "rel_volume"]
# + was alle Trader tun und Gesamtmarkt/Stimmung (kitb.market_flow, Burak 23.10.2026). Ältere Einträge ohne diese Werte
# bleiben gültig (fehlende Werte zählen beim Lernen als Durchschnitt).
SNAPSHOT_FEATURES = PRICE_FEATURES + market_flow.ALL_FEATURES

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    ki TEXT NOT NULL, round TEXT NOT NULL, trade_id INTEGER NOT NULL,
    kind TEXT, pair TEXT, timeframe TEXT, open_date TEXT, close_date TEXT, open_rate REAL, close_rate REAL,
    stake_amount REAL, profit_pct REAL, profit_abs REAL, hold_hours REAL, reason TEXT, action TEXT,
    features TEXT, exit_reason TEXT,
    PRIMARY KEY (ki, round, trade_id))
"""


def memory_dir(user_data: Path | None = None) -> Path:
    return (user_data or paths.USER_DATA) / "ai_memory"


def db_path(user_data: Path | None = None) -> Path:
    return memory_dir(user_data) / "shared.sqlite"


def market_snapshot(df: pd.DataFrame, minutes_per_candle: int, pair: str | None = None,
                    user_data: Path | None = None) -> pd.DataFrame:
    """Gleiche Marktmerkmale für beide KIs (unabhängig vom Zeitrahmen), nur aus abgeschlossenen Kerzen.
    Mit ``pair`` zusätzlich Trader-Verhalten und Gesamtmarkt (kitb.market_flow)."""
    per_h = max(1, 60 // minutes_per_candle)
    c = df["close"]
    out = pd.DataFrame(index=df.index)
    out["ret_1h"] = c.pct_change(per_h)
    out["ret_4h"] = c.pct_change(4 * per_h)
    out["ret_24h"] = c.pct_change(24 * per_h)
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi_14"] = 100 - 100 / (1 + up / down.replace(0, np.nan))
    tr = pd.concat([df["high"] - df["low"], (df["high"] - c.shift()).abs(), (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
    out["natr_14"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c
    out["dist_ema50"] = c / c.ewm(span=50, adjust=False).mean() - 1
    out["dist_ema200"] = c / c.ewm(span=200, adjust=False).mean() - 1
    out["rel_volume"] = df["volume"] / df["volume"].rolling(20).mean()
    if pair is not None and "date" in df:
        extra = market_flow.all_features(pair, df["date"], minutes_per_candle, user_data)
        extra.index = out.index
        out = pd.concat([out, extra], axis=1)
    return out.replace([np.inf, -np.inf], np.nan)  # z. B. OI-Änderung bei OI 0 -> ±inf; Modelle vertragen nur NaN


def snapshot(ki: str, *, pair: str, when: Any, kind: str, reason: str, features: dict[str, float],
             action: str = "", timeframe: str = "", user_data: Path | None = None) -> None:
    """Einstieg notieren (nur die eigene Datei der KI, anhängen + fsync -> übersteht Neustarts)."""
    d = memory_dir(user_data) / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    rec = {"pair": pair, "when": pd.Timestamp(when).isoformat(), "kind": kind, "reason": reason, "action": action,
           "timeframe": timeframe, "features": {k: (None if v is None or not np.isfinite(v) else float(v))
                                                for k, v in features.items()}}
    with open(d / f"{ki}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _snapshots(ki: str, user_data: Path | None) -> pd.DataFrame:
    p = memory_dir(user_data) / "snapshots" / f"{ki}.jsonl"
    if not p.exists():
        return pd.DataFrame(columns=["pair", "when", "kind", "reason", "action", "timeframe", "features"])
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:  # halb geschriebene letzte Zeile nach Absturz überspringen
            continue
    df = pd.DataFrame(rows)
    df["when"] = pd.to_datetime(df["when"], utc=True, format="ISO8601")
    return df


def _bot_dbs(ki: str, user_data: Path | None) -> list[tuple[str, Path]]:
    base = (user_data or paths.USER_DATA) / "paper" / ki
    dbs = [(f"Runde {i + 1}", db) for i, db in enumerate(sorted((base / "rounds").glob("round-*.sqlite")))]
    dbs.append((f"Runde {len(dbs) + 1}", base / "tradesv3.dryrun.sqlite"))
    return [(r, db) for r, db in dbs if db.exists()]


def _all_bots() -> list[str]:
    """Alle Paper-Bots: die KIs UND die Regel-Bots (Burak 24.10.2026: KIs lernen aus Daten aus allen Richtungen)."""
    try:
        from kitb import paper

        return list(dict.fromkeys([*KIS, *paper.paper_ids()]))
    except Exception:  # noqa: BLE001 – ohne Projektkonfiguration nur die KIs
        return list(KIS)


def _ensure_columns(con) -> None:
    cols = {r[1] for r in con.execute("PRAGMA table_info(trades)")}
    if "features_1h" not in cols:
        con.execute("ALTER TABLE trades ADD COLUMN features_1h TEXT")


_KLINES: dict[str, tuple[pd.Timestamp, pd.DataFrame]] = {}


def _klines_1h(pair: str, get) -> pd.DataFrame:
    """Binance-Stundenkerzen (öffentlich, ~41 Tage), je Coin höchstens einmal pro Stunde geladen."""
    now = pd.Timestamp.now(tz="UTC")
    hit = _KLINES.get(pair)
    if hit and now - hit[0] < pd.Timedelta(hours=1):
        return hit[1]
    rows = get(f"{market_flow.SPOT}/api/v3/klines", {"symbol": market_flow.symbol(pair), "interval": "1h", "limit": 1000})
    df = pd.DataFrame({"date": pd.to_datetime([r[0] for r in rows], unit="ms", utc=True),
                       **{c: [float(r[i]) for r in rows] for c, i in
                          (("open", 1), ("high", 2), ("low", 3), ("close", 4), ("volume", 5))}})
    _KLINES[pair] = (now, df)
    return df


def fill_features(user_data: Path | None = None, get=None) -> int:
    """Marktlage beim Einstieg für Trades ohne Einstiegsnotiz (Regel-Bots, ältere KI-Trades) nachtragen.

    Gleiche Merkmale wie die Notizen der KIs, aus der letzten ABGESCHLOSSENEN Stundenkerze vor dem Einstieg (kein Blick
    in die Zukunft). Liegt der Einstieg vor den verfügbaren Kerzen, wird „{}“ gespeichert (nicht erneut versuchen)."""
    get = get or market_flow._get
    p = db_path(user_data)
    if not p.exists():
        return 0
    done = 0
    with closing(sqlite3.connect(p, timeout=30)) as con:
        con.execute("PRAGMA busy_timeout = 30000")
        _ensure_columns(con)
        todo = pd.read_sql_query("SELECT ki, round, trade_id, pair, open_date FROM trades "
                                 "WHERE features IS NULL AND features_1h IS NULL", con)
        for pair, part in todo.groupby("pair"):
            try:
                k = _klines_1h(pair, get)
                snap = market_snapshot(k, 60, pair, user_data)
                snap.index = k["date"] + pd.Timedelta(hours=1)  # Wert erst nach Kerzenschluss bekannt
            except Exception:  # noqa: BLE001 – nächster Lauf versucht es erneut
                continue
            for r in part.itertuples():
                opened = pd.Timestamp(r.open_date)
                opened = opened.tz_localize("UTC") if opened.tzinfo is None else opened
                pos = snap.index.searchsorted(opened, side="right") - 1
                feats = {} if pos < 200 else {c: (None if not np.isfinite(v) else float(v))
                                              for c, v in snap.iloc[pos].items()}
                con.execute("UPDATE trades SET features_1h = ? WHERE ki = ? AND round = ? AND trade_id = ?",
                            (json.dumps(feats), r.ki, r.round, r.trade_id))
                done += 1
        con.commit()
    return done


def sync(user_data: Path | None = None) -> int:
    """Einziger Schreiber: abgeschlossene Trades beider KIs + Einstiegsnotizen -> shared.sqlite. Gibt neue Zeilen zurück."""
    out = db_path(user_data)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = 0
    with closing(sqlite3.connect(out, timeout=30)) as con:
        con.execute("PRAGMA busy_timeout = 30000")
        con.execute(_SCHEMA)
        _ensure_columns(con)
        for ki in _all_bots():
            snaps = _snapshots(ki, user_data)
            for rnd, db in _bot_dbs(ki, user_data):
                try:
                    with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as src:
                        t = pd.read_sql_query("SELECT id, pair, open_date, close_date, open_rate, close_rate, "
                                              "stake_amount, close_profit, close_profit_abs, enter_tag, exit_reason "
                                              "FROM trades WHERE is_open = 0", src)
                except Exception:  # noqa: BLE001 – leere/gesperrte DB: beim nächsten Lauf erneut
                    continue
                for _, r in t.iterrows():
                    opened = pd.Timestamp(r["open_date"], tz="UTC") if pd.Timestamp(r["open_date"]).tzinfo is None \
                        else pd.Timestamp(r["open_date"])
                    closed = pd.Timestamp(r["close_date"], tz="UTC") if pd.Timestamp(r["close_date"]).tzinfo is None \
                        else pd.Timestamp(r["close_date"])
                    snap = None
                    if not snaps.empty:  # ohne Notizdatei (z. B. Trades vor dem Update) nur Ergebnis speichern
                        s = snaps[(snaps["pair"] == r["pair"]) & ((snaps["when"] - opened).abs() <= pd.Timedelta(hours=2))]
                        snap = s.iloc[(s["when"] - opened).abs().argsort()].iloc[0] if not s.empty else None
                    tag = str(r["enter_tag"] or "")
                    kind = (snap["kind"] if snap is not None else
                            ("learn" if tag.startswith(("lern", "ki_uebt")) else "signal"))
                    cur = con.execute(
                        "INSERT OR IGNORE INTO trades (ki, round, trade_id, kind, pair, timeframe, open_date, close_date, open_rate, "
                        "close_rate, stake_amount, profit_pct, profit_abs, hold_hours, reason, action, features, exit_reason) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (ki, rnd, int(r["id"]), kind, r["pair"], snap["timeframe"] if snap is not None else "",
                         opened.isoformat(), closed.isoformat(), float(r["open_rate"]), float(r["close_rate"] or 0),
                         float(r["stake_amount"]), float(r["close_profit"] or 0), float(r["close_profit_abs"] or 0),
                         (closed - opened).total_seconds() / 3600, tag or (snap["reason"] if snap is not None else ""),
                         snap["action"] if snap is not None else "",
                         json.dumps(snap["features"]) if snap is not None else None, r["exit_reason"] or ""))
                    new += cur.rowcount
        con.commit()
    return new


def load(user_data: Path | None = None) -> pd.DataFrame:
    """Gesamtes gemeinsames Gedächtnis (nur lesend), Merkmale als Spalten."""
    p = db_path(user_data)
    if not p.exists():
        return pd.DataFrame()
    try:
        with closing(sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=30)) as con:
            df = pd.read_sql_query("SELECT * FROM trades ORDER BY close_date", con)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if df.empty:
        return df
    alt = df["features_1h"] if "features_1h" in df else pd.Series(None, index=df.index)
    feats = pd.DataFrame([json.loads(f) if isinstance(f, str) and f else
                          (json.loads(a) if isinstance(a, str) and a else {}) for f, a in zip(df["features"], alt)],
                         index=df.index)
    return pd.concat([df.drop(columns=[c for c in ("features", "features_1h") if c in df]),
                      feats.reindex(columns=SNAPSHOT_FEATURES)], axis=1)
