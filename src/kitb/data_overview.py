"""Übersicht „Wie viele Daten nutzt die KI?“ (Burak, 23.10.2026).

Zwei Teile:
1. **Eingeführte Daten:** alle Datenquellen aus dem Internet (Kurse, Trader-Verhalten, Gesamtmarkt, Backtest-Historie)
   mit Zeilenzahl und Zeitraum.
2. **Eigene Trades:** wie viele Lerndaten die KIs durch ihre eigenen Trades gewinnen (gesamt, letzte 24 h, heute).

Der Dienst ``market-flow`` schreibt die Übersicht alle paar Minuten nach ``user_data/market_flow/overview.json``;
Dashboard und Desktop-App lesen nur diese Datei.
"""

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from kitb import market_flow, paths, shared_memory
from kitb.config import load_project_config

SOURCE_TEXT = {
    "taker": "Kaufdruck (Binance, 15 Min.)", "funding": "Finanzierungsrate Terminmarkt (Binance)",
    "ratio": "Long/Short + offene Positionen (Binance)", "depth": "Orderbuch (Binance, ab jetzt aufgezeichnet)",
    "news": "Schlagzeilen mit Stimmung (Google News, ab jetzt aufgezeichnet)",
    "alpha": "Formel-Faktoren: Rang unter allen Coins (aus Tageskerzen)",
    "fear_greed": "Krypto Fear & Greed", "vix": "VIX (FRED)", "us10y": "US-Zinsen 10 J. (FRED)",
    "dollar": "Dollar-Index (FRED)", "spx": "S&P 500 (FRED)", "ndx": "Nasdaq 100 (FRED)",
    "dvol_btc": "Volatilitätsindex BTC (Deribit)", "dvol_eth": "Volatilitätsindex ETH (Deribit)",
    "stablecoins": "Stablecoin-Menge (DefiLlama)", "btc_active": "Aktive BTC-Adressen (Coin Metrics)",
    "btc_mvrv": "BTC MVRV (Coin Metrics)", "eth_active": "Aktive ETH-Adressen (Coin Metrics)",
}
KI_NAMES = {"ki_trader": "KI 1 (Muster)", "ki_reward": "KI 2 (Belohnung)", "bot_a": "KI-Bot A", "bot_b": "KI-Bot B"}


def _rows(p: Path) -> int:
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(p).metadata.num_rows)
    except Exception:  # noqa: BLE001
        return 0


def _span(p: Path) -> tuple[str, str]:
    try:
        d = pd.read_parquet(p, columns=["date"])["date"]
        return f"{pd.Timestamp(d.min()):%d.%m.%Y}", f"{pd.Timestamp(d.max()):%d.%m.%Y %H:%M}"
    except Exception:  # noqa: BLE001
        return "", ""


def imported(user_data: Path | None = None, exchange: str = "binance") -> list[dict[str, Any]]:
    ud = user_data or paths.USER_DATA
    project = load_project_config()
    pairs = project.get("learning_pairs") or project["allowed_pairs"]
    out: list[dict[str, Any]] = []
    price_dir = ud / "data" / exchange
    for tf in ("15m", "1h", "4h", "1d"):
        files = [price_dir / f"{p.replace('/', '_')}-{tf}.parquet" for p in pairs]
        files = [f for f in files if f.exists()]
        if files:
            frm, to = _span(files[0])
            out.append({"Quelle": f"Kurse {tf} (Binance)", "Coins": len(files), "Zeilen": sum(_rows(f) for f in files),
                        "von": frm, "bis": to})
    fd = market_flow.flow_dir(ud)
    for kind in market_flow.KINDS:
        files = [f for p in pairs if (f := fd / f"{market_flow.symbol(p)}-{kind}.parquet").exists()]
        if files:
            frm, to = _span(files[0])
            out.append({"Quelle": SOURCE_TEXT[kind], "Coins": len(files), "Zeilen": sum(_rows(f) for f in files),
                        "von": frm, "bis": to})
    for kind in ("fear_greed", *market_flow.FRED_SERIES.values(), *market_flow.EXTRA_CONTEXT):
        f = fd / f"CONTEXT-{kind}.parquet"
        if f.exists():
            frm, to = _span(f)
            out.append({"Quelle": SOURCE_TEXT[kind], "Coins": "alle", "Zeilen": _rows(f), "von": frm, "bis": to})
    for name, text in (("history_trades.parquet", "Backtest-Trades der Regel-Bots (Historie)"),
                       ("trades.parquet", "KI-Gedächtnis (Backtest + Paper-Trades)")):
        f = ud / "ai_memory" / name
        if f.exists():
            out.append({"Quelle": text, "Coins": "", "Zeilen": _rows(f), "von": "", "bis": ""})
    return out


def own_trades(user_data: Path | None = None, now: pd.Timestamp | None = None) -> list[dict[str, Any]]:
    """Lerndaten durch eigene Trades: abgeschlossene Trades im gemeinsamen Gedächtnis je KI."""
    now = now or pd.Timestamp.now(tz="UTC")
    mem = shared_memory.load(user_data)
    rows = []
    for ki, name in KI_NAMES.items():
        part = mem[mem["ki"] == ki] if not mem.empty else mem
        closed = pd.to_datetime(part["close_date"], utc=True, format="ISO8601") if not part.empty else pd.Series(dtype="datetime64[ns, UTC]")
        rows.append({
            "KI": name, "Trades gesamt": len(part),
            "Lern-Trades": int((part["kind"] == "learn").sum()) if not part.empty else 0,
            "Signal-Trades": int((part["kind"] == "signal").sum()) if not part.empty else 0,
            "letzte 24 h": int((closed >= now - pd.Timedelta(hours=24)).sum()),
            "heute": int((closed >= now.normalize()).sum()),
            "mit Marktdaten": int(part[market_flow.ALL_FEATURES].notna().any(axis=1).sum()) if not part.empty else 0,
        })
    return rows


def usage(user_data: Path | None = None) -> list[dict[str, Any]]:
    """Was jede KI pro Entscheidung/Training tatsächlich nutzt (Schätzung aus der Konfiguration, gekennzeichnet)."""
    project = load_project_config()
    pairs = len(project.get("learning_pairs") or project["allowed_pairs"])
    from kitb.config import load_json
    from kitb.ml import FREQAI_SHARED

    shared = load_json(FREQAI_SHARED)["freqai"]
    spec = project["bots"]["ki_trader"]
    fp = {**shared["feature_parameters"], **spec["feature_parameters"]}
    per_tf = 6 * len(fp["indicator_periods_candles"]) + 2  # expand_all × Perioden + expand_basic
    n_feats = (per_tf * len(fp["include_timeframes"]) * (1 + len(fp["include_corr_pairlist"]))
               * (1 + fp.get("include_shifted_candles", 0))) + 2 + 12 + 2 * 19 + len(market_flow.ALL_FEATURES)  # Zeit, Regel-Bots, 19 Methoden
    candles_per_day = {"15m": 96, "1h": 24}.get(spec.get("timeframe", "1h"), 24)
    rows_per_pair = shared["train_period_days"] * candles_per_day
    mem = shared_memory.load(user_data)
    return [
        {"KI": "KI 1 (Muster)", "Merkmale je Kerze (ca.)": n_feats, "Trainingszeilen je Coin": rows_per_pair,
         "Coins": pairs, "Datenpunkte je Training (ca.)": n_feats * rows_per_pair * pairs,
         "davon Internet-Zusatzdaten": len(market_flow.ALL_FEATURES)},
        {"KI": "KI 2 (Belohnung)", "Merkmale je Kerze (ca.)": len(shared_memory.SNAPSHOT_FEATURES),
         "Trainingszeilen je Coin": "", "Coins": pairs,
         "Datenpunkte je Training (ca.)": len(mem) * len(shared_memory.SNAPSHOT_FEATURES),
         "davon Internet-Zusatzdaten": len(market_flow.ALL_FEATURES)},
    ]


def overview(user_data: Path | None = None) -> dict[str, Any]:
    imp = imported(user_data)
    own = own_trades(user_data)
    return {"updated_at": pd.Timestamp.now(tz="UTC").isoformat(), "eingefuehrt": imp,
            "eingefuehrt_zeilen": int(sum(r["Zeilen"] for r in imp)), "eigene_trades": own,
            "eigene_trades_gesamt": int(sum(r["Trades gesamt"] for r in own)), "nutzung": usage(user_data),
            "hinweis": "Merkmale und Datenpunkte je Training sind Schätzungen aus der Konfiguration."}


def write(user_data: Path | None = None) -> Path:
    p = market_flow.flow_dir(user_data) / "overview.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(overview(user_data), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, p)
    return p


def read(user_data: Path | None = None) -> dict[str, Any] | None:
    p = market_flow.flow_dir(user_data) / "overview.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
