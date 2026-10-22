"""Datenzugriff fürs Dashboard (ohne Streamlit, dadurch testbar). Nur lesend – außer Not-Stopp."""

import json
import re
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


# --- Einfache Übersicht (Reiter „Übersicht“) -------------------------------------------------------------------------

def fetch_daily_public(exchange_id: str, pair: str, limit: int = 210) -> pd.DataFrame:
    """Öffentliche Tageskerzen der Börse (ohne Konto/Keys). Die letzte Kerze ist der laufende, unfertige Tag."""
    import ccxt

    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": 15000})
    rows = ex.fetch_ohlcv(pair, "1d", limit=limit)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def trend_view(daily: pd.DataFrame, period: int = 200) -> dict[str, Any]:
    """Lage zur 200-Tage-Linie wie in DailyTrendSma: entschieden wird nur mit abgeschlossenen Tageskerzen."""
    done = daily.iloc[:-1]
    if len(done) < period:
        return {"price": float(daily["close"].iloc[-1]) if len(daily) else None, "sma": None,
                "above_at_last_close": None, "above_now": None}
    sma = float(done["close"].tail(period).mean())
    price = float(daily["close"].iloc[-1])
    return {"price": price, "sma": round(sma, 2), "above_at_last_close": bool(done["close"].iloc[-1] > sma),
            "above_now": bool(price > sma), "distance": round(price / sma - 1, 4)}


def ever_started(bot: str, user_data: Path | None = None) -> bool:
    d = (user_data or paths.USER_DATA) / "paper" / bot
    return (d / "freqtrade.log").exists() or (d / "tradesv3.dryrun.sqlite").exists()


def next_daily_close(now: pd.Timestamp | None = None) -> pd.Timestamp:
    now = now or pd.Timestamp.now(tz="UTC")
    return now.normalize() + pd.Timedelta(days=1)


def overview(bot: str, starting_balance: float, user_data: Path | None = None,
             fetch=fetch_daily_public, exchange_id: str = "binance") -> dict[str, Any]:
    """Alles für die einfache Übersicht eines Paper-Bots: läuft er, was hält er, was hat er zuletzt getan."""
    from kitb import paper

    log_age = paper._log_age_seconds(bot, user_data)
    trades = paper_trades(bot, user_data)
    pairs = bot_info(bot)["pairs"]
    markets, dailies, fetch_error = {}, {}, None
    for pair in pairs:
        try:
            dailies[pair] = fetch(exchange_id, pair)
            markets[pair] = trend_view(dailies[pair])
        except Exception as e:  # kein Internet o. Ä. -> Übersicht trotzdem anzeigen
            markets[pair] = {"price": None, "sma": None, "above_at_last_close": None, "above_now": None}
            fetch_error = f"{type(e).__name__}: {e}"[:200]

    positions, unrealized = [], 0.0
    open_t = trades[trades["is_open"] == 1] if not trades.empty else trades
    for _, t in open_t.iterrows():
        price = markets.get(t["pair"], {}).get("price")
        value = float(t["amount"]) * price if price else None
        pnl = value - float(t["stake_amount"]) if value is not None else None
        unrealized += pnl or 0.0
        positions.append({"Markt": t["pair"], "gekauft am": str(t["open_date"])[:16], "Kaufkurs": float(t["open_rate"]),
                          "aktueller Kurs": price, "Einsatz (USDT)": round(float(t["stake_amount"]), 2),
                          "Gewinn/Verlust (USDT)": None if pnl is None else round(pnl, 2),
                          "Gewinn/Verlust (%)": None if pnl is None else round(pnl / float(t["stake_amount"]), 4)})
    closed = trades[trades["is_open"] == 0] if not trades.empty else trades
    realized = float(closed["close_profit_abs"].fillna(0).sum()) if not closed.empty else 0.0

    actions = []
    for _, t in trades.iterrows():
        actions.append({"Zeit": str(t["open_date"])[:16], "Aktion": "Kauf (simuliert)", "Markt": t["pair"],
                        "Kurs": float(t["open_rate"]), "Ergebnis (USDT)": None})
        if t["is_open"] == 0 and t["close_date"] is not None:
            actions.append({"Zeit": str(t["close_date"])[:16], "Aktion": "Verkauf (simuliert)", "Markt": t["pair"],
                            "Kurs": float(t["close_rate"]), "Ergebnis (USDT)": round(float(t["close_profit_abs"] or 0), 2)})
    actions.sort(key=lambda a: a["Zeit"], reverse=True)

    return {"bot": bot, "running": log_age is not None and log_age <= paper.HEARTBEAT_MAX_AGE_S,
            "log_age_seconds": None if log_age is None else round(log_age), "kill_switch": killswitch.info(user_data),
            "start_balance": starting_balance, "realized_pnl": round(realized, 2),
            "unrealized_pnl": round(unrealized, 2), "equity": round(starting_balance + realized + unrealized, 2),
            "positions": positions, "markets": markets, "actions": actions[:20],
            "next_decision": next_daily_close(), "fetch_error": fetch_error,
            "trades_table": trades_table(trades, markets),
            "performance": performance(trades, dailies, starting_balance)}


def explain_market(pair: str, m: dict[str, Any], holding: bool) -> str:
    """Ein Satz in Alltagssprache, was DailyTrendSma bei diesem Markt vorhat."""
    if m.get("sma") is None:
        return f"{pair}: aktuelle Kurse nicht abrufbar – keine Einschätzung."
    lage = "über" if m["above_now"] else "unter"
    base = f"{pair} steht {abs(m['distance']):.1%} {lage} der 200-Tage-Linie ({m['sma']:,.0f})."
    if holding and m["above_now"]:
        return base + " Der Bot bleibt investiert."
    if holding:
        return base + " Schließt der Tag so, verkauft der Bot nach Tagesschluss."
    if m["above_now"]:
        return base + " Schließt der Tag so, kauft der Bot nach Tagesschluss."
    return base + " Der Bot wartet an der Seitenlinie."


def trades_table(trades: pd.DataFrame, markets: dict[str, Any]) -> pd.DataFrame:
    """Jeder Trade eine Zeile, in Alltagssprache; offene Trades mit aktuellem (unrealisiertem) Ergebnis."""
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for _, t in trades.iterrows():
        is_open = t["is_open"] == 1
        price = markets.get(t["pair"], {}).get("price") if is_open else t["close_rate"]
        if is_open:
            pnl = float(t["amount"]) * price - float(t["stake_amount"]) if price else None
        else:
            pnl = float(t["close_profit_abs"] or 0)
        rows.append({"Markt": t["pair"], "Status": "offen" if is_open else "abgeschlossen",
                     "Kauf": str(t["open_date"])[:10], "Kaufkurs": float(t["open_rate"]),
                     "Verkauf": None if is_open else str(t["close_date"])[:10],
                     "Verkaufs-/akt. Kurs": None if price is None else float(price),
                     "Einsatz (USDT)": round(float(t["stake_amount"]), 2),
                     "Ergebnis (USDT)": None if pnl is None else round(pnl, 2),
                     "Ergebnis (%)": None if pnl is None else round(pnl / float(t["stake_amount"]), 4),
                     "Grund": {"exit_signal": "unter 200-Tage-Linie", "stop_loss": "Stoploss −10 %",
                               "force_exit": "manuell"}.get(t["exit_reason"], t["exit_reason"]) if not is_open else ""})
    return pd.DataFrame(rows[::-1])


MIN_TRADES_FOR_VERDICT = 30


def performance(trades: pd.DataFrame, dailies: dict[str, pd.DataFrame], starting_balance: float) -> dict[str, Any]:
    """Tagesverlauf seit Start: Bot-Kapital vs. „einfach halten“ (Buy&Hold BTC/ETH je zur Hälfte).

    Bewertung bewusst vorsichtig: Unter MIN_TRADES_FOR_VERDICT abgeschlossenen Trades gibt es kein Urteil.
    """
    if trades.empty or not dailies:
        return {"curve": pd.DataFrame(), "bot_return": None, "hold_return": None, "verdict": "noch keine Trades",
                "closed": 0, "winrate": None}
    start = pd.to_datetime(trades["open_date"].min(), utc=True)
    days = pd.date_range(start.normalize(), pd.Timestamp.now(tz="UTC").normalize(), freq="D", tz="UTC")
    prices = {pair: df.set_index(df["date"].dt.normalize())["close"] for pair, df in dailies.items()}
    ref = {}
    for pair, px in prices.items():
        first = trades[trades["pair"] == pair]
        ref[pair] = float(first["open_rate"].iloc[0]) if not first.empty else float(px.asof(start.normalize()))
    opened = pd.to_datetime(trades["open_date"], utc=True)
    closed_at = pd.to_datetime(trades["close_date"], utc=True)
    rows = []
    for d in days:
        end = d + pd.Timedelta(days=1)
        realized = float(trades.loc[(trades["is_open"] == 0) & (closed_at < end), "close_profit_abs"].fillna(0).sum())
        unreal = 0.0
        held = trades[(opened < end) & ((trades["is_open"] == 1) | (closed_at >= end))]
        for _, t in held.iterrows():
            px = prices.get(t["pair"])
            if px is not None and len(px.loc[:d]):
                unreal += float(t["amount"]) * float(px.loc[:d].iloc[-1]) - float(t["stake_amount"])
        hold = [float(px.loc[:d].iloc[-1]) / ref[p] - 1 for p, px in prices.items() if len(px.loc[:d])]
        rows.append({"Tag": d.date(), "Bot": starting_balance + realized + unreal,
                     "Einfach halten": starting_balance * (1 + (sum(hold) / len(hold) if hold else 0.0))})
    curve = pd.DataFrame(rows).set_index("Tag")
    bot_ret = curve["Bot"].iloc[-1] / starting_balance - 1
    hold_ret = curve["Einfach halten"].iloc[-1] / starting_balance - 1
    closed = trades[trades["is_open"] == 0]
    n = len(closed)
    if n < MIN_TRADES_FOR_VERDICT:
        verdict = f"zu früh für ein Urteil ({n} von mindestens {MIN_TRADES_FOR_VERDICT} abgeschlossenen Trades)"
    elif bot_ret > hold_ret:
        verdict = "besser als einfach halten"
    else:
        verdict = "schlechter als einfach halten"
    return {"curve": curve, "bot_return": round(bot_ret, 4), "hold_return": round(hold_ret, 4),
            "bot_drawdown": round(float((1 - curve["Bot"] / curve["Bot"].cummax()).max()), 4),
            "hold_drawdown": round(float((1 - curve["Einfach halten"] / curve["Einfach halten"].cummax()).max()), 4),
            "verdict": verdict, "closed": n,
            "winrate": round(float((closed["close_profit_abs"] > 0).mean()), 3) if n else None,
            "since": start}


def fleet_table(starting_balance: float, user_data: Path | None = None) -> pd.DataFrame:
    """Alle Paper-Bots auf einen Blick: Art, läuft, Runde, Trades, Ergebnis der aktuellen Runde und aller Runden."""
    from kitb import paper

    project = load_project_config()
    rows = []
    for bot in paper.paper_ids():
        if not ever_started(bot, user_data):
            continue
        spec = project.get("paper_strategies", {}).get(bot) or project["bots"].get(bot, {})
        mode = spec.get("mode") or "learning"
        kind = {"candidate": "Kandidat (feste Regel)", "explore": "Datensammler", "learning": "KI-Bot (lernt täglich)"}[mode]
        if spec.get("ai_filter"):
            kind += " + KI-Filter"
        cur = paper_trades(bot, user_data)
        closed = cur[cur["is_open"] == 0] if not cur.empty else cur
        pnl = float(closed["close_profit_abs"].fillna(0).sum()) if not closed.empty else 0.0
        rounds = sorted(paper.rounds_dir(bot, user_data).glob("round-*.sqlite"))
        past = 0.0
        for r in rounds:
            t = _query(r, "SELECT close_profit_abs FROM trades WHERE is_open = 0")
            past += float(t["close_profit_abs"].fillna(0).sum()) if not t.empty else 0.0
        age = paper._log_age_seconds(bot, user_data)
        rows.append({"Bot": bot, "Art": kind, "Strategie": spec.get("strategy", ""),
                     "läuft": "🟢" if age is not None and age <= paper.HEARTBEAT_MAX_AGE_S else "🔴",
                     "Runde": len(rounds) + 1, "Trades (Runde)": len(cur),
                     "offen": int((cur["is_open"] == 1).sum()) if not cur.empty else 0,
                     "Ergebnis Runde (%)": round(pnl / starting_balance, 4),
                     "Ergebnis alle Runden (USDT)": round(pnl + past, 2)})
    return pd.DataFrame(rows)


def bot_info(bot: str) -> dict[str, Any]:
    """Art, Strategie, Beschreibung und Märkte eines Paper-Bots (Sammler/KI-Bots lernen auf mehr Märkten)."""
    project = load_project_config()
    spec = project.get("paper_strategies", {}).get(bot) or project["bots"].get(bot, {})
    mode = spec.get("mode") or "learning"
    pairs = project.get("learning_pairs") if mode in ("explore", "learning") else None
    kind = {"candidate": "Kandidat (feste Regel)", "explore": "Datensammler",
            "learning": "KI-Bot (lernt täglich neu)"}[mode] + (" + KI-Filter" if spec.get("ai_filter") else "")
    return {"kind": kind, "strategy": spec.get("strategy", ""), "description": spec.get("description", ""),
            "pairs": list(pairs or project["allowed_pairs"])}


_LOG_KEYS = ("KI-Filter", "Risikomodul", "Verlustlimit", "Long signal", "entry", "Exit", "exit", "Done training",
             "Not-Stopp")


def recent_events(bot: str, n: int = 8, user_data: Path | None = None) -> list[str]:
    """Die letzten relevanten Zeilen aus dem Freqtrade-Log des Bots (Signale, Filter, Risiko, Training)."""
    log = (user_data or paths.USER_DATA) / "paper" / bot / "freqtrade.log"
    if not log.exists():
        return []
    with log.open("rb") as f:
        f.seek(max(0, log.stat().st_size - 200_000))
        lines = f.read().decode("utf-8", errors="replace").splitlines()
    hits = [ln for ln in lines if any(k in ln for k in _LOG_KEYS) and "heartbeat" not in ln]
    return [ln[:19] + "  " + ln.split(" - ", 3)[-1][:160] for ln in hits[-n:]][::-1]


_FILTER_RE = re.compile(r"KI-Filter (\S+): Gewinnchance (\S+) -> (\w+)")


def ai_filter_decisions(n: int = 30, user_data: Path | None = None) -> pd.DataFrame:
    """Jüngste Entscheidungen des KI-Filters aus den Logs aller Bots mit ai_filter (neueste zuerst)."""
    rows = []
    for bot, spec in load_project_config().get("paper_strategies", {}).items():
        if not spec.get("ai_filter"):
            continue
        log = (user_data or paths.USER_DATA) / "paper" / bot / "freqtrade.log"
        if not log.exists():
            continue
        with log.open("rb") as f:
            f.seek(max(0, log.stat().st_size - 500_000))
            lines = f.read().decode("utf-8", errors="replace").splitlines()
        for ln in lines:
            m = _FILTER_RE.search(ln)
            if m:
                rows.append({"Zeit (UTC)": ln[:19], "Bot": bot, "Markt": m.group(1), "Gewinnchance": m.group(2),
                             "Entscheidung": m.group(3)})
    df = pd.DataFrame(rows, columns=["Zeit (UTC)", "Bot", "Markt", "Gewinnchance", "Entscheidung"])
    return df.sort_values("Zeit (UTC)", ascending=False).head(n).reset_index(drop=True)
