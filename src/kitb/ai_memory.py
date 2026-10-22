"""KI-Gedächtnis und KI-Filter (Freigabe Burak, 22.10.2026 – nur Paper-Trading).

Ablauf (einmal täglich, ``kitb memory daily``):
1. **Gedächtnis füllen:** Alle abgeschlossenen Trades aller Paper-Bots (aktuelle Runde + archivierte Runden) und – einmalig –
   die historischen Backtest-Trades der Datensammler-Strategien werden mit der Marktlage beim Einstieg gespeichert
   (``user_data/ai_memory/trades.parquet``).
2. **Lernen:** Ein Klassifikator lernt, welche Trades sich gelohnt haben (Strategie, Markt, Marktlage -> Gewinn ja/nein).
3. **Prüfen:** Gemessen wird immer auf den jüngsten 20 % der Trades, die das Modell beim Prüftraining nicht gesehen hat.
4. **Anwenden:** Bots mit ``kitb_ai_filter`` fragen das jüngste Modell vor jedem Einstieg; bei geringer
   Gewinnwahrscheinlichkeit lassen sie den Trade aus. Die festen Datensammler bleiben ungefiltert (Kontrollgruppe).

Die Marktlage wird nur aus abgeschlossenen Tageskerzen **vor** dem Einstieg berechnet (kein Blick in die Zukunft).
"""

import json
import logging
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from kitb import paths
from kitb.config import load_project_config

log = logging.getLogger(__name__)

FEATURES = ["dist_sma200", "dist_sma50", "ret_7d", "ret_30d", "vol_30d", "rsi_14d"]
FEATURE_TEXT = {
    "dist_sma200": "Abstand zur 200-Tage-Linie", "dist_sma50": "Abstand zur 50-Tage-Linie",
    "ret_7d": "Kursänderung der letzten 7 Tage", "ret_30d": "Kursänderung der letzten 30 Tage",
    "vol_30d": "Schwankungsstärke der letzten 30 Tage", "rsi_14d": "RSI (überkauft/überverkauft)",
    "pair": "Markt (BTC oder ETH)", "strategy": "Art der Strategie",
}
MIN_TRADES_TO_TRAIN = 50
HOLDOUT_SHARE = 0.2
# Schwelle = Gewinnquote im Training: erlaubt werden Trades mit überdurchschnittlicher Gewinnchance. Eine feste 0,5 würde
# bei Strategien mit wenigen, aber großen Gewinnen (Trefferquote ~25 %) jeden Trade ablehnen.


def memory_dir(user_data: Path | None = None) -> Path:
    return (user_data or paths.USER_DATA) / "ai_memory"


# --- Marktlage --------------------------------------------------------------------------------------------------------

def daily_market_table(daily: pd.DataFrame) -> pd.DataFrame:
    """Merkmale je Tag; Zeile für Tag D nutzt nur Kerzen bis einschließlich D-1 (beim Einstieg an Tag D bekannt)."""
    d = daily[["date", "close"]].copy().sort_values("date")
    d["date"] = pd.to_datetime(d["date"], utc=True).dt.normalize()
    c = d["close"]
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    feats = pd.DataFrame({
        "date": d["date"],
        "dist_sma200": c / c.rolling(200).mean() - 1, "dist_sma50": c / c.rolling(50).mean() - 1,
        "ret_7d": c / c.shift(7) - 1, "ret_30d": c / c.shift(30) - 1,
        "vol_30d": np.log(c).diff().rolling(30).std(), "rsi_14d": 100 - 100 / (1 + gain / loss.replace(0, np.nan)),
    })
    feats[FEATURES] = feats[FEATURES].shift(1)  # nur Vergangenheit
    return feats.set_index("date")


def features_at(table: pd.DataFrame, when: Any) -> dict[str, float] | None:
    day = pd.Timestamp(when, tz="UTC").normalize() if pd.Timestamp(when).tzinfo is None else pd.Timestamp(when).tz_convert("UTC").normalize()
    if day not in table.index:
        sub = table.loc[:day]
        if sub.empty:
            return None
        row = sub.iloc[-1]
    else:
        row = table.loc[day]
    if row[FEATURES].isna().any():
        return None
    return {k: float(row[k]) for k in FEATURES}


def fetch_public(exchange_id: str, pair: str, limit: int = 1000) -> pd.DataFrame:
    """Öffentliche Tageskerzen (ohne Konto/Keys) – dieselbe Quelle wie die Übersicht."""
    from kitb.dashboard.data import fetch_daily_public

    return fetch_daily_public(exchange_id, pair, limit)


def load_daily(pair: str, user_data: Path | None = None, fetch: Callable | None = None,
               exchange_id: str = "binance") -> pd.DataFrame:
    """Lokale Tageskerzen + (falls möglich) aktuelle öffentliche Kerzen der Börse."""
    from kitb.data.storage import load_ohlcv_raw

    frames = []
    local = (user_data or paths.USER_DATA) / "data" / exchange_id
    if local.exists():
        try:
            frames.append(load_ohlcv_raw(local, pair, "1d")[["date", "close"]])
        except Exception:  # noqa: BLE001 – fehlende Datei
            pass
    if fetch is not None:
        try:
            frames.append(fetch(exchange_id, pair, 1000)[["date", "close"]])
        except Exception as e:  # noqa: BLE001 – offline: mit lokalen Daten weiter
            log.warning("Tageskerzen %s nicht abrufbar: %s", pair, e)
    if not frames:
        return pd.DataFrame(columns=["date", "close"])
    df = pd.concat(frames)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


# --- Gedächtnis -------------------------------------------------------------------------------------------------------

def _paper_dbs(user_data: Path | None = None) -> list[tuple[str, str, Path]]:
    """(paper_id, Runde, DB) für die aktuelle Runde und alle archivierten Runden aller Paper-Bots."""
    from kitb import paper

    out = []
    for bot in paper.paper_ids():
        d = paper.paper_dir(bot, user_data)
        if (d / "tradesv3.dryrun.sqlite").exists():
            out.append((bot, "aktuell", d / "tradesv3.dryrun.sqlite"))
        for r in sorted(paper.rounds_dir(bot, user_data).glob("round-*.sqlite")):
            out.append((bot, r.stem, r))
    return out


def paper_trades_all(user_data: Path | None = None) -> pd.DataFrame:
    project = load_project_config()
    specs = {**project["bots"], **project.get("paper_strategies", {})}
    rows = []
    for bot, rnd, db in _paper_dbs(user_data):
        try:
            with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as con:
                t = pd.read_sql_query("SELECT pair, open_date, close_date, close_profit, close_profit_abs, "
                                      "stake_amount, exit_reason FROM trades WHERE is_open = 0", con)
        except Exception:  # noqa: BLE001 – leere/kaputte DB überspringen
            continue
        t["bot"], t["round"], t["source"] = bot, rnd, "paper"
        t["strategy"] = specs.get(bot, {}).get("strategy", bot)
        rows.append(t)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def seed_from_backtests(*, datadir: Path, data_source: str, user_data: Path | None = None,
                        offline: bool = False) -> int:
    """Einmalig: historische Trades der Datensammler-Strategien als Startwissen ins Gedächtnis legen."""
    from kitb.backtest import run_freqtrade_backtest

    import tempfile

    from kitb.config import load_freqtrade_config

    project = load_project_config()
    fleet = {v["strategy"] for v in project.get("paper_strategies", {}).values() if v.get("mode") == "explore"}
    # Breit lernen: alle Lernmärkte, für die lokale Daten vorliegen (kitb download --pairs learning).
    pairs = [p for p in project.get("learning_pairs", project["allowed_pairs"])
             if all((datadir / f"{p.replace('/', '_')}-{tf}.parquet").exists() for tf in ("1h", "1d"))
             ] or project["allowed_pairs"]
    ft = load_freqtrade_config()
    ft["exchange"] = {**ft["exchange"], "pair_whitelist": pairs}
    tmp_cfg = Path(tempfile.mkdtemp()) / "seed-config.json"
    tmp_cfg.write_text(json.dumps(ft), encoding="utf-8")
    daily = {s for s in fleet if s in project["evaluation"].get("daily_strategies", [])}
    rows = []
    for tf, strategies in (("1h", sorted(fleet - daily)), ("1d", sorted(daily))):
        if not strategies:
            continue
        try:
            res = run_freqtrade_backtest(strategies=strategies, timeframe=tf, timerange=None, datadir=datadir,
                                         data_source=data_source, offline=offline, registry=None,
                                         results_dir=None, config_path=tmp_cfg)
        except Exception as e:  # noqa: BLE001 – z. B. zu wenig Tageskerzen: übrige Zeitintervalle trotzdem nutzen
            log.warning("Startwissen %s nicht erzeugt: %s", tf, e)
            continue
        for s, r in res["strategies"].items():
            for t in r["trades"]:
                rows.append({"pair": t["pair"], "open_date": t["open_date"], "close_date": t["close_date"],
                             "close_profit": t["profit_ratio"], "close_profit_abs": t["profit_abs"],
                             "stake_amount": t["stake_amount"], "exit_reason": t["exit_reason"],
                             "bot": "backtest", "round": "historisch", "source": "backtest", "strategy": s})
    df = pd.DataFrame(rows)
    out = memory_dir(user_data)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "history_trades.parquet", index=False)
    return len(df)


def build_memory(user_data: Path | None = None, fetch: Callable | None = None) -> pd.DataFrame:
    """Gedächtnis neu aufbauen: historische + Paper-Trades mit Marktlage beim Einstieg und Ergebnis."""
    parts = []
    hist = memory_dir(user_data) / "history_trades.parquet"
    if hist.exists():
        parts.append(pd.read_parquet(hist))
    paper_df = paper_trades_all(user_data)
    if not paper_df.empty:
        parts.append(paper_df)
    if not parts:
        return pd.DataFrame()
    trades = pd.concat(parts, ignore_index=True)
    trades["open_date"] = pd.to_datetime(trades["open_date"], utc=True, format="mixed")
    trades["close_date"] = pd.to_datetime(trades["close_date"], utc=True, format="mixed")
    tables = {p: daily_market_table(load_daily(p, user_data, fetch)) for p in trades["pair"].unique()}
    feats = [features_at(tables[r.pair], r.open_date) for r in trades.itertuples()]
    keep = [f is not None for f in feats]
    trades = trades[keep].reset_index(drop=True)
    trades[FEATURES] = pd.DataFrame([f for f in feats if f is not None])
    trades["win"] = (trades["close_profit"].astype(float) > 0).astype(int)
    trades = trades.sort_values("close_date").reset_index(drop=True)
    out = memory_dir(user_data)
    out.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(out / "trades.parquet", index=False)
    return trades


# --- Lernen und Prüfen ------------------------------------------------------------------------------------------------

def _design(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    x = pd.get_dummies(df[FEATURES + ["strategy", "pair"]], columns=["strategy", "pair"], dtype=float)
    if columns is not None:
        x = x.reindex(columns=columns, fill_value=0.0)
    return x


def _fit(x: pd.DataFrame, y: pd.Series):
    from sklearn.ensemble import HistGradientBoostingClassifier

    # Bewusst vorsichtig (flache Bäume, große Blätter, wenige Runden): Marktdaten sind verrauscht, sonst lernt
    # das Modell Zufallsmuster auswendig.
    return HistGradientBoostingClassifier(max_depth=2, max_iter=60, learning_rate=0.05, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=42).fit(x, y)


def evaluate_and_train(memory: pd.DataFrame, user_data: Path | None = None, registry=None) -> dict[str, Any]:
    """Prüftraining auf den älteren 80 %, Messung auf den jüngsten 20 %; danach Modell auf allem für den Einsatz."""
    import joblib
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score

    n = len(memory)
    if n < MIN_TRADES_TO_TRAIN:
        return {"trained": False, "reason": f"zu wenig Trades im Gedächtnis ({n} < {MIN_TRADES_TO_TRAIN})"}
    cut = int(n * (1 - HOLDOUT_SHARE))
    train, hold = memory.iloc[:cut], memory.iloc[cut:]
    x_tr = _design(train)
    x_ho = _design(hold, list(x_tr.columns))
    m = _fit(x_tr, train["win"])
    p = m.predict_proba(x_ho)[:, 1]
    base_rate = float(train["win"].mean())
    allow = p >= base_rate
    acc = float(((p >= 0.5).astype(int) == hold["win"]).mean())
    naive_acc = float((hold["win"] == int(base_rate >= 0.5)).mean())
    try:
        auc = float(roc_auc_score(hold["win"], p))
    except ValueError:
        auc = None
    avg_all = float(hold["close_profit"].astype(float).mean())
    avg_kept = float(hold.loc[allow, "close_profit"].astype(float).mean()) if allow.any() else None
    imp = permutation_importance(m, x_ho, hold["win"], n_repeats=5, random_state=42)
    raw = dict(zip(x_ho.columns, imp.importances_mean))
    grouped: dict[str, float] = {}
    for col, v in raw.items():
        key = col.split("_", 1)[0] if col.startswith(("strategy_", "pair_")) else col
        grouped[key] = grouped.get(key, 0.0) + float(v)
    metrics = {"accuracy": round(acc, 4), "naive_accuracy": round(naive_acc, 4),
               "auc": None if auc is None else round(auc, 4), "win_rate_holdout": round(float(hold["win"].mean()), 4),
               "avg_profit_all": round(avg_all, 5), "avg_profit_kept": None if avg_kept is None else round(avg_kept, 5),
               "kept_share": round(float(allow.mean()), 4), "threshold": round(base_rate, 4),
               "holdout_from": str(hold["close_date"].iloc[0]), "holdout_to": str(hold["close_date"].iloc[-1]),
               "paper_trades": int((memory["source"] == "paper").sum())}
    final = _fit(_design(memory), memory["win"])
    out = memory_dir(user_data)
    out.mkdir(parents=True, exist_ok=True)
    model_file = out / f"filter-{datetime.now(UTC):%Y%m%d-%H%M%S}.joblib"
    joblib.dump({"model": final, "columns": list(_design(memory).columns),
                 "threshold": float(memory["win"].mean())}, model_file)
    (out / "filter-latest.json").write_text(json.dumps({"model_file": model_file.name}), encoding="utf-8")
    importances = dict(sorted(grouped.items(), key=lambda kv: -kv[1]))
    if registry is not None:
        registry.insert("ai_filter_versions", {"n_trades": n, "n_holdout": len(hold), "metrics": json.dumps(metrics),
                                               "importances": json.dumps(importances), "model_file": model_file.name})
    return {"trained": True, "n_trades": n, "metrics": metrics, "importances": importances,
            "model_file": str(model_file)}


# --- Anwenden (im Paper-Bot) ------------------------------------------------------------------------------------------

_cache: dict[str, Any] = {}


def allow_entry(*, strategy: str, pair: str, when: Any, user_data: Path | None = None,
                fetch: Callable | None = fetch_public) -> tuple[bool, float | None]:
    """True, wenn der jüngste KI-Filter den Trade erlaubt. Ohne Modell oder ohne Marktlage: erlauben (protokolliert)."""
    import joblib

    latest = memory_dir(user_data) / "filter-latest.json"
    if not latest.exists():
        return True, None
    name = json.loads(latest.read_text(encoding="utf-8"))["model_file"]
    if _cache.get("name") != name:
        _cache.update(name=name, bundle=joblib.load(memory_dir(user_data) / name), tables={})
    day = pd.Timestamp(when).tz_convert("UTC").normalize() if pd.Timestamp(when).tzinfo else pd.Timestamp(when, tz="UTC").normalize()
    key = (pair, str(day))
    if key not in _cache["tables"]:
        _cache["tables"][key] = daily_market_table(load_daily(pair, user_data, fetch))
    f = features_at(_cache["tables"][key], when)
    if f is None:
        return True, None
    row = pd.DataFrame([{**f, "strategy": strategy, "pair": pair}])
    x = _design(row, _cache["bundle"]["columns"])
    p = float(_cache["bundle"]["model"].predict_proba(x)[:, 1][0])
    return p >= _cache["bundle"].get("threshold", 0.5), p


# --- Auswertung für die Seite „Was hat die KI gelernt?“ ---------------------------------------------------------------

def learning_progress(registry) -> dict[str, Any]:
    rows = registry.query("SELECT * FROM ai_filter_versions ORDER BY id")
    if not rows:
        return {"versions": pd.DataFrame(), "first": None, "last": None, "learned_pct": None, "verdict": "noch kein Training"}
    v = pd.DataFrame([{"Version": r["id"], "Datum": r["created_at"][:10], "Trades im Gedächtnis": r["n_trades"],
                       **json.loads(r["metrics"])} for r in rows])
    first, last = v.iloc[0], v.iloc[-1]
    # Trennschärfe (AUC): 50 % = Raten, 100 % = perfekt. „Dazugelernt“ = Veränderung seit Version 1 in %-Punkten.
    auc_first = first["auc"] if first["auc"] is not None else 0.5
    auc_last = last["auc"] if last["auc"] is not None else 0.5
    edge_first, edge_last = auc_first - 0.5, auc_last - 0.5
    learned = round((auc_last - auc_first) * 100, 1)
    filt_gain = (None if last["avg_profit_kept"] is None
                 else round((last["avg_profit_kept"] - last["avg_profit_all"]) * 100, 2))
    auc = last.get("auc")
    # Urteil über Trennschärfe (AUC, 0,5 = Raten) und den tatsächlichen Gewinnvorteil der erlaubten Trades.
    if auc is not None and auc >= 0.55 and (filt_gain or 0) > 0:
        verdict = "🟢 besser als Raten – der Filter trennt gute von schlechten Trades"
    elif (auc is not None and auc < 0.5) or (filt_gain is not None and filt_gain < 0):
        verdict = "🔴 schlechter als Raten – der Filter hilft derzeit nicht"
    else:
        verdict = "🟡 noch unklar – kein messbarer Vorteil"
    return {"versions": v, "first": first, "last": last, "learned_pct": learned,
            "edge_first": round(edge_first * 100, 1), "edge_last": round(edge_last * 100, 1),
            "filter_gain_pct": filt_gain, "verdict": verdict,
            "importances": json.loads(rows[-1]["importances"])}


def daily(*, user_data: Path | None = None, registry=None, fetch: Callable | None = None) -> dict[str, Any]:
    mem = build_memory(user_data, fetch)
    res = evaluate_and_train(mem, user_data, registry)
    if registry is not None:
        registry.log_source_event("ai_memory", "daily", "ok" if res.get("trained") else "skipped",
                                  {"n_trades": len(mem), **{k: v for k, v in res.items() if k != "importances"}})
    return res
