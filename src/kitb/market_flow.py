"""Marktdaten „was alle Trader gerade tun“ als Lerndaten für beide KIs (Burak, 23.10.2026).

Quellen (öffentlich, kostenlos, ohne Konto/Keys, nur lesend):
- Binance Spot, Kerzen inkl. aggressiver Käufe (taker buy volume) -> Kauf-/Verkaufsdruck. Verlauf ab 2022 nachladbar.
- Binance USDT-Terminmarkt (nur als Information, gehandelt wird weiter nur Spot): Finanzierungsrate (Verlauf ab 2022),
  Long/Short-Verhältnis aller Konten und der Top-Trader, offene Positionen. Unsicher: Binance liefert die Verhältnisse
  nur für etwa die letzten 30 Tage, daher wird ab jetzt laufend mitgeschrieben.
- Binance Spot Orderbuch: Ungleichgewicht der Kauf-/Verkaufsaufträge nahe am Kurs (nur Aufzeichnung, kein Verlauf).
- Gesamtmarkt/Stimmung (für alle Coins gleich): Krypto-Fear-&-Greed-Index (alternative.me, ab 2018) und aus der
  Datenbank der US-Notenbank St. Louis (FRED, CSV ohne Key): VIX, Rendite 10-jähriger US-Anleihen, Dollar-Index, S&P 500,
  Nasdaq 100 (Krypto läuft seit etwa 2020 oft ähnlich wie Technologie-Aktien, Stärke schwankt – Unsicher).

Einziger Schreiber ist der Dienst ``market-flow`` (``kitb flow loop``). Die Strategien lesen nur. Damit kein Blick in die
Zukunft entsteht, bekommt eine Kerze nur Werte, die beim Schließen der **vorherigen** Kerze bekannt waren.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from kitb import paths

log = logging.getLogger(__name__)

SPOT = "https://api.binance.com"
FUTURES = "https://fapi.binance.com"
BACKFILL_FROM = "2022-01-01"
FLOW_FEATURES = ["flow_taker_buy", "flow_taker_buy_4h", "flow_funding", "flow_ls_accounts", "flow_ls_top",
                 "flow_oi_change_4h"]
FLOW_TEXT = {
    "flow_taker_buy": "Kaufdruck (Anteil aggressiver Käufe, letzte Kerze)",
    "flow_taker_buy_4h": "Kaufdruck der letzten 4 Stunden",
    "flow_funding": "Finanzierungsrate am Terminmarkt",
    "flow_ls_accounts": "Long/Short-Verhältnis aller Trader",
    "flow_ls_top": "Long/Short-Verhältnis der Top-Trader",
    "flow_oi_change_4h": "Veränderung der offenen Positionen (4 h)",
}
KINDS = ("taker", "funding", "ratio", "depth")
FNG_URL = "https://api.alternative.me/fng/"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_SERIES = {"VIXCLS": "vix", "DGS10": "us10y", "DTWEXBGS": "dollar", "SP500": "spx", "NASDAQ100": "ndx"}
CONTEXT_FEATURES = ["ctx_fear_greed", "ctx_fear_greed_chg_7d", "ctx_vix", "ctx_vix_chg_5d", "ctx_us10y_chg_20d",
                    "ctx_dollar_chg_20d", "ctx_spx_ret_5d", "ctx_spx_ret_20d", "ctx_ndx_ret_5d", "ctx_ndx_ret_20d"]
CONTEXT_TEXT = {
    "ctx_fear_greed": "Krypto-Stimmung (Fear & Greed)", "ctx_fear_greed_chg_7d": "Stimmungsänderung 7 Tage",
    "ctx_vix": "Angstbarometer der Aktienbörse (VIX)", "ctx_vix_chg_5d": "VIX-Änderung 5 Tage",
    "ctx_us10y_chg_20d": "Änderung US-Zinsen (10 Jahre) 20 Tage", "ctx_dollar_chg_20d": "Dollar-Stärke 20 Tage",
    "ctx_spx_ret_5d": "S&P 500 letzte 5 Tage", "ctx_spx_ret_20d": "S&P 500 letzte 20 Tage",
    "ctx_ndx_ret_5d": "Nasdaq 100 (Technologie) letzte 5 Tage", "ctx_ndx_ret_20d": "Nasdaq 100 letzte 20 Tage",
}
ALL_FEATURES = FLOW_FEATURES + CONTEXT_FEATURES


def flow_dir(user_data: Path | None = None) -> Path:
    return (user_data or paths.USER_DATA) / "market_flow"


def symbol(pair: str) -> str:
    return pair.split(":")[0].replace("/", "")


def _path(pair: str, kind: str, user_data: Path | None) -> Path:
    return flow_dir(user_data) / f"{symbol(pair)}-{kind}.parquet"


def load(pair: str, kind: str, user_data: Path | None = None) -> pd.DataFrame:
    p = _path(pair, kind, user_data)
    if not p.exists():
        return pd.DataFrame(columns=["date"])
    try:
        return pd.read_parquet(p)
    except Exception:  # noqa: BLE001 – gerade ersetzt/gesperrt: beim nächsten Mal erneut
        return pd.DataFrame(columns=["date"])


def _store(pair: str, kind: str, new: pd.DataFrame, user_data: Path | None) -> int:
    """Anhängen, doppelte Zeitpunkte entfernen, atomar ersetzen (Leser sehen nie eine halbe Datei)."""
    if new.empty:
        return 0
    old = load(pair, kind, user_data)
    df = (pd.concat([old, new]) if not old.empty else new).drop_duplicates("date", keep="last").sort_values("date")
    p = _path(pair, kind, user_data)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    df.reset_index(drop=True).to_parquet(tmp, index=False)
    for i in range(5):
        try:
            os.replace(tmp, p)
            break
        except PermissionError:  # Windows-Bind-Mount: Leser hält die Datei kurz offen
            time.sleep(0.5 * (i + 1))
    return len(df) - len(old)


def _get(url: str, params: dict[str, Any]) -> Any:
    import requests

    time.sleep(0.15)  # Binance-Abfragegrenzen schonen (Gewicht weit unter dem Limit)
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _get_text(url: str, params: dict[str, Any]) -> str:
    import requests

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.text


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def _start(pair: str, kind: str, user_data: Path | None, default: str) -> pd.Timestamp:
    old = load(pair, kind, user_data)
    if old.empty:
        return pd.Timestamp(default, tz="UTC")
    return pd.Timestamp(old["date"].max())


def update_taker(pair: str, user_data: Path | None = None, get: Callable = _get, until: pd.Timestamp | None = None,
                 max_requests: int = 200) -> int:
    """15-Minuten-Kerzen mit Volumen aggressiver Käufer (Spalte 9 der Binance-Kerzen) – nur abgeschlossene Kerzen."""
    until = until or pd.Timestamp.now(tz="UTC")
    start, added = _start(pair, "taker", user_data, BACKFILL_FROM), 0
    for _ in range(max_requests):
        rows = get(f"{SPOT}/api/v3/klines", {"symbol": symbol(pair), "interval": "15m", "startTime": _ms(start),
                                              "limit": 1000})
        if not rows:
            break
        df = pd.DataFrame({"date": pd.to_datetime([r[0] for r in rows], unit="ms", utc=True),
                           "close_time": pd.to_datetime([r[6] for r in rows], unit="ms", utc=True),
                           "volume": [float(r[5]) for r in rows], "taker_buy": [float(r[9]) for r in rows]})
        df = df[df["close_time"] < until].drop(columns="close_time")
        added += _store(pair, "taker", df, user_data)
        if len(rows) < 1000 or df.empty:
            break
        start = df["date"].max() + pd.Timedelta(minutes=15)
    return added


def update_funding(pair: str, user_data: Path | None = None, get: Callable = _get, max_requests: int = 20) -> int:
    start, added = _start(pair, "funding", user_data, BACKFILL_FROM), 0
    for _ in range(max_requests):
        rows = get(f"{FUTURES}/fapi/v1/fundingRate", {"symbol": symbol(pair), "startTime": _ms(start), "limit": 1000})
        if not rows:
            break
        df = pd.DataFrame({"date": pd.to_datetime([int(r["fundingTime"]) for r in rows], unit="ms", utc=True),
                           "funding_rate": [float(r["fundingRate"]) for r in rows]})
        added += _store(pair, "funding", df, user_data)
        if len(rows) < 1000:
            break
        start = df["date"].max() + pd.Timedelta(seconds=1)
    return added


def update_ratios(pair: str, user_data: Path | None = None, get: Callable = _get) -> int:
    """Long/Short-Verhältnisse und offene Positionen (15-Minuten-Raster, Binance liefert nur die letzten ~30 Tage)."""
    def frame(path: str, col: str, src: str) -> pd.DataFrame:
        rows = get(f"{FUTURES}{path}", {"symbol": symbol(pair), "period": "15m", "limit": 500})
        return pd.DataFrame({"date": pd.to_datetime([int(r["timestamp"]) for r in rows], unit="ms", utc=True),
                             col: [float(r[src]) for r in rows]}).set_index("date")

    parts = [frame("/futures/data/globalLongShortAccountRatio", "ls_accounts", "longShortRatio"),
             frame("/futures/data/topLongShortPositionRatio", "ls_top", "longShortRatio"),
             frame("/futures/data/openInterestHist", "oi_value", "sumOpenInterestValue")]
    df = pd.concat(parts, axis=1).reset_index()
    return _store(pair, "ratio", df, user_data)


def record_depth(pair: str, user_data: Path | None = None, get: Callable = _get, band: float = 0.01) -> int:
    """Orderbuch-Ungleichgewicht innerhalb ±1 % um die Mitte: +1 nur Kaufaufträge, -1 nur Verkaufsaufträge."""
    book = get(f"{SPOT}/api/v3/depth", {"symbol": symbol(pair), "limit": 500})
    bids = np.array(book["bids"], dtype=float).reshape(-1, 2)
    asks = np.array(book["asks"], dtype=float).reshape(-1, 2)
    if not len(bids) or not len(asks):
        return 0
    mid = (bids[0, 0] + asks[0, 0]) / 2
    b = float((bids[bids[:, 0] >= mid * (1 - band)].prod(axis=1)).sum())
    a = float((asks[asks[:, 0] <= mid * (1 + band)].prod(axis=1)).sum())
    now = pd.Timestamp.now(tz="UTC").floor("min")
    return _store(pair, "depth", pd.DataFrame({"date": [now], "depth_imbalance": [(b - a) / max(b + a, 1e-9)],
                                               "spread": [(asks[0, 0] - bids[0, 0]) / mid]}), user_data)


def update_all(pairs: list[str], user_data: Path | None = None, get: Callable = _get, slow: bool = True) -> dict:
    """Ein Durchlauf für alle Coins. ``slow``: auch Finanzierungsrate und Verhältnisse (alle 5 Minuten reicht)."""
    out: dict[str, Any] = {}
    for pair in pairs:
        res = {}
        jobs = [("taker", update_taker), ("depth", record_depth)]
        if slow:
            jobs += [("funding", update_funding), ("ratio", update_ratios)]
        for name, fn in jobs:
            try:
                res[name] = fn(pair, user_data, get=get)
            except Exception as e:  # noqa: BLE001 – ein fehlender Markt (z. B. kein Terminkontrakt) stoppt nichts
                res[name] = f"Fehler: {str(e)[:120]}"
        out[pair] = res
    return out


def update_context(user_data: Path | None = None, get: Callable = _get, get_text: Callable = _get_text) -> dict:
    """Tageswerte für alle Coins gemeinsam (komplett neu geladen – wenige tausend Zeilen)."""
    import io

    res: dict[str, Any] = {}
    try:
        rows = get(FNG_URL, {"limit": 0, "format": "json"})["data"]
        df = pd.DataFrame({"date": pd.to_datetime([int(r["timestamp"]) for r in rows], unit="s", utc=True),
                           "value": [float(r["value"]) for r in rows]})
        res["fear_greed"] = _store("CONTEXT", "fear_greed", df, user_data)
    except Exception as e:  # noqa: BLE001
        res["fear_greed"] = f"Fehler: {str(e)[:120]}"
    for sid, name in FRED_SERIES.items():
        try:
            raw = pd.read_csv(io.StringIO(get_text(FRED_URL, {"id": sid, "cosd": BACKFILL_FROM[:4] + "-01-01"})))
            raw.columns = ["date", "value"]
            df = pd.DataFrame({"date": pd.to_datetime(raw["date"], utc=True),
                               "value": pd.to_numeric(raw["value"], errors="coerce")}).dropna()
            res[name] = _store("CONTEXT", name, df, user_data)
        except Exception as e:  # noqa: BLE001
            res[name] = f"Fehler: {str(e)[:120]}"
    return res


def context_features(dates: pd.Series, user_data: Path | None = None) -> pd.DataFrame:
    """Gesamtmarkt je Kerze. Tageswerte gelten erst ab dem Folgetag (FRED: Schlusskurs des Vortags, Unsicher wann genau
    veröffentlicht -> 36 h Puffer; Fear & Greed: Wert des Tages ab 1 h nach Veröffentlichung)."""
    idx = pd.DataFrame({"date": pd.to_datetime(dates, utc=True).astype("datetime64[ns, UTC]")})
    key = pd.DataFrame({"t": idx["date"], "i": idx.index}).sort_values("t")
    out = pd.DataFrame(index=idx.index, columns=CONTEXT_FEATURES, dtype=float)

    def series(kind: str, delay: pd.Timedelta, cols: dict[str, Callable[[pd.Series], pd.Series]]) -> None:
        df = load("CONTEXT", kind, user_data)
        if df.empty:
            return
        df = df.sort_values("date").reset_index(drop=True)
        right = pd.DataFrame({"t": (pd.to_datetime(df["date"], utc=True) + delay).astype("datetime64[ns, UTC]")})
        for c, fn in cols.items():
            right[c] = fn(df["value"].astype(float)).to_numpy()
        m = (pd.merge_asof(key, right, on="t", direction="backward", tolerance=pd.Timedelta(days=5))  # Wochenenden
             .set_index("i").reindex(idx.index))
        out[list(cols)] = m[list(cols)]

    series("fear_greed", pd.Timedelta(hours=1), {"ctx_fear_greed": lambda v: v / 100 - 0.5,
                                                 "ctx_fear_greed_chg_7d": lambda v: v.diff(7) / 100})
    day = pd.Timedelta(hours=36)
    series("vix", day, {"ctx_vix": lambda v: v / 100, "ctx_vix_chg_5d": lambda v: v.pct_change(5, fill_method=None)})
    series("us10y", day, {"ctx_us10y_chg_20d": lambda v: v.diff(20)})
    series("dollar", day, {"ctx_dollar_chg_20d": lambda v: v.pct_change(20, fill_method=None)})
    series("spx", day, {"ctx_spx_ret_5d": lambda v: v.pct_change(5, fill_method=None),
                        "ctx_spx_ret_20d": lambda v: v.pct_change(20, fill_method=None)})
    series("ndx", day, {"ctx_ndx_ret_5d": lambda v: v.pct_change(5, fill_method=None),
                        "ctx_ndx_ret_20d": lambda v: v.pct_change(20, fill_method=None)})
    return out.astype(float)


def all_features(pair: str, dates: pd.Series, minutes: int, user_data: Path | None = None) -> pd.DataFrame:
    """Trader-Verhalten (je Coin) + Gesamtmarkt/Stimmung. Fehler beim Lesen -> NaN (nie ein Handelsstopp)."""
    try:
        f = features(pair, dates, minutes, user_data)
    except Exception:  # noqa: BLE001
        log.exception("Marktdaten %s nicht lesbar", pair)
        f = pd.DataFrame(index=pd.RangeIndex(len(dates)), columns=FLOW_FEATURES, dtype=float)
    try:
        c = context_features(dates, user_data)
    except Exception:  # noqa: BLE001
        log.exception("Gesamtmarktdaten nicht lesbar")
        c = pd.DataFrame(index=pd.RangeIndex(len(dates)), columns=CONTEXT_FEATURES, dtype=float)
    return pd.concat([f.reset_index(drop=True), c.reset_index(drop=True)], axis=1)


def features(pair: str, dates: pd.Series, minutes: int, user_data: Path | None = None) -> pd.DataFrame:
    """Merkmale je Kerze (``dates`` = Kerzenbeginn). Nur Werte, die beim Schließen der vorherigen Kerze bekannt waren.

    Fehlende Daten bleiben NaN (die Aufrufer entscheiden, ob neutral aufgefüllt wird)."""
    idx = pd.DataFrame({"date": pd.to_datetime(dates, utc=True)})
    out = pd.DataFrame(index=idx.index, columns=FLOW_FEATURES, dtype=float)
    # Bekannt zum Zeitpunkt: Beginn der aktuellen Kerze (= Schluss der vorherigen).
    known = idx["date"].astype("datetime64[ns, UTC]")
    key = pd.DataFrame({"t": known, "i": idx.index}).sort_values("t")

    def asof(df: pd.DataFrame, ready: pd.Series, cols: list[str], tol: pd.Timedelta) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(index=idx.index, columns=cols, dtype=float)
        right = pd.DataFrame({"t": ready.astype("datetime64[ns, UTC]").to_numpy()}, index=df.index).join(df[cols])
        # tolerance: zu alte Werte (z. B. Dienst war aus) nicht weitertragen -> NaN
        m = pd.merge_asof(key, right.sort_values("t"), on="t", direction="backward", tolerance=tol)
        return m.set_index("i").reindex(idx.index)[cols]

    tk = load(pair, "taker", user_data)
    if not tk.empty:
        tk = tk.sort_values("date").reset_index(drop=True)
        vol = tk["volume"].replace(0, np.nan)
        tk["flow_taker_buy"] = tk["taker_buy"] / vol - 0.5
        tk["flow_taker_buy_4h"] = tk["taker_buy"].rolling(16).sum() / tk["volume"].rolling(16).sum() - 0.5
        ready = pd.to_datetime(tk["date"], utc=True) + pd.Timedelta(minutes=15)  # erst nach Kerzenschluss bekannt
        out[["flow_taker_buy", "flow_taker_buy_4h"]] = asof(tk, ready, ["flow_taker_buy", "flow_taker_buy_4h"],
                                                            pd.Timedelta(hours=2))
    fu = load(pair, "funding", user_data)
    if not fu.empty:
        fu = fu.sort_values("date").reset_index(drop=True)
        fu["flow_funding"] = fu["funding_rate"] * 1000  # in Promille
        out[["flow_funding"]] = asof(fu, pd.to_datetime(fu["date"], utc=True), ["flow_funding"],
                                     pd.Timedelta(hours=12))
    ra = load(pair, "ratio", user_data)
    if not ra.empty:
        ra = ra.sort_values("date").reset_index(drop=True)
        ra["flow_ls_accounts"] = np.log(ra["ls_accounts"].where(ra["ls_accounts"] > 0))
        ra["flow_ls_top"] = np.log(ra["ls_top"].where(ra["ls_top"] > 0))
        ra["flow_oi_change_4h"] = ra["oi_value"].pct_change(16, fill_method=None)
        ready = pd.to_datetime(ra["date"], utc=True) + pd.Timedelta(minutes=15)  # Zeitstempel = Periodenbeginn (Unsicher)
        cols = ["flow_ls_accounts", "flow_ls_top", "flow_oi_change_4h"]
        out[cols] = asof(ra, ready, cols, pd.Timedelta(hours=2))
    return out.astype(float)


def loop(pairs: list[str], user_data: Path | None = None, once: bool = False) -> None:
    """Dienst ``market-flow``: jede Minute Kaufdruck + Orderbuch, alle 5 Minuten Terminmarkt, alle 6 h Gesamtmarkt."""
    last_slow = last_ctx = pd.Timestamp(0, tz="UTC")
    while True:
        now = pd.Timestamp.now(tz="UTC")
        slow = now - last_slow >= pd.Timedelta(minutes=5)
        res = update_all(pairs, user_data, slow=slow)
        if slow:
            last_slow = now
            try:  # Übersicht „Wie viele Daten nutzt die KI?“ für Dashboard/Desktop-App
                from kitb import data_overview

                data_overview.write(user_data)
            except Exception as e:  # noqa: BLE001
                print(f"Datenübersicht: Fehler {e}", flush=True)
        if now - last_ctx >= pd.Timedelta(hours=6):
            res["Gesamtmarkt"] = update_context(user_data)
            last_ctx = now
        errors = {p: {k: v for k, v in r.items() if isinstance(v, str)} for p, r in res.items()}
        errors = {p: e for p, e in errors.items() if e}
        new = sum(v for r in res.values() for v in r.values() if isinstance(v, int))
        print(f"{now:%H:%M} Marktdaten: {new} neue Zeilen" + (f", Fehler: {errors}" if errors else ""), flush=True)
        if once:
            return
        time.sleep(max(5.0, 60.0 - (pd.Timestamp.now(tz="UTC") - now).total_seconds()))
