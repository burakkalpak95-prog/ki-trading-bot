"""Marktdaten „was alle Trader gerade tun“ als Lerndaten für beide KIs (Burak, 23.10.2026).

Quellen (öffentlich, kostenlos, ohne Konto/Keys, nur lesend):
- Binance Spot, Kerzen inkl. aggressiver Käufe (taker buy volume) -> Kauf-/Verkaufsdruck. Verlauf ab 2022 nachladbar.
- Binance USDT-Terminmarkt (nur als Information, gehandelt wird weiter nur Spot): Finanzierungsrate (Verlauf ab 2022),
  Long/Short-Verhältnis aller Konten und der Top-Trader, offene Positionen. Die API liefert nur ~30 Tage; der Verlauf
  davor kommt aus dem öffentlichen Binance-Archiv data.binance.vision (Tagesdateien, ab 2020/2021 je Coin).
- Binance Spot Orderbuch: Ungleichgewicht der Kauf-/Verkaufsaufträge nahe am Kurs (nur Aufzeichnung, kein Verlauf).
- Gesamtmarkt/Stimmung (für alle Coins gleich): Krypto-Fear-&-Greed-Index (alternative.me, ab 2018) und aus der
  Datenbank der US-Notenbank St. Louis (FRED, CSV ohne Key): VIX, Rendite 10-jähriger US-Anleihen, Dollar-Index, S&P 500,
  Nasdaq 100 (Krypto läuft seit etwa 2020 oft ähnlich wie Technologie-Aktien, Stärke schwankt – Unsicher).
- Weitere freie Quellen ohne Key (GitHub-/Web-Recherche 23.10.2026): Deribit-Volatilitätsindex DVOL für BTC/ETH (was
  Optionshändler an Schwankung erwarten), DefiLlama Stablecoin-Menge (neues Geld im Kryptomarkt), Coin Metrics Community
  (aktive Adressen und MVRV für BTC/ETH; Lizenz CC BY-NC 4.0, für private Nutzung frei). Antwortformate vor dem ersten
  Lauf am PC noch nicht geprüft (Unsicher) – Fehler werden nur protokolliert, die KIs bekommen dann NaN/0.

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
                 "flow_oi_change_4h", "flow_pro_net_24h"]
FLOW_TEXT = {
    "flow_taker_buy": "Kaufdruck (Anteil aggressiver Käufe, letzte Kerze)",
    "flow_taker_buy_4h": "Kaufdruck der letzten 4 Stunden",
    "flow_funding": "Finanzierungsrate am Terminmarkt",
    "flow_ls_accounts": "Long/Short-Verhältnis aller Trader",
    "flow_ls_top": "Long/Short-Verhältnis der Top-Trader",
    "flow_oi_change_4h": "Veränderung der offenen Positionen (4 h)",
    "flow_pro_net_24h": "Profi-Trader (Hyperliquid): Long- minus Short-Eröffnungen 24 h",
}
KINDS = ("taker", "funding", "ratio", "depth", "news", "alpha")
FNG_URL = "https://api.alternative.me/fng/"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_SERIES = {"VIXCLS": "vix", "DGS10": "us10y", "DTWEXBGS": "dollar", "SP500": "spx", "NASDAQ100": "ndx"}
CONTEXT_FEATURES = ["ctx_fear_greed", "ctx_fear_greed_chg_7d", "ctx_vix", "ctx_vix_chg_5d", "ctx_us10y_chg_20d",
                    "ctx_dollar_chg_20d", "ctx_spx_ret_5d", "ctx_spx_ret_20d", "ctx_ndx_ret_5d", "ctx_ndx_ret_20d",
                    "ctx_dvol_btc", "ctx_dvol_btc_chg_5d", "ctx_dvol_eth", "ctx_stable_chg_7d", "ctx_stable_chg_30d",
                    "ctx_btc_active_chg_7d", "ctx_btc_mvrv", "ctx_eth_active_chg_7d",
                    "ctx_wiki_btc_chg_7d", "ctx_wiki_eth_chg_7d", "ctx_wiki_crypto_chg_7d",
                    "ctx_cot_btc_lev_net", "ctx_cot_btc_am_net", "ctx_cot_eth_lev_net"]
WIKI_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{page}/daily/{a}/{b}"
WIKI_PAGES = {"Bitcoin": "wiki_btc", "Ethereum": "wiki_eth", "Cryptocurrency": "wiki_crypto"}
# Wikimedia verlangt einen aussagekräftigen User-Agent mit Kontakt; hier das (öffentliche) Projekt-Repo, keine E-Mail.
WIKI_UA = "ki-trading-bot/1.0 (private research; https://github.com/burakkalpak95-prog/ki-trading-bot)"
DERIBIT_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
STABLE_URL = "https://stablecoins.llama.fi/stablecoincharts/all"
COINMETRICS_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
ONCHAIN_SERIES = {("btc", "AdrActCnt"): "btc_active", ("btc", "CapMVRVCur"): "btc_mvrv", ("eth", "AdrActCnt"): "eth_active"}
EXTRA_CONTEXT = ("dvol_btc", "dvol_eth", "stablecoins", *ONCHAIN_SERIES.values(), *WIKI_PAGES.values())
CONTEXT_TEXT = {
    "ctx_fear_greed": "Krypto-Stimmung (Fear & Greed)", "ctx_fear_greed_chg_7d": "Stimmungsänderung 7 Tage",
    "ctx_vix": "Angstbarometer der Aktienbörse (VIX)", "ctx_vix_chg_5d": "VIX-Änderung 5 Tage",
    "ctx_us10y_chg_20d": "Änderung US-Zinsen (10 Jahre) 20 Tage", "ctx_dollar_chg_20d": "Dollar-Stärke 20 Tage",
    "ctx_spx_ret_5d": "S&P 500 letzte 5 Tage", "ctx_spx_ret_20d": "S&P 500 letzte 20 Tage",
    "ctx_ndx_ret_5d": "Nasdaq 100 (Technologie) letzte 5 Tage", "ctx_ndx_ret_20d": "Nasdaq 100 letzte 20 Tage",
    "ctx_dvol_btc": "Erwartete Schwankung BTC (Optionen, DVOL)", "ctx_dvol_btc_chg_5d": "DVOL-Änderung 5 Tage",
    "ctx_dvol_eth": "Erwartete Schwankung ETH (Optionen, DVOL)", "ctx_stable_chg_7d": "Stablecoin-Menge 7 Tage",
    "ctx_stable_chg_30d": "Stablecoin-Menge 30 Tage", "ctx_btc_active_chg_7d": "Aktive BTC-Adressen 7 Tage",
    "ctx_btc_mvrv": "BTC Marktwert/realisierter Wert (MVRV)", "ctx_eth_active_chg_7d": "Aktive ETH-Adressen 7 Tage",
    "ctx_wiki_btc_chg_7d": "Aufmerksamkeit: Wikipedia-Aufrufe „Bitcoin“ 7 Tage",
    "ctx_wiki_eth_chg_7d": "Aufmerksamkeit: Wikipedia-Aufrufe „Ethereum“ 7 Tage",
    "ctx_wiki_crypto_chg_7d": "Aufmerksamkeit: Wikipedia-Aufrufe „Cryptocurrency“ 7 Tage",
    "ctx_cot_btc_lev_net": "Hedgefonds netto in BTC-Futures (CFTC, wöchentlich)",
    "ctx_cot_btc_am_net": "Vermögensverwalter netto in BTC-Futures (CFTC, wöchentlich)",
    "ctx_cot_eth_lev_net": "Hedgefonds netto in ETH-Futures (CFTC, wöchentlich)",
}
# Profi-Daten (Burak 25.10.2026, „Beides einbauen“): öffentlich und kostenlos, ohne Key.
# CFTC „Traders in Financial Futures“ (Socrata-Schnittstelle), CME Bitcoin 133741 / Ether 146021. Stand dienstags,
# veröffentlicht freitags -> 4 Tage Verzug. Datensatz-Kennung und Spaltennamen aus dem Gedächtnis, Unsicher: auf dem PC
# prüfen (Fehler landen nur in der Ausgabe, nie im Handel).
COT_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
COT_MARKETS = {"133741": "btc", "146021": "eth"}
# Hyperliquid: öffentliche Rangliste + Trades je Wallet. Nur Wallets, die im Monat UND insgesamt Gewinn machen und
# nicht vorwiegend Market Making betreiben (Monatsumsatz ≤ 50 × Kontowert). Test 25.10.: Historie nur ~2.000 Trades je
# Wallet, deshalb wird ab jetzt laufend aufgezeichnet; Nutzungsbedingungen ungeprüft (Seite gesperrt).
HL_LEADERBOARD = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
HL_INFO = "https://api.hyperliquid.xyz/info"
HL_WALLETS = 40
HL_MAX_TURNOVER = 50
ALPHA_FEATURES = ["alpha_ts_klen", "alpha_x_illiq20", "alpha_x_vol20", "alpha_x_beta60", "alpha_x_high60",
                  "alpha_ts_corr20"]  # = kitb.alphas.LIVE, Rang unter allen Lern-Coins am Vortag
ALL_FEATURES = FLOW_FEATURES + CONTEXT_FEATURES + ALPHA_FEATURES


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

    # FRED bremst manche Abruf-Kennungen und bricht gelegentlich ab (25.10.) -> schlichte Kennung, bis zu 3 Versuche
    for i in range(3):
        try:
            r = requests.get(url, params=params, timeout=30, headers={"User-Agent": "kitb/1.0"})
            r.raise_for_status()
            return r.text
        except requests.RequestException:
            if i == 2:
                raise
            time.sleep(5 * (i + 1))
    raise AssertionError("unerreichbar")


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


ARCHIVE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"
ARCHIVE_DAYS = 180  # reicht für das 90-Tage-Training von KI 1 mit Puffer (Tagesdateien, je Coin und Tag ein Abruf)


def _get_bytes(url: str) -> bytes | None:
    import requests

    time.sleep(0.1)
    r = requests.get(url, timeout=30)
    if r.status_code == 404:  # Tag fehlt im Archiv (z. B. Coin damals noch nicht gelistet)
        return None
    r.raise_for_status()
    return r.content


def backfill_ratios(pair: str, user_data: Path | None = None, days: int = ARCHIVE_DAYS,
                    get_bytes: Callable = _get_bytes, today: pd.Timestamp | None = None) -> int:
    """Long/Short + offene Positionen rückwirkend aus dem öffentlichen Binance-Archiv (data.binance.vision).

    Spalten im Archiv (Namen irreführend, geprüft 23.10.2026): count_long_short_ratio = alle Konten,
    sum_toptrader_long_short_ratio = Top-Trader Positionen, sum_open_interest_value = offene Positionen in USDT.
    5-Minuten-Raster, unsortiert, create_time UTC ohne Zeitzone; übernommen wird jede volle Viertelstunde (wie die API)."""
    import io
    import zipfile

    today = (today or pd.Timestamp.now(tz="UTC")).normalize()
    have = load(pair, "ratio", user_data)
    known = set(pd.to_datetime(have["date"], utc=True).dt.normalize()) if not have.empty else set()
    sym, added = symbol(pair), 0
    for i in range(days, 0, -1):
        day = today - pd.Timedelta(days=i)
        if day in known:
            continue
        raw = get_bytes(f"{ARCHIVE_URL}/{sym}/{sym}-metrics-{day:%Y-%m-%d}.zip")
        if not raw:
            continue
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            df = pd.read_csv(z.open(z.namelist()[0]))
        df["date"] = pd.to_datetime(df["create_time"], utc=True)
        df = df[df["date"].dt.minute % 15 == 0].sort_values("date").drop_duplicates("date")
        out = pd.DataFrame({"date": df["date"], "ls_accounts": df["count_long_short_ratio"].astype(float),
                            "ls_top": df["sum_toptrader_long_short_ratio"].astype(float),
                            "oi_value": df["sum_open_interest_value"].astype(float)})
        added += _store(pair, "ratio", out, user_data)
    return added


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


def update_all(pairs: list[str], user_data: Path | None = None, get: Callable = _get, slow: bool = True,
               get_bytes: Callable = _get_bytes) -> dict:
    """Ein Durchlauf für alle Coins. ``slow``: auch Finanzierungsrate und Verhältnisse (alle 5 Minuten reicht)."""
    out: dict[str, Any] = {}
    for pair in pairs:
        res = {}
        jobs = [("taker", update_taker), ("depth", record_depth)]
        if slow:
            jobs += [("funding", update_funding), ("ratio", update_ratios), ("archive", backfill_ratios)]
        for name, fn in jobs:
            try:
                res[name] = (fn(pair, user_data, get_bytes=get_bytes) if name == "archive"
                             else fn(pair, user_data, get=get))
            except Exception as e:  # noqa: BLE001 – ein fehlender Markt (z. B. kein Terminkontrakt) stoppt nichts
                res[name] = f"Fehler: {str(e)[:120]}"
        out[pair] = res
    return out


COIN_NAMES = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "xrp ripple", "BNB": "bnb binance coin",
              "ADA": "cardano", "DOGE": "dogecoin", "LINK": "chainlink", "AVAX": "avalanche avax", "DOT": "polkadot",
              "LTC": "litecoin", "TRX": "tron trx", "NEAR": "near protocol", "ATOM": "cosmos atom", "BCH": "bitcoin cash",
              "UNI": "uniswap"}
NEWS_URL = "https://news.google.com/rss/search"
# Kleines Stimmungs-Wörterbuch für Schlagzeilen (Annahme, nicht kalibriert – deshalb vorerst nur Aufzeichnung).
_POS = {"surge", "surges", "soar", "soars", "rally", "rallies", "jump", "jumps", "gain", "gains", "record", "high",
        "bull", "bullish", "breakout", "approve", "approved", "approval", "adoption", "inflow", "inflows", "rise",
        "rises", "climb", "climbs", "beat", "upgrade", "partnership", "launch", "etf"}
_NEG = {"crash", "crashes", "plunge", "plunges", "drop", "drops", "fall", "falls", "slump", "bear", "bearish",
        "hack", "hacked", "exploit", "lawsuit", "sue", "sues", "ban", "bans", "fraud", "outflow", "outflows", "sell-off",
        "selloff", "liquidation", "liquidations", "fear", "risk", "warning", "probe", "delist", "loss", "losses"}


def headline_score(title: str) -> float:
    words = {w.strip(".,:;!?'\"()").lower() for w in title.split()}
    pos, neg = len(words & _POS), len(words & _NEG)
    return 0.0 if pos + neg == 0 else (pos - neg) / (pos + neg)


def record_news(pair: str, user_data: Path | None = None, get_text: Callable = _get_text) -> int:
    """Schlagzeilen der letzten 24 h je Coin (Google-News-RSS) mit einfachem Stimmungswert – Aufzeichnung für später.

    Als KI-Merkmal erst einbauen, wenn genug Verlauf vorliegt (kein rückwirkender Abruf möglich)."""
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    coin = pair.split("/")[0]
    q = " OR ".join(f'"{n}"' for n in COIN_NAMES.get(coin, coin.lower()).split()[:1]) + " crypto when:1d"
    root = ET.fromstring(get_text(NEWS_URL, {"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"}))
    rows = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        try:
            when = pd.Timestamp(parsedate_to_datetime(it.findtext("pubDate") or "")).tz_convert("UTC")
        except Exception:  # noqa: BLE001
            continue
        rows.append({"date": when, "title": title, "source": (it.findtext("source") or "").strip(),
                     "score": headline_score(title)})
    if not rows:
        return 0
    new = pd.DataFrame(rows)
    old = load(pair, "news", user_data)
    if not old.empty:
        new = new[~new["title"].isin(set(old["title"]))]
    if new.empty:
        return 0
    df = pd.concat([old, new]) if not old.empty else new
    p = _path(pair, "news", user_data)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    df.sort_values("date").reset_index(drop=True).to_parquet(tmp, index=False)
    os.replace(tmp, p)
    return len(new)


def update_context(user_data: Path | None = None, get: Callable = _get, get_text: Callable = _get_text,
                   pairs: list[str] | None = None) -> dict:
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
    for name, fn in (("dvol", lambda: update_dvol(user_data, get)),
                     ("alphas", lambda: update_alphas(pairs, user_data, get) if pairs else 0), ("stablecoins", lambda: update_stablecoins(user_data, get)),
                     ("onchain", lambda: update_onchain(user_data, get)),
                     ("cot", lambda: update_cot(user_data, get)),
                     ("wiki", lambda: update_wiki(user_data, None if get is _get else get))):
        try:
            out = fn()
            res.update(out if isinstance(out, dict) else {name: out})
        except Exception as e:  # noqa: BLE001
            res[name] = f"Fehler: {str(e)[:120]}"
    return res


def update_alphas(pairs: list[str], user_data: Path | None = None, get: Callable = _get) -> int:
    """Formel-Faktoren (kitb.alphas.LIVE) aus Binance-Tageskerzen aller Coins; nur abgeschlossene Tage."""
    from kitb import alphas

    now = pd.Timestamp.now(tz="UTC")
    frames = {}
    for pair in pairs:
        rows = get(f"{SPOT}/api/v3/klines", {"symbol": symbol(pair), "interval": "1d", "limit": 200})
        df = pd.DataFrame({"date": pd.to_datetime([r[0] for r in rows], unit="ms", utc=True),
                           "close_time": pd.to_datetime([r[6] for r in rows], unit="ms", utc=True),
                           **{c: [float(r[i]) for r in rows] for c, i in
                              (("open", 1), ("high", 2), ("low", 3), ("close", 4), ("volume", 5))}})
        df = df[df["close_time"] < now].set_index("date")
        # Verlauf ab 2022 aus den geladenen Tageskerzen (die API liefert nur 200 Tage) – für Training und Backtests
        try:
            hist = alphas.load_panel([pair], paths.DATA_DIR / "binance")
            if pair in hist["close"].columns:
                old = pd.DataFrame({c: hist[c][pair] for c in ("open", "high", "low", "close", "volume")}).dropna()
                df = pd.concat([old[old.index < df.index.min()], df]) if not df.empty else old
        except Exception:  # noqa: BLE001 – ohne lokale Daten nur die 200 Tage der API
            log.exception("Tageskerzen-Verlauf %s nicht lesbar", pair)
        frames[pair] = df
    panel = {c: pd.DataFrame({p: f[c] for p, f in frames.items()}).sort_index()
             for c in ("open", "high", "low", "close", "volume")}
    return sum(_store(p, "alpha", df, user_data) for p, df in alphas.daily_ranks(panel).items())


def _get_wiki(url: str, params: dict[str, Any]) -> Any:
    import requests

    r = requests.get(url, params=params, timeout=30, headers={"User-Agent": WIKI_UA})
    r.raise_for_status()
    return r.json()


def update_wiki(user_data: Path | None = None, get: Callable | None = None) -> dict[str, int]:
    """Wikipedia-Seitenaufrufe je Tag (Wikimedia, frei, ohne Key) – Maß dafür, wie viele Menschen sich gerade für
    Krypto interessieren (neue Käufer kommen oft mit steigender Aufmerksamkeit, Unsicher wie stark)."""
    get = get or _get_wiki
    a, b = BACKFILL_FROM.replace("-", ""), f"{pd.Timestamp.now(tz='UTC'):%Y%m%d}"
    res = {}
    for page, name in WIKI_PAGES.items():
        items = get(WIKI_URL.format(page=page, a=a, b=b), {}).get("items") or []
        df = pd.DataFrame({"date": pd.to_datetime([i["timestamp"][:8] for i in items], format="%Y%m%d", utc=True),
                           "value": [float(i["views"]) for i in items]})
        res[name] = _store("CONTEXT", name, df, user_data)
    return res


def update_dvol(user_data: Path | None = None, get: Callable = _get) -> dict[str, int]:
    """Deribit-Volatilitätsindex (DVOL) als Tageskerzen; gespeichert wird der Schlusswert, Datum = Tagesbeginn."""
    res = {}
    for cur in ("BTC", "ETH"):
        start, end = pd.Timestamp(BACKFILL_FROM, tz="UTC"), pd.Timestamp.now(tz="UTC")
        rows: list = []
        while start < end:
            stop = min(start + pd.Timedelta(days=500), end)
            r = get(DERIBIT_URL, {"currency": cur, "start_timestamp": _ms(start), "end_timestamp": _ms(stop),
                                  "resolution": "1D"})
            rows += (r.get("result") or {}).get("data") or []
            start = stop
        df = pd.DataFrame({"date": pd.to_datetime([int(x[0]) for x in rows], unit="ms", utc=True),
                           "value": [float(x[4]) for x in rows]}).drop_duplicates("date")
        res[f"dvol_{cur.lower()}"] = _store("CONTEXT", f"dvol_{cur.lower()}", df, user_data)
    return res


def update_stablecoins(user_data: Path | None = None, get: Callable = _get) -> int:
    """Gesamtmenge aller Dollar-Stablecoins (DefiLlama) je Tag – wächst sie, fließt neues Geld in den Kryptomarkt."""
    rows = get(STABLE_URL, {})
    vals = [(r["date"], (r.get("totalCirculatingUSD") or r.get("totalCirculating") or {}).get("peggedUSD")) for r in rows]
    df = pd.DataFrame({"date": pd.to_datetime([int(d) for d, v in vals if v], unit="s", utc=True),
                       "value": [float(v) for d, v in vals if v]})
    return _store("CONTEXT", "stablecoins", df[df["date"] >= pd.Timestamp(BACKFILL_FROM, tz="UTC")], user_data)


def update_onchain(user_data: Path | None = None, get: Callable = _get) -> dict[str, int]:
    """Coin Metrics Community (ohne Key, max. 10 Abfragen/6 s): aktive Adressen, MVRV – Tageswerte."""
    res = {}
    for asset in sorted({a for a, _ in ONCHAIN_SERIES}):
        metrics = [m for a, m in ONCHAIN_SERIES if a == asset]
        params: dict[str, Any] = {"assets": asset, "metrics": ",".join(metrics), "frequency": "1d",
                                  "start_time": BACKFILL_FROM, "page_size": 10000}
        data: list = []
        url = COINMETRICS_URL
        for _ in range(10):
            r = get(url, params)
            data += r.get("data") or []
            url, params = r.get("next_page_url"), {}
            if not url:
                break
        raw = pd.DataFrame(data)
        for m in metrics:
            name = ONCHAIN_SERIES[(asset, m)]
            if raw.empty or m not in raw:
                res[name] = "Fehler: Kennzahl fehlt"
                continue
            df = pd.DataFrame({"date": pd.to_datetime(raw["time"], utc=True),
                               "value": pd.to_numeric(raw[m], errors="coerce")}).dropna()
            res[name] = _store("CONTEXT", name, df, user_data)
    return res


def update_cot(user_data: Path | None = None, get: Callable = _get) -> dict[str, Any]:
    """CFTC-Wochenbericht: Netto-Position (Long − Short, Anteil an allen offenen Kontrakten) von Hedgefonds („Leveraged
    Funds“) und Vermögensverwaltern in CME-Bitcoin- und Ether-Futures. Gespeichert je Gruppe als CONTEXT/cot_<coin>_<gruppe>."""
    res: dict[str, Any] = {}
    for code, coin in COT_MARKETS.items():
        rows = get(COT_URL, {"cftc_contract_market_code": code, "$limit": 2000,
                             "$order": "report_date_as_yyyy_mm_dd"})
        raw = pd.DataFrame(rows)
        if raw.empty:
            res[f"cot_{coin}"] = "Fehler: keine Daten"
            continue
        date = pd.to_datetime(raw["report_date_as_yyyy_mm_dd"], utc=True)
        oi = pd.to_numeric(raw.get("open_interest_all"), errors="coerce").replace(0, np.nan)
        for grp, pre in (("lev", "lev_money_positions"), ("am", "asset_mgr_positions")):
            if f"{pre}_long" not in raw:
                res[f"cot_{coin}_{grp}"] = "Fehler: Spalte fehlt"
                continue
            net = (pd.to_numeric(raw[f"{pre}_long"], errors="coerce")
                   - pd.to_numeric(raw[f"{pre}_short"], errors="coerce")) / oi
            res[f"cot_{coin}_{grp}"] = _store("CONTEXT", f"cot_{coin}_{grp}",
                                              pd.DataFrame({"date": date, "value": net}).dropna(), user_data)
    return res


def _post(url: str, body: dict[str, Any]) -> Any:
    import requests

    r = requests.post(url, json=body, timeout=30)
    r.raise_for_status()
    return r.json()


def pro_wallets(board: dict[str, Any], n: int = HL_WALLETS) -> list[str]:
    """Profis aus der Hyperliquid-Rangliste: Monat UND Gesamt im Gewinn, kein reines Market Making."""
    out = []
    for row in board.get("leaderboardRows", []):
        perf = {k: v for k, v in row.get("windowPerformances", [])}
        try:
            value = float(row.get("accountValue") or 0)
            month, total = perf.get("month", {}), perf.get("allTime", {})
            if value <= 0 or float(month.get("pnl", 0)) <= 0 or float(total.get("pnl", 0)) <= 0:
                continue
            if float(month.get("vlm", 0)) > HL_MAX_TURNOVER * value:
                continue
            out.append((float(total["pnl"]), row["ethAddress"]))
        except (KeyError, TypeError, ValueError):
            continue
    return [a for _, a in sorted(out, reverse=True)[:n]]


def update_pro_trades(pairs: list[str], user_data: Path | None = None, get: Callable = _get,
                      post: Callable = _post, wallets: list[str] | None = None, sleep: float = 1.2) -> int:
    """Eröffnungen der Profi-Wallets in unseren Coins (Long/Short) anhängen: CONTEXT/pro_trades (je Trade eine Zeile)."""
    coins = {p.split("/")[0]: p for p in pairs}
    if wallets is None:
        wallets = pro_wallets(get(HL_LEADERBOARD, {}))
    rows = []
    for w in wallets:
        fills, err = None, None
        for i in range(2):
            try:
                fills = post(HL_INFO, {"type": "userFills", "user": w})
                break
            except Exception as e:  # noqa: BLE001 – 429 bei zu schnellen Abfragen: kurz warten, einmal wiederholen
                err = e
                time.sleep(sleep * 4)
        time.sleep(sleep)  # Hyperliquid: ~1200 Gewicht/min je IP, userFills zählt 20
        if fills is None:  # weiter gesperrt: Rest beim nächsten Durchlauf (alle 15 Min.), den Dienst nicht aufhalten
            print(f"pro_trades: Abbruch nach {wallets.index(w)} von {len(wallets)} Wallets ({err})", flush=True)
            break
        for f in fills:
            d = str(f.get("dir", ""))
            if f.get("coin") in coins and d.startswith("Open"):
                rows.append({"date": pd.Timestamp(int(f["time"]), unit="ms", tz="UTC"), "wallet": w,
                             "coin": f["coin"], "side": 1 if "Long" in d else -1,
                             "usd": float(f.get("px", 0)) * float(f.get("sz", 0))})
    if not rows:
        return 0
    p = _path("CONTEXT", "pro_trades", user_data)
    old = load("CONTEXT", "pro_trades", user_data)
    new = pd.DataFrame(rows)
    df = (pd.concat([old, new]) if not old.empty else new).drop_duplicates(["date", "wallet", "coin", "side"])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    df.sort_values("date").reset_index(drop=True).to_parquet(tmp, index=False)
    for i in range(5):
        try:
            os.replace(tmp, p)
            break
        except PermissionError:  # Windows-Bind-Mount: Leser hält die Datei kurz offen
            time.sleep(0.5 * (i + 1))
    return len(df) - len(old)


def pro_net(pair: str, dates: pd.Series, user_data: Path | None = None) -> pd.Series:
    """Je Kerze: (Long- minus Short-Eröffnungen der Profis im Coin, letzte 24 h) / alle Eröffnungen; ohne Daten NaN."""
    known = pd.Series(pd.to_datetime(dates, utc=True)).dt.tz_localize(None).astype("datetime64[ns]")
    out = pd.Series(np.nan, index=range(len(known)))
    df = load("CONTEXT", "pro_trades", user_data)
    if df.empty:
        return out
    df = df[df["coin"] == pair.split("/")[0]]
    start = pd.to_datetime(load("CONTEXT", "pro_trades", user_data)["date"], utc=True).min().tz_localize(None)
    t = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).astype("datetime64[ns]").to_numpy()
    side = df["side"].to_numpy(float)
    order = np.argsort(t)
    t, side = t[order], side[order]
    cum, cnt = np.r_[0.0, np.cumsum(side)], np.arange(len(t) + 1)
    hi = np.searchsorted(t, known.to_numpy(), side="left")  # nur vor Kerzenbeginn bekannt
    lo = np.searchsorted(t, (known - pd.Timedelta(hours=24)).to_numpy(), side="left")
    n = cnt[hi] - cnt[lo]
    val = np.where(n > 0, (cum[hi] - cum[lo]) / np.maximum(n, 1), 0.0)
    # vor Beginn der Aufzeichnung ist „keine Profi-Eröffnung“ unbekannt, nicht 0
    val = np.where(known.to_numpy() >= (start + pd.Timedelta(hours=24)).to_datetime64(), val, np.nan)
    return pd.Series(val, index=out.index)


def context_features(dates: pd.Series, user_data: Path | None = None) -> pd.DataFrame:
    """Gesamtmarkt je Kerze. Tageswerte gelten erst ab dem Folgetag (FRED: Schlusskurs des Vortags, Unsicher wann genau
    veröffentlicht -> 36 h Puffer; Fear & Greed: Wert des Tages ab 1 h nach Veröffentlichung)."""
    idx = pd.DataFrame({"date": pd.to_datetime(dates, utc=True).astype("datetime64[ns, UTC]")})
    key = pd.DataFrame({"t": idx["date"], "i": idx.index}).sort_values("t")
    out = pd.DataFrame(index=idx.index, columns=CONTEXT_FEATURES, dtype=float)

    def series(kind: str, delay: pd.Timedelta, cols: dict[str, Callable[[pd.Series], pd.Series]],
               max_age: pd.Timedelta = pd.Timedelta(days=5)) -> None:
        df = load("CONTEXT", kind, user_data)
        if df.empty:
            return
        df = df.sort_values("date").reset_index(drop=True)
        right = pd.DataFrame({"t": (pd.to_datetime(df["date"], utc=True) + delay).astype("datetime64[ns, UTC]")})
        for c, fn in cols.items():
            right[c] = fn(df["value"].astype(float)).to_numpy()
        m = (pd.merge_asof(key, right, on="t", direction="backward", tolerance=max_age)  # Wochenenden
             .set_index("i").reindex(idx.index))
        out[list(cols)] = m[list(cols)]

    series("fear_greed", pd.Timedelta(hours=1), {"ctx_fear_greed": lambda v: v / 100 - 0.5,
                                                 "ctx_fear_greed_chg_7d": lambda v: v.diff(7) / 100})
    day = pd.Timedelta(hours=36)
    series("vix", day, {"ctx_vix": lambda v: v / 100, "ctx_vix_chg_5d": lambda v: v.pct_change(5, fill_method=None)})
    series("us10y", day, {"ctx_us10y_chg_20d": lambda v: v.diff(20)})
    # Dollar-Index (DTWEXBGS): FRED veröffentlicht ihn nur wöchentlich (montags bis Freitag der Vorwoche) -> der neueste
    # Wert ist normal 3–10 Tage alt; mit 5 Tagen Grenze war das Merkmal fast immer leer.
    series("dollar", day, {"ctx_dollar_chg_20d": lambda v: v.pct_change(20, fill_method=None)},
           max_age=pd.Timedelta(days=14))
    series("spx", day, {"ctx_spx_ret_5d": lambda v: v.pct_change(5, fill_method=None),
                        "ctx_spx_ret_20d": lambda v: v.pct_change(20, fill_method=None)})
    series("ndx", day, {"ctx_ndx_ret_5d": lambda v: v.pct_change(5, fill_method=None),
                        "ctx_ndx_ret_20d": lambda v: v.pct_change(20, fill_method=None)})
    # Tageskerze/Tageswert mit Datum = Tagesbeginn: erst nach Tagesende fertig (+ Puffer bis zur Veröffentlichung)
    series("dvol_btc", pd.Timedelta(hours=25), {"ctx_dvol_btc": lambda v: v / 100,
                                                "ctx_dvol_btc_chg_5d": lambda v: v.pct_change(5, fill_method=None)})
    series("dvol_eth", pd.Timedelta(hours=25), {"ctx_dvol_eth": lambda v: v / 100})
    series("stablecoins", pd.Timedelta(hours=26), {"ctx_stable_chg_7d": lambda v: v.pct_change(7, fill_method=None),
                                                   "ctx_stable_chg_30d": lambda v: v.pct_change(30, fill_method=None)})
    series("btc_active", day, {"ctx_btc_active_chg_7d": lambda v: v.rolling(7).mean().pct_change(7, fill_method=None)})
    series("btc_mvrv", day, {"ctx_btc_mvrv": lambda v: v - 1})
    series("eth_active", day, {"ctx_eth_active_chg_7d": lambda v: v.rolling(7).mean().pct_change(7, fill_method=None)})
    cot = pd.Timedelta(days=4)  # Stand dienstags, veröffentlicht freitags
    for kind, col in (("cot_btc_lev", "ctx_cot_btc_lev_net"), ("cot_btc_am", "ctx_cot_btc_am_net"),
                      ("cot_eth_lev", "ctx_cot_eth_lev_net")):
        series(kind, cot, {col: lambda v: v}, max_age=pd.Timedelta(days=14))
    for kind, col in (("wiki_btc", "ctx_wiki_btc_chg_7d"), ("wiki_eth", "ctx_wiki_eth_chg_7d"),
                      ("wiki_crypto", "ctx_wiki_crypto_chg_7d")):
        series(kind, day, {col: lambda v: np.log(v.rolling(7).mean() / v.rolling(7).mean().shift(7))})
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
    try:
        a = alpha_features(pair, dates, user_data)
    except Exception:  # noqa: BLE001
        log.exception("Faktoren %s nicht lesbar", pair)
        a = pd.DataFrame(index=pd.RangeIndex(len(dates)), columns=ALPHA_FEATURES, dtype=float)
    return pd.concat([f.reset_index(drop=True), c.reset_index(drop=True), a.reset_index(drop=True)], axis=1)


def alpha_features(pair: str, dates: pd.Series, user_data: Path | None = None) -> pd.DataFrame:
    """Rang des Coins bei den Formel-Faktoren; Tageswert (Datum = Tagesbeginn) gilt erst nach Tagesschluss."""
    idx = pd.DataFrame({"date": pd.to_datetime(dates, utc=True).astype("datetime64[ns, UTC]")})
    out = pd.DataFrame(index=idx.index, columns=ALPHA_FEATURES, dtype=float)
    df = load(pair, "alpha", user_data)
    cols = [c for c in ALPHA_FEATURES if c in df.columns]
    if df.empty or not cols:
        return out
    key = pd.DataFrame({"t": idx["date"], "i": idx.index}).sort_values("t")
    right = pd.DataFrame({"t": (pd.to_datetime(df["date"], utc=True) + pd.Timedelta(days=1, minutes=5))
                          .astype("datetime64[ns, UTC]")}).join(df[cols]).sort_values("t")
    m = pd.merge_asof(key, right, on="t", direction="backward", tolerance=pd.Timedelta(days=3))
    out[cols] = m.set_index("i").reindex(idx.index)[cols]
    return out.astype(float)


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
        # Zeitstempel = Messzeitpunkt/Periodenende (geprüft 23.10.2026); 5 Min. Puffer bis zur Veröffentlichung
        ready = pd.to_datetime(ra["date"], utc=True) + pd.Timedelta(minutes=5)
        cols = ["flow_ls_accounts", "flow_ls_top", "flow_oi_change_4h"]
        out[cols] = asof(ra, ready, cols, pd.Timedelta(hours=2))
    out["flow_pro_net_24h"] = pro_net(pair, dates, user_data).to_numpy()
    return out.astype(float)


def loop(pairs: list[str], user_data: Path | None = None, once: bool = False) -> None:
    """Dienst ``market-flow``: jede Minute Kaufdruck + Orderbuch, alle 5 Minuten Terminmarkt, alle 30 Min. Schlagzeilen,
    alle 6 h Gesamtmarkt."""
    last_slow = last_ctx = last_news = last_pro = last_board = pd.Timestamp(0, tz="UTC")
    board: list[str] = []
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
            res["Gesamtmarkt"] = update_context(user_data, pairs=pairs)
            last_ctx = now
        if now - last_pro >= pd.Timedelta(minutes=15):
            try:
                if now - last_board >= pd.Timedelta(hours=24) or not board:
                    board[:] = pro_wallets(_get(HL_LEADERBOARD, {}))
                    last_board = now
                res["Profis"] = {"pro_trades": update_pro_trades(pairs, user_data, wallets=board)}
            except Exception as e:  # noqa: BLE001
                res["Profis"] = {"pro_trades": f"Fehler: {str(e)[:120]}"}
            last_pro = now
        if now - last_news >= pd.Timedelta(minutes=30):
            for pair in pairs:
                try:
                    res.setdefault(pair, {})["news"] = record_news(pair, user_data)
                except Exception as e:  # noqa: BLE001
                    res.setdefault(pair, {})["news"] = f"Fehler: {str(e)[:120]}"
            last_news = now
        errors = {p: {k: v for k, v in r.items() if isinstance(v, str) and v.startswith("Fehler")}
                  for p, r in res.items()}
        errors = {p: e for p, e in errors.items() if e}
        new = sum(v for r in res.values() for v in r.values() if isinstance(v, int))
        print(f"{now:%H:%M} Marktdaten: {new} neue Zeilen" + (f", Fehler: {errors}" if errors else ""), flush=True)
        if once:
            return
        time.sleep(max(5.0, 60.0 - (pd.Timestamp.now(tz="UTC") - now).total_seconds()))
