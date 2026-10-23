"""Test: Sagt eine Schlagzeilen-Bewertung die Kurse danach voraus? (Burak, 23.10.2026: „Ja“ zum Test mit lokaler KI)

Vorgehen, bevor eine lokale Sprach-KI dauerhaft mitläuft:
1. Alte Krypto-Schlagzeilen mit Zeitstempel laden (Datensatz DLT-Sentiment-News, CryptoPanic 2021–05/2025,
   CC BY-NC 4.0, nur private Nutzung). Die Abstimmungs-Spalten des Datensatzes entstehen NACH der Veröffentlichung und
   werden deshalb nicht verwendet.
2. Jede Schlagzeile bewerten: (a) unsere einfache Wortliste, (b) lokales Sprachmodell über Ollama (−1 … +1).
3. Coin aus dem Text bestimmen (ohne Coin-Namen: Gesamtmarkt = BTC). Rendite ab der ersten vollen Stunde NACH der
   Veröffentlichung über 1 h, 4 h, 24 h aus Binance-Stundenkerzen.
4. Rangkorrelation (IC) Bewertung ↔ Rendite, getrennt früh/spät. Nur wenn das Sprachmodell in beiden Zeiträumen
   klar besser ist als die Wortliste, lohnt es sich, es dauerhaft für die KIs laufen zu lassen.
"""

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kitb import market_flow, paths

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
HORIZONS = {"1h": 1, "4h": 4, "24h": 24}
SPLIT = "2023-07-01"  # früh: bis Mitte 2023, spät: danach
PROMPT = ("You rate crypto news headlines for short-term price impact. Reply only with JSON "
          '{{"score": x}} where x is a number from -1 (very bearish) to 1 (very bullish), 0 if neutral or irrelevant.'
          "\nHeadline: {title}")


def load_headlines(path: Path) -> pd.DataFrame:
    """CSV, Parquet oder JSON-Lines mit Zeitstempel und Titel (Spaltennamen werden erkannt)."""
    if path.is_dir():
        files = sorted([*path.rglob("*.parquet"), *path.rglob("*.csv"), *path.rglob("*.jsonl")])
        return pd.concat([load_headlines(f) for f in files], ignore_index=True)
    df = (pd.read_parquet(path) if path.suffix == ".parquet" else
          pd.read_json(path, lines=True) if path.suffix == ".jsonl" else pd.read_csv(path))
    tcol = next(c for c in ("timestamp", "published_at", "date", "time") if c in df.columns)
    title = next(c for c in ("title", "text", "headline") if c in df.columns)
    ts = df[tcol]
    ts = pd.to_datetime(ts, unit="s", utc=True) if pd.api.types.is_numeric_dtype(ts) else pd.to_datetime(ts, utc=True)
    out = pd.DataFrame({"date": ts, "title": df[title].astype(str).str.strip()})
    return out.dropna().drop_duplicates("title").sort_values("date").reset_index(drop=True)


def coin_of(title: str) -> str:
    """Erster genannter Lern-Coin; „bitcoin cash“ vor „bitcoin“. Ohne Coin: Gesamtmarkt (BTC)."""
    t = title.lower()
    for coin, names in sorted(market_flow.COIN_NAMES.items(), key=lambda kv: -len(kv[1])):
        for n in [names] if coin == "BCH" else names.split():
            if re.search(rf"\b{re.escape(n)}\b", t) or re.search(rf"\b{coin.lower()}\b", t):
                return coin
    return "BTC"


def ollama_score(title: str, model: str, post: Callable | None = None) -> float:
    import requests

    post = post or (lambda url, body: requests.post(url, json=body, timeout=120).json())
    r = post(f"{OLLAMA_URL}/api/generate", {"model": model, "prompt": PROMPT.format(title=title), "stream": False,
                                            "format": "json", "options": {"temperature": 0}})
    try:
        return float(np.clip(float(json.loads(r.get("response", "{}"))["score"]), -1, 1))
    except Exception:  # noqa: BLE001
        return float("nan")


def score_all(news: pd.DataFrame, cache: Path, model: str | None, post: Callable | None = None,
              every: int = 200) -> pd.DataFrame:
    """Bewertet alle Schlagzeilen; Zwischenstand in ``cache`` (Abbruch und Fortsetzen möglich)."""
    done = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["title", "llm"])
    known = dict(zip(done["title"], done["llm"]))
    news = news.assign(coin=news["title"].map(coin_of), words=news["title"].map(market_flow.headline_score))
    if model:
        todo = [t for t in news["title"] if t not in known]
        for i, t in enumerate(todo, 1):
            known[t] = ollama_score(t, model, post)
            if i % every == 0 or i == len(todo):
                cache.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"title": list(known), "llm": list(known.values())}).to_parquet(cache, index=False)
                print(f"{i}/{len(todo)} Schlagzeilen bewertet", flush=True)
    return news.assign(llm=news["title"].map(known).astype(float))


def forward_returns(news: pd.DataFrame, closes: dict[str, pd.Series]) -> pd.DataFrame:
    """Rendite ab der ersten vollen Stunde nach der Veröffentlichung (kein Blick zurück in schon bekannte Kurse)."""
    out = news.copy()
    entry = out["date"].dt.floor("h") + pd.Timedelta(hours=1)
    for name, h in HORIZONS.items():
        vals = []
        for coin, t in zip(out["coin"], entry):
            c = closes.get(coin)
            if c is None:
                vals.append(np.nan)
                continue
            a, b = c.get(t - pd.Timedelta(hours=1)), c.get(t + pd.Timedelta(hours=h - 1))  # Schluss = Beginn + 1 h
            vals.append(b / a - 1 if a and b else np.nan)
        out[f"ret_{name}"] = vals
    return out


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, part in (("früh", df[df["date"] < pd.Timestamp(SPLIT, tz="UTC")]),
                         ("spät", df[df["date"] >= pd.Timestamp(SPLIT, tz="UTC")])):
        for method in ("words", "llm"):
            for h in HORIZONS:
                d = part[[method, f"ret_{h}"]].dropna()
                d = d[d[method] != 0]  # neutrale Schlagzeilen sagen nichts
                ic = d[method].rank().corr(d[f"ret_{h}"].rank()) if len(d) > 30 else np.nan
                t = ic * np.sqrt((len(d) - 2) / max(1e-9, 1 - ic ** 2)) if ic == ic else np.nan
                pos, neg = d[d[method] > 0][f"ret_{h}"].mean(), d[d[method] < 0][f"ret_{h}"].mean()
                rows.append({"Zeitraum": period, "Bewertung": "Wortliste" if method == "words" else "Sprachmodell",
                             "Horizont": h, "Schlagzeilen": len(d), "IC": ic, "t": t,
                             "Rendite nach positiv": pos, "Rendite nach negativ": neg})
    return pd.DataFrame(rows)


def run(news_path: Path, *, model: str | None, data_source: str = "binance", out_dir: Path | None = None,
        limit: int | None = None) -> dict[str, Any]:
    from kitb.data.storage import load_ohlcv_raw

    news = load_headlines(news_path)
    if limit:
        news = news.sample(n=min(limit, len(news)), random_state=1).sort_values("date")
    out_dir = out_dir or (paths.USER_DATA / "reports" / "news-test")
    scored = score_all(news, out_dir / f"llm-cache-{(model or 'none').replace(':', '_').replace('/', '_')}.parquet",
                       model)
    closes = {}
    for coin in scored["coin"].unique():
        df = load_ohlcv_raw(paths.DATA_DIR / data_source, f"{coin}/USDT", "1h")
        if not df.empty:
            closes[coin] = df.set_index("date")["close"]
    res = evaluate(forward_returns(scored, closes))
    report = "\n".join([
        f"# Schlagzeilen-Test ({len(scored)} Schlagzeilen, Modell: {model or '—'})", "",
        f"Früh: bis {SPLIT}, spät: danach. IC = Rangkorrelation Bewertung ↔ Rendite (nur nicht-neutrale Schlagzeilen).",
        "Unsicher: Zeitstempel des Datensatzes als UTC angenommen; Einstieg erst zur nächsten vollen Stunde.", "",
        res.to_markdown(index=False, floatfmt=".4f")])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "news-test.md").write_text(report, encoding="utf-8")
    res.to_csv(out_dir / "news-test.csv", index=False)
    return {"report": str(out_dir / "news-test.md"), "headlines": len(scored),
            "coins": scored["coin"].value_counts().head(8).to_dict()}
