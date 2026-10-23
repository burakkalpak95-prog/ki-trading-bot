"""Schlagzeilen-Test (kitb.news_test): Coin-Erkennung, kein Blick zurück, Auswertung, Sprachmodell-Antworten."""

import json

import numpy as np
import pandas as pd

from kitb import news_test


def test_coin_detection():
    assert news_test.coin_of("Bitcoin Cash hard fork ahead") == "BCH"
    assert news_test.coin_of("Solana rallies after upgrade") == "SOL"
    assert news_test.coin_of("SEC delays decision on ETH ETF") == "ETH"
    assert news_test.coin_of("Fed raises rates again") == "BTC"  # Gesamtmarkt


def test_llm_score_parsing_and_cache(tmp_path):
    calls = []

    def post(url, body):
        calls.append(body["prompt"])
        return {"response": json.dumps({"score": 2.0 if "surge" in body["prompt"] else -0.5})}

    news = pd.DataFrame({"date": pd.to_datetime(["2023-01-01 10:20", "2023-01-02 11:00"], utc=True),
                         "title": ["Bitcoin surges", "Ethereum hack"]})
    s = news_test.score_all(news, tmp_path / "c.parquet", "m", post)
    assert list(s["llm"]) == [1.0, -0.5] and len(calls) == 2  # auf −1 … 1 begrenzt
    news_test.score_all(news, tmp_path / "c.parquet", "m", post)
    assert len(calls) == 2  # aus dem Zwischenspeicher
    assert np.isnan(news_test.ollama_score("x", "m", lambda u, b: {"response": "kaputt"}))


def test_forward_return_starts_after_publication_and_evaluates():
    idx = pd.date_range("2022-12-31", "2024-01-10", freq="h", tz="UTC")
    rng = np.random.default_rng(2)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), idx)
    dates = pd.Series(pd.date_range("2023-01-01 10:20", periods=400, freq="22h", tz="UTC"))
    news = pd.DataFrame({"date": dates, "title": "x", "coin": "BTC"})
    entry = dates.dt.floor("h") + pd.Timedelta(hours=1)
    # Bewertung = tatsächliche spätere 4h-Rendite -> perfekte Vorhersage
    fut = [close[t + pd.Timedelta(hours=3)] / close[t - pd.Timedelta(hours=1)] - 1 for t in entry]
    news["words"] = np.sign(fut)
    news["llm"] = fut
    r = news_test.forward_returns(news, {"BTC": close})
    assert np.allclose(r["ret_4h"], fut)
    ev = news_test.evaluate(r).set_index(["Zeitraum", "Bewertung", "Horizont"])
    assert ev.loc[("früh", "Sprachmodell", "4h"), "IC"] > 0.99
    # 10:20 veröffentlicht: Einstieg 11:00 (Schluss der 10-Uhr-Kerze), nicht 10:00
    assert r["ret_1h"].iloc[0] == close[pd.Timestamp("2023-01-01 11:00", tz="UTC")] / close[pd.Timestamp("2023-01-01 10:00", tz="UTC")] - 1
