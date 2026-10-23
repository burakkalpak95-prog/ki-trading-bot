"""Marktdaten für die KIs (kitb.market_flow): Sammeln mit Fake-Börse, kein Blick in die Zukunft, Einbau in die KIs."""

import numpy as np
import pandas as pd

from kitb import market_flow, paths, shared_memory

T0 = pd.Timestamp("2026-10-01", tz="UTC")


def _ms(ts):
    return int(ts.timestamp() * 1000)


def fake_get(url, params):
    if url.endswith("/api/v3/klines"):
        start = pd.Timestamp(params["startTime"], unit="ms", tz="UTC")
        start = max(start, T0)
        end = T0 + pd.Timedelta(days=2)
        dates = pd.date_range(start, end, freq="15min", inclusive="left")[: params["limit"]]
        # Kerze i: Volumen 100, aggressive Käufe 50 + (i % 10) -> Anteil 0,50..0,59
        return [[_ms(d), "1", "1", "1", "1", "100", _ms(d + pd.Timedelta(minutes=15)) - 1, "0", 0,
                 str(50 + (int((d - T0) / pd.Timedelta(minutes=15)) % 10)), "0", "0"] for d in dates]
    if url.endswith("/fapi/v1/fundingRate"):
        return [{"fundingTime": _ms(T0 + pd.Timedelta(hours=8 * i)), "fundingRate": "0.0001"} for i in range(6)]
    if "/futures/data/" in url:
        key = "sumOpenInterestValue" if "openInterest" in url else "longShortRatio"
        return [{"timestamp": _ms(T0 + pd.Timedelta(minutes=15 * i)), key: str(1.0 + i / 100)} for i in range(192)]
    if url.endswith("/api/v3/depth"):
        return {"bids": [["99.9", "10"], ["99", "5"]], "asks": [["100.1", "5"], ["105", "100"]]}
    if url.startswith(market_flow.FNG_URL):
        return {"data": [{"timestamp": str(int((T0 - pd.Timedelta(days=i)).timestamp())), "value": str(40 + i)}
                         for i in range(30)]}
    raise AssertionError(url)


def fake_text(url, params):
    dates = pd.date_range(T0 - pd.Timedelta(days=40), T0 + pd.Timedelta(days=2), freq="D")
    return "observation_date,X\n" + "\n".join(f"{d:%Y-%m-%d},{100 + i}" for i, d in enumerate(dates))


def _fill(ud):
    for pair in ("BTC/USDT",):
        res = market_flow.update_all([pair], ud, get=fake_get)
        assert all(isinstance(v, int) for v in res[pair].values()), res
    assert all(isinstance(v, int) for v in market_flow.update_context(ud, get=fake_get, get_text=fake_text).values())


def test_collect_is_incremental_and_idempotent(tmp_path):
    ud = tmp_path / "ud"
    _fill(ud)
    n = len(market_flow.load("BTC/USDT", "taker", ud))
    assert n > 100
    again = market_flow.update_all(["BTC/USDT"], ud, get=fake_get)["BTC/USDT"]
    assert again["taker"] == 0 and again["funding"] == 0 and again["ratio"] == 0
    assert len(market_flow.load("BTC/USDT", "taker", ud)) == n
    d = market_flow.load("BTC/USDT", "depth", ud)
    assert len(d) == 1 and d["depth_imbalance"].iloc[0] > 0  # mehr Kaufaufträge nahe am Kurs


def test_features_use_only_previous_closed_candle(tmp_path):
    ud = tmp_path / "ud"
    _fill(ud)
    dates = pd.Series(pd.date_range(T0 + pd.Timedelta(hours=6), periods=20, freq="15min", tz="UTC"))
    f = market_flow.features("BTC/USDT", dates, 15, ud)
    # Zeile mit Kerzenbeginn t darf nur den Kaufdruck der Kerze t-15min kennen
    for i, d in enumerate(dates):
        prev = int((d - pd.Timedelta(minutes=15) - T0) / pd.Timedelta(minutes=15))
        assert np.isclose(f["flow_taker_buy"].iloc[i], (50 + prev % 10) / 100 - 0.5)
    assert f["flow_funding"].notna().all() and f["flow_ls_top"].notna().all()


def test_stale_values_are_not_carried_forward(tmp_path):
    ud = tmp_path / "ud"
    _fill(ud)
    late = pd.Series([T0 + pd.Timedelta(days=10)])
    f = market_flow.features("BTC/USDT", late, 15, ud)
    assert f[["flow_taker_buy", "flow_ls_top", "flow_funding"]].isna().all(axis=None)


def test_context_waits_for_publication(tmp_path):
    ud = tmp_path / "ud"
    _fill(ud)
    c = market_flow.context_features(pd.Series([T0 + pd.Timedelta(hours=12), T0 + pd.Timedelta(hours=40)]), ud)
    # FRED-Tageswert vom 01.10. erst nach 36 h nutzbar -> um 12:00 gilt noch der Wert vom 30.09.
    assert c["ctx_spx_ret_5d"].notna().all()
    assert c["ctx_fear_greed"].iloc[0] == (40 / 100 - 0.5)


def test_missing_data_never_breaks_snapshot(tmp_path):
    df = pd.DataFrame({"date": pd.date_range(T0, periods=300, freq="15min", tz="UTC"),
                       "open": 1.0, "high": 1.1, "low": 0.9, "close": np.linspace(1, 2, 300), "volume": 10.0})
    snap = shared_memory.market_snapshot(df, 15, "BTC/USDT", tmp_path / "leer")
    assert list(snap.columns) == shared_memory.SNAPSHOT_FEATURES
    assert snap[market_flow.ALL_FEATURES].isna().all(axis=None)


def test_snapshot_and_reward_learner_use_flow(tmp_path):
    from kitb import reward_learner

    ud = tmp_path / "ud"
    _fill(ud)
    df = pd.DataFrame({"date": pd.date_range(T0, periods=150, freq="15min", tz="UTC"),
                       "open": 1.0, "high": 1.1, "low": 0.9, "close": np.linspace(1, 2, 150), "volume": 10.0})
    snap = shared_memory.market_snapshot(df, 15, "BTC/USDT", ud)
    assert snap["flow_taker_buy"].iloc[-1] == snap["flow_taker_buy"].iloc[-1]  # nicht NaN
    m = reward_learner.RewardModel()
    assert m.design(snap.iloc[-1].to_dict()).shape == (1, len(shared_memory.SNAPSHOT_FEATURES) + 1)


def test_ki_trader_gets_flow_features(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("KiTrader", paths.USER_DATA / "strategies" / "KiTrader.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    s = mod.KiTrader.__new__(mod.KiTrader)
    s.timeframe = "15m"
    df = pd.DataFrame({"date": pd.date_range(T0, periods=300, freq="15min", tz="UTC"),
                       "open": 1.0, "high": 1.1, "low": 0.9, "close": np.linspace(1, 2, 300), "volume": 10.0})
    out = s.feature_engineering_standard(df.copy(), {"pair": "BTC/USDT"})
    cols = [f"%-{c}" for c in market_flow.ALL_FEATURES]
    assert set(cols) <= set(out.columns) and out[cols].notna().all(axis=None)  # ohne Daten neutral 0


def test_data_overview_counts_sources_and_own_trades(tmp_path):
    from kitb import data_overview

    ud = tmp_path / "ud"
    _fill(ud)
    o = data_overview.overview(ud)
    names = [r["Quelle"] for r in o["eingefuehrt"]]
    assert any("Kaufdruck" in n for n in names) and any("Fear" in n for n in names)
    assert o["eingefuehrt_zeilen"] > 100 and o["eigene_trades_gesamt"] == 0
    assert [r["KI"] for r in o["nutzung"]] == ["KI 1 (Muster)", "KI 2 (Belohnung)"]
    data_overview.write(ud)
    assert data_overview.read(ud)["eingefuehrt_zeilen"] == o["eingefuehrt_zeilen"]


def test_news_headlines_recorded_with_sentiment(tmp_path):
    rss = """<rss><channel>
    <item><title>Bitcoin price surges to record high</title><pubDate>Thu, 23 Oct 2026 10:00:00 GMT</pubDate><source>X</source></item>
    <item><title>Exchange hacked, bitcoin plunges</title><pubDate>Thu, 23 Oct 2026 11:00:00 GMT</pubDate><source>Y</source></item>
    </channel></rss>"""
    ud = tmp_path / "ud"
    assert market_flow.record_news("BTC/USDT", ud, get_text=lambda u, p: rss) == 2
    assert market_flow.record_news("BTC/USDT", ud, get_text=lambda u, p: rss) == 0  # keine Doppel
    n = market_flow.load("BTC/USDT", "news", ud).set_index("source")["score"]
    assert n["X"] > 0 > n["Y"]
