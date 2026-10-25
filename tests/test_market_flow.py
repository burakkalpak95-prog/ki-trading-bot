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
    days = pd.date_range(T0 - pd.Timedelta(days=60), T0 + pd.Timedelta(days=2), freq="D")
    if url == market_flow.DERIBIT_URL:
        a, b = pd.Timestamp(params["start_timestamp"], unit="ms", tz="UTC"), pd.Timestamp(params["end_timestamp"], unit="ms", tz="UTC")
        return {"result": {"data": [[_ms(d), 50, 55, 45, 50 + i % 7] for i, d in enumerate(days) if a <= d < b]}}
    if url == market_flow.STABLE_URL:
        return [{"date": str(int(d.timestamp())), "totalCirculatingUSD": {"peggedUSD": 1e11 + i * 1e8}} for i, d in enumerate(days)]
    if url == market_flow.COINMETRICS_URL:
        return {"data": [{"asset": params["assets"], "time": f"{d:%Y-%m-%d}T00:00:00.000000000Z",
                          **{m: str(1000 + i) for m in params["metrics"].split(",")}} for i, d in enumerate(days)]}
    if url == market_flow.COT_URL:
        return [{"report_date_as_yyyy_mm_dd": f"{d:%Y-%m-%d}T00:00:00.000", "open_interest_all": "1000",
                 "lev_money_positions_long": "100", "lev_money_positions_short": "300",
                 "asset_mgr_positions_long": "200", "asset_mgr_positions_short": "100"} for d in days[::7]]
    if url.startswith("https://wikimedia.org/"):
        return {"items": [{"timestamp": f"{d:%Y%m%d}00", "views": 1000 + 10 * i} for i, d in enumerate(days)]}
    raise AssertionError(url)


def fake_text(url, params):
    dates = pd.date_range(T0 - pd.Timedelta(days=40), T0 + pd.Timedelta(days=2), freq="D")
    return "observation_date,X\n" + "\n".join(f"{d:%Y-%m-%d},{100 + i}" for i, d in enumerate(dates))


def _fill(ud):
    for pair in ("BTC/USDT",):
        res = market_flow.update_all([pair], ud, get=fake_get, get_bytes=lambda u: None)
        assert all(isinstance(v, int) for v in res[pair].values()), res
    assert all(isinstance(v, int) for v in market_flow.update_context(ud, get=fake_get, get_text=fake_text).values())


def test_collect_is_incremental_and_idempotent(tmp_path):
    ud = tmp_path / "ud"
    _fill(ud)
    n = len(market_flow.load("BTC/USDT", "taker", ud))
    assert n > 100
    again = market_flow.update_all(["BTC/USDT"], ud, get=fake_get, get_bytes=lambda u: None)["BTC/USDT"]
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
    new = ["ctx_dvol_btc", "ctx_dvol_eth", "ctx_stable_chg_7d", "ctx_btc_mvrv", "ctx_btc_active_chg_7d",
           "ctx_wiki_btc_chg_7d", "ctx_wiki_crypto_chg_7d"]
    assert c[new].notna().all(axis=None)
    # Tageswert vom 01.10. (Tagesbeginn) ist um 12:00 am selben Tag noch nicht fertig -> Wert vom 30.09.
    dv = market_flow.load("CONTEXT", "dvol_btc", ud).set_index("date")["value"]
    assert c["ctx_dvol_btc"].iloc[0] == dv[T0 - pd.Timedelta(days=1)] / 100


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
    m = [c for c in out.columns if c.startswith("%-m_")]
    assert len(m) >= 15 and out[m].isin([0.0, 1.0]).all(axis=None) and out[m].to_numpy().sum() > 0


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


def test_archive_backfill_maps_misleading_columns(tmp_path):
    import io
    import zipfile

    rows = ["create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
            "sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio"]
    for m in (20, 0, 5, 10, 15):  # unsortiert wie im Archiv
        rows.append(f"2026-09-01 00:{m:02d}:00,BTCUSDT,1,5000,9,2.5,1.5,1.1")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("BTCUSDT-metrics-2026-09-01.csv", "\n".join(rows))
    calls = []

    def fake_bytes(url):
        calls.append(url)
        return buf.getvalue() if url.endswith("2026-09-01.zip") else None

    ud = tmp_path / "ud"
    n = market_flow.backfill_ratios("BTC/USDT", ud, days=3, get_bytes=fake_bytes, today=pd.Timestamp("2026-09-03", tz="UTC"))
    r = market_flow.load("BTC/USDT", "ratio", ud)
    assert n == 2 and list(r["date"].dt.minute) == [0, 15]  # nur volle Viertelstunden, sortiert
    assert r["ls_accounts"].iloc[0] == 1.5 and r["ls_top"].iloc[0] == 2.5 and r["oi_value"].iloc[0] == 5000
    calls.clear()
    market_flow.backfill_ratios("BTC/USDT", ud, days=3, get_bytes=fake_bytes, today=pd.Timestamp("2026-09-03", tz="UTC"))
    assert not any(c.endswith("2026-09-01.zip") for c in calls)  # vorhandene Tage nicht erneut laden


def test_alpha_ranks_from_daily_klines_without_lookahead(tmp_path):
    days = pd.date_range(T0 - pd.Timedelta(days=150), T0, freq="D")
    rng = np.random.default_rng(5)
    pairs = [f"C{i}/USDT" for i in range(8)]
    walks = {p: 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(days)))) for p in pairs}

    def get(url, params):
        assert params["interval"] == "1d"
        c = walks[params["symbol"].replace("USDT", "/USDT")]
        return [[_ms(d), str(c[i] * 0.99), str(c[i] * 1.02), str(c[i] * 0.97), str(c[i]), str(1000 + i),
                 _ms(d + pd.Timedelta(days=1)) - 1] for i, d in enumerate(days)]

    ud = tmp_path / "ud"
    assert market_flow.update_alphas(pairs, ud, get=get) > 0
    a = market_flow.load("C0/USDT", "alpha", ud)
    assert set(market_flow.ALPHA_FEATURES) <= set(a.columns) and a[market_flow.ALPHA_FEATURES].abs().max().max() <= 0.5
    # Kerze am 10.09. 12:00: Tageswert vom 09.09. ist fertig, der vom 10.09. noch nicht
    t = pd.Timestamp("2026-09-10 12:00", tz="UTC")
    f = market_flow.alpha_features("C0/USDT", pd.Series([t]), ud)
    exp = a.set_index("date").loc[pd.Timestamp("2026-09-09", tz="UTC"), "alpha_x_high60"]
    assert f["alpha_x_high60"].iloc[0] == exp


def test_alpha_history_merges_local_daily_candles(tmp_path, monkeypatch):
    from kitb import paths
    from kitb.data.storage import store_ohlcv

    pairs = [f"C{i}/USDT" for i in range(8)]
    rng = np.random.default_rng(7)
    old_days = pd.date_range(T0 - pd.Timedelta(days=500), T0 - pd.Timedelta(days=151), freq="D")
    api_days = pd.date_range(T0 - pd.Timedelta(days=150), T0, freq="D")
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    for p in pairs:
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(old_days))))
        store_ohlcv(tmp_path / "data" / "binance", p, "1d", pd.DataFrame(
            {"date": old_days, "open": c, "high": c * 1.02, "low": c * 0.97, "close": c, "volume": 1000.0}))

    def get(url, params):
        return [[_ms(d), "100", "102", "97", str(100 + i % 7), "1000", _ms(d + pd.Timedelta(days=1)) - 1]
                for i, d in enumerate(api_days)]

    ud = tmp_path / "ud"
    market_flow.update_alphas(pairs, ud, get=get)
    a = market_flow.load("C0/USDT", "alpha", ud)
    assert a["date"].min() < T0 - pd.Timedelta(days=300)  # Verlauf reicht vor die 200 API-Tage zurück
    assert a["date"].max() >= T0 - pd.Timedelta(days=2)


def test_dollar_index_weekly_publication_still_used(tmp_path):
    """25.10.: FRED veröffentlicht den Dollar-Index nur wöchentlich -> der neueste Wert ist oft >5 Tage alt."""
    ud = tmp_path / "ud"
    days = pd.date_range(T0 - pd.Timedelta(days=60), T0 - pd.Timedelta(days=9), freq="B", tz="UTC")
    market_flow._store("CONTEXT", "dollar", pd.DataFrame({"date": days, "value": np.linspace(120, 125, len(days))}), ud)
    market_flow._store("CONTEXT", "vix", pd.DataFrame({"date": days, "value": 20.0}), ud)
    c = market_flow.context_features(pd.Series([T0]), ud)
    assert c["ctx_dollar_chg_20d"].notna().all()
    assert c["ctx_vix"].isna().all()  # tägliche Reihen: 9 Tage alt bleibt veraltet


def test_cot_net_positions_weekly(tmp_path):
    """Burak 25.10.: Profi-Daten. CFTC-Wochenbericht -> Netto-Anteil Hedgefonds/Vermögensverwalter, erst ab Freitag."""
    ud = tmp_path / "ud"
    tue = [T0 - pd.Timedelta(days=7 * k + 1) for k in range(10)]

    def fake_get(url, params):
        assert url == market_flow.COT_URL
        return [{"report_date_as_yyyy_mm_dd": d.strftime("%Y-%m-%dT00:00:00.000"), "open_interest_all": "1000",
                 "lev_money_positions_long": "100", "lev_money_positions_short": "400",
                 "asset_mgr_positions_long": "300", "asset_mgr_positions_short": "100"} for d in tue]

    res = market_flow.update_cot(ud, get=fake_get)
    assert res["cot_btc_lev"] == 10 and res["cot_eth_am"] == 10
    c = market_flow.context_features(pd.Series([T0]), ud)
    assert np.isclose(c["ctx_cot_btc_lev_net"].iloc[0], -0.3) and np.isclose(c["ctx_cot_btc_am_net"].iloc[0], 0.2)


def test_pro_traders_filtered_and_without_lookahead(tmp_path):
    ud = tmp_path / "ud"
    perf = lambda m, t, v: [["day", {"pnl": "1"}], ["month", {"pnl": str(m), "vlm": str(v)}], ["allTime", {"pnl": str(t)}]]
    board = {"leaderboardRows": [
        {"ethAddress": "0xpro", "accountValue": "1000", "windowPerformances": perf(50, 900, 10_000)},
        {"ethAddress": "0xmm", "accountValue": "1000", "windowPerformances": perf(50, 5000, 10_000_000)},
        {"ethAddress": "0xlose", "accountValue": "1000", "windowPerformances": perf(-5, 900, 100)}]}
    assert market_flow.pro_wallets(board) == ["0xpro"]
    t1, t2 = T0 + pd.Timedelta(hours=30), T0 + pd.Timedelta(hours=40)
    fills = [{"coin": "BTC", "dir": "Open Long", "time": int(t1.timestamp() * 1000), "px": "100", "sz": "1"},
             {"coin": "BTC", "dir": "Open Short", "time": int(t2.timestamp() * 1000), "px": "100", "sz": "1"},
             {"coin": "BTC", "dir": "Close Long", "time": int(t2.timestamp() * 1000), "px": "100", "sz": "1"},
             {"coin": "WLD", "dir": "Open Long", "time": int(t2.timestamp() * 1000), "px": "1", "sz": "1"},
             {"coin": "ETH", "dir": "Open Long", "time": int(T0.timestamp() * 1000), "px": "1", "sz": "1"}]
    n = market_flow.update_pro_trades(["BTC/USDT", "ETH/USDT"], ud, post=lambda u, b: fills, wallets=["0xpro"], sleep=0)
    assert n == 3
    assert market_flow.update_pro_trades(["BTC/USDT"], ud, post=lambda u, b: fills, wallets=["0xpro"], sleep=0) == 0
    v = market_flow.pro_net("BTC/USDT", pd.Series([T0 + pd.Timedelta(hours=12), t1, t1 + pd.Timedelta(hours=1),
                                                   t2 + pd.Timedelta(hours=1)]), ud)
    assert np.isnan(v[0])                  # vor 24 h Aufzeichnung: unbekannt
    assert v[1] == 0.0                     # Eröffnung genau zum Kerzenbeginn ist noch nicht bekannt
    assert v[2] == 1.0 and v[3] == 0.0     # 1 Long, dann 1 Long + 1 Short


def test_pro_trades_stop_early_when_blocked(tmp_path):
    calls = []

    def post(u, b):
        calls.append(b["user"])
        raise RuntimeError("429 Too Many Requests")

    assert market_flow.update_pro_trades(["BTC/USDT"], tmp_path, post=post, wallets=["a", "b", "c"], sleep=0) == 0
    assert calls == ["a", "a"]
