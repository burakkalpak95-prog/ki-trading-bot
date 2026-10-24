"""Simulierte Trades: Ergebnisse je Haltedauer/Stoploss korrekt, kein Blick in die Zukunft, immer verwertet."""

import numpy as np
import pandas as pd

from kitb import self_learning as sl
from kitb import simulate


def _candles(n=600, seed=2):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"), "open": o,
                         "high": np.maximum(o, c) * 1.003, "low": np.minimum(o, c) * 0.997, "close": c,
                         "volume": 1000.0})


def test_outcomes_hold_and_stop():
    df = pd.DataFrame({"open": [100, 100, 100, 100.0], "low": [99, 90, 99, 99.0], "close": [101, 95, 102, 103.0]})
    r = simulate.outcomes(df, cost=0.0)
    assert abs(r[(1, 2)][0] - 0.01) < 1e-12  # 1 h halten: Schluss 101
    assert abs(r[(2, 2)][0] + 0.02) < 1e-12  # Kerze 2 fällt auf 90 -> bei −2 % ausgestoppt
    assert abs(r[(2, 6)][0] + 0.06) < 1e-12 and np.isnan(r[(8, 2)][0])  # 8 h: nicht genug Kerzen


def test_simulated_features_come_from_previous_candle(tmp_path):
    df = _candles()
    sim = simulate.simulate_pair(df, "ZZZ/USDT", 0.0034, np.random.default_rng(1), tmp_path)
    i = 300
    row = sim[sim["open_date"] == str(df["date"][i])].iloc[0]
    assert abs(row["ret_1h"] - (df["close"][i - 1] / df["close"][i - 2] - 1)) < 1e-12
    assert set(sim["kind"]) == {"sim"} and sim["reason"].str.contains("stop:").all()


def test_simulations_always_used_and_weight_chosen(tmp_path, monkeypatch):
    df = _candles(1500)
    sim = simulate.simulate_pair(df, "ZZZ/USDT", 0.0034, np.random.default_rng(1), tmp_path)
    sim_dir = simulate.sim_dir(tmp_path)
    sim_dir.mkdir(parents=True, exist_ok=True)
    sim.to_parquet(sim_dir / "simulated_trades.parquet", index=False)
    rng = np.random.default_rng(5)
    n = 200
    real = pd.DataFrame({"ki": "ki_trader", "kind": "signal", "own": True, "reason": "signal",
                         "open_date": pd.date_range("2024-03-10", periods=n, freq="h", tz="UTC").astype(str),
                         "close_date": pd.date_range("2024-03-10 02:00", periods=n, freq="h", tz="UTC").astype(str),
                         "profit_pct": rng.normal(0, 0.01, n), "hold_hours": 2.0,
                         "ret_1h": rng.normal(0, 0.01, n), "rsi_14": rng.uniform(20, 80, n)})
    monkeypatch.setattr(sl, "own_trades", lambda ki, ud=None: real)
    monkeypatch.setattr("kitb.shared_memory.load", lambda ud=None: real)
    res = simulate.evaluate(tmp_path)
    assert res["weight"] in simulate.WEIGHTS and simulate.weight(tmp_path) == res["weight"]
    sl._cache.clear()
    sl._SIM.clear()
    st = sl.state("ki_trader", tmp_path)
    assert st["simulated"] > 0 and (st["_trades"]["kind"] == "sim").any()


def test_stock_simulation_from_fred_closes():
    days = pd.date_range("2015-01-01", periods=800, freq="B")
    closes = 2000 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.01, len(days))))
    csv = "observation_date,SP500\n" + "\n".join(f"{d:%Y-%m-%d},{c:.2f}" for d, c in zip(days, closes))
    sim = simulate.simulate_stocks(np.random.default_rng(1), get_text=lambda url, params: csv)
    assert len(sim) > 500 and set(sim["ki"]) == {"simulation_aktie"}
    assert sim["hold_hours"].isin([24, 48, 96, 192]).all() and sim["reason"].str.startswith("sim_aktie").all()


def test_build_writes_file_without_network(tmp_path, monkeypatch):
    from kitb.data.storage import store_ohlcv

    store_ohlcv(tmp_path / "data", "ZZZ/USDT", "1h", _candles(800))
    res = simulate.build(pairs=["ZZZ/USDT"], datadir=tmp_path / "data", user_data=tmp_path, stocks=False, get=False)
    assert res["krypto"] > 400 and (simulate.sim_dir(tmp_path) / "simulated_trades.parquet").exists()
    assert len(simulate.load(tmp_path)) > 0


def test_self_play_rounds_learn_and_grow_memory(tmp_path, monkeypatch):
    from kitb.data.storage import store_ohlcv

    for i, pair in enumerate(["AAA/USDT", "BBB/USDT"]):
        store_ohlcv(tmp_path / "data", pair, "1h", _candles(900, seed=i))
    monkeypatch.setattr(simulate.paths, "DATA_DIR", tmp_path / "data_root")
    (tmp_path / "data_root").mkdir()
    (tmp_path / "data_root" / "binance").symlink_to(tmp_path / "data")
    orig = simulate.build
    monkeypatch.setattr(simulate, "build", lambda user_data=None: orig(pairs=["AAA/USDT", "BBB/USDT"],
                                                                       user_data=user_data, stocks=False, get=False))
    monkeypatch.setattr("kitb.shared_memory.load", lambda ud=None: pd.DataFrame())
    simulate.loop(tmp_path, rounds=3, pause=0)
    d = simulate.sim_dir(tmp_path)
    mem = pd.read_parquet(d / "simulated_trades.parquet")
    lines = (d / "sim_rounds.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and mem["round"].max() == 3 and (mem["round"] == 3).sum() > 0
    assert mem.loc[mem["round"] == 3, "reason"].str.contains("runde 3").all()
    st = __import__("json").loads((d / "sim_status.json").read_text(encoding="utf-8"))
    assert st["runden"] == 3 and st["sim_trades_gespielt_gesamt"] > 0 and st["im_gedaechtnis"] == len(mem)
