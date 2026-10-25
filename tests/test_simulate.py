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


def test_ki2_learns_from_weighted_simulations(tmp_path, monkeypatch):
    from kitb import reward_learner

    sim = simulate.simulate_pair(_candles(900), "ZZZ/USDT", 0.0034, np.random.default_rng(1), tmp_path)
    monkeypatch.setattr(simulate, "load", lambda user_data=None, n=0: sim)
    monkeypatch.setattr(simulate, "weight", lambda user_data=None: 0.2)
    mem = reward_learner.with_simulations(pd.DataFrame())
    assert len(mem) == len(sim) and np.isclose(mem["weight"].sum(), 0.2 * 100)  # zusammen: Anteil × max(echt, 100)
    m = reward_learner.fit(mem)
    assert m.n_trades == len(sim) and set(m.arms) <= set(reward_learner.ARMS) and m.arms


def test_simulations_do_not_drown_real_trades():
    """24.10. live: 60.000 Sim-Trades mit festem Gewicht je Trade übertönten ~400 echte. Jetzt zählen sie ZUSAMMEN nur
    Anteil × echte Trades."""
    from kitb import self_learning as sl

    rng = np.random.default_rng(0)
    n_real, n_sim = 400, 60_000
    sim = np.r_[np.ones(n_sim, bool), np.zeros(n_real, bool)]
    t = pd.DataFrame({"kind": np.where(sim, "sim", "signal"), "own": False})
    w = sl._weights(t)
    assert np.isclose(w[sim].sum(), sl.SIM_SHARE * n_real) and (w[~sim] == 1.0).all()
    x = pd.DataFrame({"a": rng.normal(size=len(sim))})
    sl._fit(x, rng.integers(0, 2, len(sim)), sim=sim, sim_share=0.1)  # läuft mit Anteil statt Gewicht je Trade


def test_diagnose_runs(tmp_path, monkeypatch):
    sim = simulate.simulate_pair(_candles(900), "ZZZ/USDT", 0.0034, np.random.default_rng(1), tmp_path)
    monkeypatch.setattr(simulate, "load", lambda user_data=None, n=0: sim)
    rng = np.random.default_rng(2)
    n = 200
    real = pd.DataFrame({"ki": "ki_trader", "pair": "ZZZ/USDT", "profit_pct": rng.normal(0, 0.01, n),
                         "open_date": pd.date_range("2024-03-10", periods=n, freq="h", tz="UTC").astype(str),
                         "close_date": pd.date_range("2024-03-10 02:00", periods=n, freq="h", tz="UTC").astype(str),
                         "ret_1h": rng.normal(0, 0.01, n), "rsi_14": rng.uniform(20, 80, n)})
    monkeypatch.setattr("kitb.shared_memory.load", lambda ud=None: real)
    res = simulate.diagnose(tmp_path)
    assert res["echte_trades"] == n and "auc_nur_sims_auf_echten" in res
    assert (tmp_path / "reports" / "sim-diagnose.json").exists()


def test_simulation_pauses_when_nothing_useful_and_resumes(tmp_path, monkeypatch):
    """Burak 25.10.: kann die KI aus der Simulation nichts Brauchbares mehr gewinnen -> Feedback + Stopp."""
    import json

    d = simulate.sim_dir(tmp_path)
    d.mkdir(parents=True)
    assert simulate.saturated(d) == (False, "")
    (d / "sim_evals.jsonl").write_text("".join(json.dumps({"hilft": False, "auc_ohne": 0.76, "auc_mit": 0.44}) + "\n"
                                               for _ in range(3)), encoding="utf-8")
    rounds = [{"Ø genommen": -0.003, "Ø zufällig": -0.003}] * 200
    (d / "sim_rounds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rounds), encoding="utf-8")
    stop, why = simulate.saturated(d)
    assert stop and "nicht" in why
    better = [{"Ø genommen": -0.003, "Ø zufällig": -0.003}] * 100 + [{"Ø genommen": 0.002, "Ø zufällig": -0.003}] * 100
    (d / "sim_rounds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in better), encoding="utf-8")
    assert simulate.saturated(d)[0] is False  # wird in der Simulation noch besser -> weiter
    # pausiert: läuft erst nach genug neuen echten Trades und nur, wenn die Prüfung Nutzen zeigt, wieder an
    simulate._set_pause(d, 100, "Test")
    real = pd.DataFrame({"profit_pct": [0.0] * 120})
    monkeypatch.setattr("kitb.shared_memory.load", lambda ud=None: real)
    simulate.loop(tmp_path, rounds=1)
    assert simulate._pause_state(d)["echte_trades"] == 100  # zu wenige neue -> bleibt pausiert, nichts gespielt
    real = pd.DataFrame({"profit_pct": [0.0] * 200})
    monkeypatch.setattr("kitb.shared_memory.load", lambda ud=None: real)
    monkeypatch.setattr(simulate, "evaluate", lambda ud=None: {"hilft": False})
    simulate.loop(tmp_path, rounds=1)
    assert simulate._pause_state(d)["echte_trades"] == 200 and "weiterhin nicht" in simulate._pause_state(d)["grund"]
