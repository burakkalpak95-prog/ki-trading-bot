"""Zwei KIs: gemeinsames Gedächtnis (ein Schreiber, idempotent, Neustart-fest) und Belohnungslerner."""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from kitb import paths, reward_learner, shared_memory


@pytest.fixture
def ud(tmp_path, monkeypatch):
    u = tmp_path / "user_data"
    u.mkdir()
    monkeypatch.setattr(paths, "USER_DATA", u)
    monkeypatch.setattr(paths, "REGISTRY_DB", u / "kitb.sqlite")
    return u


def _ft_db(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE trades (id INTEGER, pair TEXT, is_open INT, open_date TEXT, close_date TEXT, "
                "open_rate REAL, close_rate REAL, stake_amount REAL, close_profit REAL, close_profit_abs REAL, "
                "enter_tag TEXT, exit_reason TEXT)")
    con.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


def test_shared_memory_merges_both_ais_once_and_survives_restart(ud):
    feats = {f: 0.1 for f in shared_memory.SNAPSHOT_FEATURES}
    shared_memory.snapshot("ki_trader", pair="SOL/USDT", when="2026-10-23 01:00:00+00:00", kind="signal",
                           reason="signal: Modell 61 %", features=feats, timeframe="1h", user_data=ud)
    shared_memory.snapshot("ki_reward", pair="ADA/USDT", when="2026-10-23 02:00:00+00:00", kind="learn",
                           reason="lern: Zufallsversuch, halten 4h", features=feats, action="4h", timeframe="15m",
                           user_data=ud)
    _ft_db(ud / "paper" / "ki_trader" / "tradesv3.dryrun.sqlite", [
        (1, "SOL/USDT", 0, "2026-10-23 01:00:05", "2026-10-23 05:00:00", 100, 102, 1000, 0.018, 18, "signal: Modell 61 %", "x"),
        (2, "SOL/USDT", 1, "2026-10-23 06:00:00", None, 100, None, 1000, None, None, "signal", None)])
    _ft_db(ud / "paper" / "ki_reward" / "tradesv3.dryrun.sqlite", [
        (1, "ADA/USDT", 0, "2026-10-23 02:00:10", "2026-10-23 06:00:00", 1, 0.97, 1000, -0.032, -32, "lern: Zufallsversuch, halten 4h", "haltedauer 4h")])
    assert shared_memory.sync(ud) == 2
    assert shared_memory.sync(ud) == 0  # erneuter Lauf / Neustart: keine Doppel
    mem = shared_memory.load(ud)
    assert set(mem["ki"]) == {"ki_trader", "ki_reward"} and set(mem["kind"]) == {"signal", "learn"}
    assert mem.loc[mem["ki"] == "ki_reward", "action"].iloc[0] == "4h" and mem["rsi_14"].notna().all()


def test_reward_learner_learns_from_rewards_and_explores():
    rng = np.random.default_rng(1)
    rows = []
    for _ in range(600):
        f = {c: rng.normal() for c in shared_memory.SNAPSHOT_FEATURES}
        arm = str(rng.choice(list(reward_learner.ARMS)))
        good = f["ret_24h"] > 0
        reward = (0.01 if good else -0.01) if arm == "12h" else -0.002
        rows.append({**f, "profit_pct": reward + rng.normal(0, 0.002), "hold_hours": reward_learner.ARMS[arm],
                     "action": arm})
    m = reward_learner.fit(pd.DataFrame(rows))
    up = {c: 0.0 for c in shared_memory.SNAPSHOT_FEATURES} | {"ret_24h": 2.0}
    down = {c: 0.0 for c in shared_memory.SNAPSHOT_FEATURES} | {"ret_24h": -2.0}
    assert max(m.expected(up), key=m.expected(up).get) == "12h" and m.expected(down)["12h"] < 0
    picks = [reward_learner.choose(m, up, np.random.default_rng(i), epsilon=0.0) for i in range(50)]
    assert sum(p["arm"] == "12h" and p["kind"] == "signal" for p in picks) > 40
    assert all(p["arm"] is None for p in
               [reward_learner.choose(m, down, np.random.default_rng(i), epsilon=0.0) for i in range(20)])
    rnd = [reward_learner.choose(m, down, np.random.default_rng(i), epsilon=1.0) for i in range(20)]
    assert all(p["kind"] == "learn" and p["reason"].startswith("lern") for p in rnd)


def test_empty_memory_still_trades_to_learn():
    m = reward_learner.fit(pd.DataFrame())
    feats = {c: 0.0 for c in shared_memory.SNAPSHOT_FEATURES}
    picks = [reward_learner.choose(m, feats, np.random.default_rng(i)) for i in range(40)]
    assert any(p["arm"] for p in picks)  # ohne Wissen probiert sie aus statt zu warten


def test_ki_reward_paper_config(tmp_path):
    from kitb import paper, strategy_approval
    from kitb.registry import Registry

    reg = Registry(tmp_path / "r.sqlite")
    strategy_approval.approve(reg, "ki_reward", by="Burak", reason="KI 2")
    _, cfg = paper.build_paper_config("ki_reward", reg, tmp_path / "ud")
    assert cfg["kitb_mode"] == "learning" and cfg["dry_run"] is True and cfg["kitb_practice"]["explore_share"] > 0
    assert len(cfg["exchange"]["pair_whitelist"]) == 8 and not cfg["exchange"].get("key")


def test_fleet_approval_includes_both_ais(ud, capsys):
    from kitb import cli

    assert cli.main(["strategies", "approve", "--id", "fleet", "--by", "Burak", "--reason", "zwei KIs"]) in (0, None)
    out = capsys.readouterr().out
    assert "für ki_trader." in out and "für ki_reward." in out and "für donchian_15m." in out
    assert "für trend_d1." not in out  # Kandidat bleibt eigene Freigabe


def test_learning_ais_get_six_slots_within_exposure_limit(tmp_path):
    from kitb import paper, strategy_approval
    from kitb.registry import Registry
    from kitb.risk import RiskLimits

    reg = Registry(tmp_path / "r.sqlite")
    lim = RiskLimits.load()
    for bot in ("ki_reward", "ki_trader", "trend_d1", "donchian_15m"):
        strategy_approval.approve(reg, bot, by="Burak", reason="t")
    for bot in ("ki_reward", "ki_trader"):
        _, cfg = paper.build_paper_config(bot, reg, tmp_path / "ud")
        assert cfg["max_open_trades"] == 6 == cfg["kitb_max_open_positions"]
        assert cfg["stake_amount"] * 6 <= cfg["dry_run_wallet"] * lim.max_total_exposure_fraction
    for bot in ("trend_d1", "donchian_15m"):
        _, cfg = paper.build_paper_config(bot, reg, tmp_path / "ud")
        assert cfg["max_open_trades"] == lim.max_open_positions and "kitb_max_open_positions" not in cfg


def test_tests_never_touch_real_user_data():
    from pathlib import Path

    from kitb import killswitch

    real = Path(__file__).resolve().parents[1] / "user_data"
    assert paths.USER_DATA != real and killswitch.path() != real / "KILL_SWITCH"


def test_explore_draw_is_well_mixed_per_pair():
    from kitb.ml_strategy import explore_draw

    dates = pd.Series(pd.date_range("2026-01-01", periods=2000, freq="h", tz="UTC"))
    for pair in ("BTC/USDT", "ADA/USDT", "LINK/USDT"):
        picked = (explore_draw(dates, pair) < 400).to_numpy()
        assert 0.33 < picked.mean() < 0.47
        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, picked.astype(int), 0]) != 0))
        assert runs.max() < 20  # keine tagelangen Blöcke mit/ohne Lern-Trades


def test_ki_reward_custom_exit_is_active():
    import importlib.util

    spec = importlib.util.spec_from_file_location("KiReward", paths.USER_DATA / "strategies" / "KiReward.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.KiReward.use_exit_signal is True  # sonst ruft Freqtrade custom_exit nie auf (Haltedauer)


def test_sync_without_snapshot_file_keeps_result(ud):
    _ft_db(ud / "paper" / "bot_b" / "tradesv3.dryrun.sqlite", [
        (1, "BTC/USDT", 0, "2026-10-23 01:00:00", "2026-10-23 03:00:00", 100, 101, 1000, 0.008, 8, "ml_up", "x")])
    assert shared_memory.sync(ud) == 1
    mem = shared_memory.load(ud)
    assert mem["ki"].tolist() == ["bot_b"] and mem["kind"].tolist() == ["signal"] and mem["rsi_14"].isna().all()
