"""KI lernt aus eigenen Trades: Filter nur mit Nachweis, Haltedauer nach eigenen Ergebnissen, Tag-Parsing."""

import numpy as np
import pandas as pd

from kitb import self_learning as sl


def _trades(n=200, signal=True, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    profit = (0.01 * x if signal else 0) + rng.normal(0, 0.005, n)
    return pd.DataFrame({"close_date": pd.date_range("2026-10-01", periods=n, freq="h").astype(str),
                         "profit_pct": profit, "rsi_14": x, "ret_1h": rng.normal(0, 1, n), "reason": "signal"})


def test_filter_only_active_with_proven_edge():
    assert not sl.train(_trades(20))["active"]  # zu wenige Trades
    s = sl.train(_trades(300, signal=True))
    assert s["active"] and s["check_filtered"] > s["check_all"]


def test_win_chance_uses_own_results(monkeypatch):
    t = _trades(300)
    monkeypatch.setattr(sl, "own_trades", lambda ki, ud=None: t)
    sl._cache.clear()
    ok_good, p_good, _ = sl.win_chance("ki_trader", {"rsi_14": 2.0, "ret_1h": 0.0})
    ok_bad, p_bad, _ = sl.win_chance("ki_trader", {"rsi_14": -2.0, "ret_1h": 0.0})
    assert p_good > 0.5 > p_bad and ok_good and not ok_bad


def test_hold_choice_prefers_better_option_and_parses_tag():
    rows = [{"reason": "signal · halten:4", "profit_pct": -0.01}] * 30 + \
           [{"reason": "signal · halten:16 · Bots: keiner", "profit_pct": 0.01}] * 30
    t = pd.DataFrame(rows)
    picks = [sl.choose_hold(t, [4, 8, 16, 32], seed) for seed in range(200)]
    assert picks.count(16) > picks.count(4) and picks.count(4) < 10
    assert sl.hold_from_tag("signal: Modell 61 % · halten:16 · Bots: EMA") == 16 and sl.hold_from_tag("x") is None
    st = sl.hold_stats(t, [4, 16])
    assert list(st["n"]) == [30, 30]


def test_fill_features_for_rule_bot_trades_without_lookahead(tmp_path):
    import json
    import sqlite3

    from kitb import shared_memory

    hours = pd.date_range("2026-09-01", periods=1000, freq="h", tz="UTC")
    close = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, len(hours))))

    def get(url, params):
        return [[int(d.timestamp() * 1000), str(c), str(c * 1.01), str(c * 0.99), str(c), "1000"]
                for d, c in zip(hours, close)]

    p = shared_memory.db_path(tmp_path)
    p.parent.mkdir(parents=True)
    with sqlite3.connect(p) as con:
        con.execute(shared_memory._SCHEMA)
        opened = hours[600] + pd.Timedelta(minutes=30)
        con.execute("INSERT INTO trades (ki, round, trade_id, kind, pair, open_date, close_date, profit_pct, hold_hours, "
                    "reason) VALUES ('macd_rsi_1h', 'R1', 1, 'signal', 'ZZZ/USDT', ?, ?, 0.01, 2.0, 'macd')",
                    (opened.isoformat(), (opened + pd.Timedelta(hours=2)).isoformat()))
    shared_memory._KLINES.clear()
    assert shared_memory.fill_features(tmp_path, get=get) == 1
    with sqlite3.connect(p) as con:
        f = json.loads(con.execute("SELECT features_1h FROM trades").fetchone()[0])
    # Einstieg 00:30 nach Kerze 600 -> letzte abgeschlossene Kerze ist 599 (Schluss = Beginn von 600)
    assert abs(f["ret_1h"] - (close[599] / close[598] - 1)) < 1e-12
    mem = shared_memory.load(tmp_path)
    assert mem.loc[0, "ret_1h"] == f["ret_1h"] and shared_memory.fill_features(tmp_path, get=get) == 0


def test_stop_choice_learns_and_never_looser_than_limit():
    rows = [{"reason": "signal · halten:4 · stop:2", "profit_pct": -0.02}] * 30 + \
           [{"reason": "signal · halten:4 · stop:6", "profit_pct": 0.01}] * 30
    t = pd.DataFrame(rows)
    picks = [sl.choose_stop(t, seed) for seed in range(200)]
    assert picks.count(6) > picks.count(2) and set(picks) <= set(sl.STOP_OPTIONS) and max(sl.STOP_OPTIONS) <= 6
    assert sl.stop_from_tag("signal · halten:8 · stop:4 · Bots: EMA") == 4 and sl.stop_from_tag("signal") is None


def test_custom_stoploss_uses_learned_level():
    from types import SimpleNamespace

    from kitb.ml_strategy import KitbFreqaiBase

    s = KitbFreqaiBase.__new__(KitbFreqaiBase)
    trade = SimpleNamespace(enter_tag="signal · halten:4 · stop:2", leverage=1.0)
    stop = s.custom_stoploss("BTC/USDT", trade, None, 100.0, 0.0)
    assert abs(stop - 0.02) < 1e-9  # bei 0 % Gewinn: 2 % unter dem aktuellen Kurs
    assert s.custom_stoploss("BTC/USDT", SimpleNamespace(enter_tag="signal", leverage=1.0), None, 100.0, 0.0) is None


def test_threshold_is_base_rate_so_filter_does_not_block_everything(monkeypatch):
    rng = np.random.default_rng(4)
    n = 400
    x = rng.normal(0, 1, n)
    win = rng.random(n) < 0.15 + 0.2 * (x > 0)  # ~25 % Gewinner, besser bei x > 0
    t = pd.DataFrame({"close_date": pd.date_range("2026-10-01", periods=n, freq="h").astype(str),
                      "profit_pct": np.where(win, 0.02, -0.01), "rsi_14": x, "reason": "signal"})
    monkeypatch.setattr(sl, "own_trades", lambda ki, ud=None: t)
    sl._cache.clear()
    s = sl.state("bot_b")
    assert 0.15 < s["threshold"] < 0.4
    allowed = [sl.win_chance("bot_b", {"rsi_14": v})[0] for v in np.linspace(-2, 2, 21)]
    assert any(allowed) and not all(allowed) if s["active"] else all(allowed)
