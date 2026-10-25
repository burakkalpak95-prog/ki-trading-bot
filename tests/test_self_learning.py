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


def test_design_tolerates_infinity():
    """ai-simulation stürzte 24.10. ab: flow_oi_change_4h war ±inf (OI 0). inf muss wie fehlend behandelt werden."""
    import numpy as np
    import pandas as pd

    from kitb import self_learning

    df = pd.DataFrame({"a": [1.0, np.inf, -np.inf, 2.0]})
    x, _ = self_learning._design(df, ["a"])
    assert np.isfinite(x.to_numpy()).all()


def test_relevance_prefers_simulations_like_real_trades():
    """Burak 25.10.: die KI nutzt nur, was sie selbst als brauchbar einstuft – Simulationen aus Marktlagen, die bei
    echten Trades nie vorkommen, zählen kaum."""
    import numpy as np
    import pandas as pd

    from kitb import self_learning

    rng = np.random.default_rng(0)
    real = pd.DataFrame({"rsi_14": rng.normal(50, 5, 200), "ret_1h": rng.normal(0, 0.01, 200), "profit_pct": 0.0})
    near = pd.DataFrame({"rsi_14": rng.normal(50, 5, 500), "ret_1h": rng.normal(0, 0.01, 500)})
    far = pd.DataFrame({"rsi_14": rng.normal(85, 3, 500), "ret_1h": rng.normal(0, 0.01, 500)})
    rel = self_learning.relevance(pd.concat([near, far], ignore_index=True), real)
    assert rel[:500].mean() > 5 * rel[500:].mean() and np.isclose(rel.mean(), 1.0)
    assert self_learning.effective(rel) < 1000


def test_untagged_trades_still_exit_after_longest_hold():
    """25.10.: Bot A/B (kein Lernhorizont) hielten Trades von vor dem Selbstlernen endlos – ml_weak ist aus."""
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace

    from kitb.ml_strategy import KitbFreqaiBase

    s = KitbFreqaiBase.__new__(KitbFreqaiBase)
    s.timeframe = "1h"
    s._kitb = lambda: {"self_learning": True, "hold_options": [4, 8, 16, 32]}
    now = datetime(2026, 10, 25, tzinfo=timezone.utc)
    old = SimpleNamespace(enter_tag="signal: Modell 60 %", open_date_utc=now - timedelta(hours=33))
    young = SimpleNamespace(enter_tag="signal: Modell 60 %", open_date_utc=now - timedelta(hours=5))
    assert s.custom_exit("BTC/USDT", old, now, 1.0, 0.0) == "lernhorizont"
    assert s.custom_exit("BTC/USDT", young, now, 1.0, 0.0) is None


def test_unstable_features_are_dropped_by_the_ai_itself():
    """Punkt 7 (Burak 25.10.): Merkmale, deren Zusammenhang mit Gewinn je Jahr die Richtung wechselt, fliegen raus."""
    import numpy as np
    import pandas as pd

    from kitb import self_learning

    rng = np.random.default_rng(1)
    rows = []
    for i, year in enumerate([2022, 2023, 2024, 2025]):
        n = 1000
        good = rng.normal(size=n)
        flip = rng.normal(size=n)
        p = 0.5 + 0.2 * np.tanh(good) + 0.2 * np.tanh(flip) * (1 if i % 2 else -1)
        rows.append(pd.DataFrame({"close_date": f"{year}-06-01T00:00:00+00:00", "good": good, "flip": flip,
                                  "profit_pct": np.where(rng.random(n) < p, 0.01, -0.01)}))
    sim = pd.concat(rows, ignore_index=True)
    assert self_learning.stable_columns(sim, ["good", "flip", "fehlt"]) == ["good", "fehlt"]


def test_net_goal_takes_only_trades_expected_above_fees(monkeypatch):
    """Test 25.10.: Signal-Trades nur bei erwartetem Gewinn NACH Gebühren (einzige Idee früh UND spät im Plus)."""
    rng = np.random.default_rng(5)
    n = 400
    x = rng.normal(0, 1, n)
    t = pd.DataFrame({"close_date": pd.date_range("2026-10-01", periods=n, freq="h").astype(str),
                      "profit_pct": 0.01 * x - 0.006 + rng.normal(0, 0.003, n), "rsi_14": x,
                      "ret_1h": rng.normal(0, 1, n), "reason": "signal"})
    s = sl.train(t)
    assert s["net_active"] and s["check_net"] > s["check_all"]
    monkeypatch.setattr(sl, "own_trades", lambda ki, ud=None: t)
    sl._cache.clear()
    ok_hi, _, why = sl.win_chance("ema_cross_15m", {"rsi_14": 2.0, "ret_1h": 0.0}, sims=False)
    ok_lo, p_lo, _ = sl.win_chance("ema_cross_15m", {"rsi_14": 0.5, "ret_1h": 0.0}, sims=False)
    assert ok_hi and "erwartet" in why
    assert not ok_lo and p_lo > s["threshold"]  # die Gewinnchance allein hätte den Verlust-Trade erlaubt


def test_net_goal_stays_off_without_edge():
    s = sl.train(_trades(300, signal=False))
    assert not s.get("net_active") and s.get("net_model") is None


def test_new_sim_weight_or_content_refits(monkeypatch):
    """Prüfbericht 25.10., Punkt 3: gleiche Zeilenzahl, aber neues Gewicht bzw. neue Sims -> neu anpassen."""
    from kitb import simulate

    t = _trades(300)
    monkeypatch.setattr(sl, "own_trades", lambda ki, ud=None: t)
    calls = []
    real_train = sl.train
    monkeypatch.setattr(sl, "train", lambda d: calls.append(1) or real_train(d))
    sl._cache.clear()
    monkeypatch.setattr(simulate, "weight", lambda ud=None: 0.5)
    sl.state("ki_trader", sims=False)
    sl.state("ki_trader", sims=False)
    assert len(calls) == 1
    monkeypatch.setattr(simulate, "weight", lambda ud=None: 1.0)
    sl.state("ki_trader", sims=False)
    assert len(calls) == 2
