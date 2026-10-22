"""KI-Gedächtnis und KI-Filter: kein Blick in die Zukunft, Prüfung auf ungesehenen Trades, Filter lehnt nur ab."""

import numpy as np
import pandas as pd
import pytest

from kitb import ai_memory, paths
from kitb.registry import Registry


@pytest.fixture
def home(tmp_path, monkeypatch):
    ud = tmp_path / "user_data"
    ud.mkdir()
    monkeypatch.setattr(paths, "USER_DATA", ud)
    monkeypatch.setattr(paths, "REGISTRY_DB", ud / "kitb.sqlite")
    return ud


def _daily(n=700, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"date": dates, "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))})


def test_market_features_use_only_past_candles():
    d = _daily()
    t = ai_memory.daily_market_table(d)
    day = d["date"].iloc[400]
    changed = d.copy()
    changed.loc[400:, "close"] *= 5  # Zukunft (ab Einstiegstag) ändern
    t2 = ai_memory.daily_market_table(changed)
    assert ai_memory.features_at(t, day) == ai_memory.features_at(t2, day)


def _history(home, n=800, seed=2):
    """Künstliche Trades: Gewinn wahrscheinlicher, wenn der Kurs über der 200-Tage-Linie lag (lernbares Muster)."""
    d = _daily(seed=seed)
    t = ai_memory.daily_market_table(d).dropna()
    rng = np.random.default_rng(seed)
    days = rng.choice(t.index, n)
    rows = []
    for day in sorted(days):
        above = t.loc[day, "dist_sma200"] > 0
        win = rng.random() < (0.9 if above else 0.1)
        rows.append({"pair": "BTC/USDT", "open_date": day + pd.Timedelta(hours=3),
                     "close_date": day + pd.Timedelta(days=1), "close_profit": 0.02 if win else -0.02,
                     "close_profit_abs": 40.0 if win else -40.0, "stake_amount": 2000.0, "exit_reason": "x",
                     "bot": "backtest", "round": "historisch", "source": "backtest",
                     "strategy": "BreakoutDonchian"})
    out = ai_memory.memory_dir(home)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / "history_trades.parquet", index=False)
    return d


def test_daily_learns_pattern_and_filter_only_rejects(home):
    d = _history(home)
    fetch = lambda ex, pair, limit=1000: d  # noqa: E731
    reg = Registry(paths.REGISTRY_DB)
    res = ai_memory.daily(registry=reg, fetch=fetch)
    assert res["trained"] and res["metrics"]["accuracy"] > res["metrics"]["naive_accuracy"]
    assert next(iter(res["importances"])) == "dist_sma200"
    prog = ai_memory.learning_progress(reg)
    assert prog["learned_pct"] == 0.0 and prog["verdict"].startswith("🟢")
    t = ai_memory.daily_market_table(d).dropna()
    up = t["dist_sma200"].idxmax()
    down = t["dist_sma200"].idxmin()
    ok_up, p_up = ai_memory.allow_entry(strategy="BreakoutDonchian", pair="BTC/USDT", when=up, fetch=fetch)
    ok_down, p_down = ai_memory.allow_entry(strategy="BreakoutDonchian", pair="BTC/USDT", when=down, fetch=fetch)
    assert p_up > p_down and ok_up and not ok_down


def test_too_little_memory_does_not_train_and_filter_allows(home):
    reg = Registry(paths.REGISTRY_DB)
    res = ai_memory.daily(registry=reg, fetch=None)
    assert res["trained"] is False
    assert ai_memory.allow_entry(strategy="X", pair="BTC/USDT", when=pd.Timestamp.now(tz="UTC"), fetch=None) == (True, None)
    assert ai_memory.learning_progress(reg)["first"] is None


def test_seed_from_backtests_on_synthetic_data(tmp_path):
    from kitb.data.synthetic import write_synthetic_dataset

    d = tmp_path / "synthetic"
    write_synthetic_dataset(d, ["BTC/USDT", "ETH/USDT"], ["1h", "1d"], "2023-01-01", "2024-06-01", seed=5)
    home = tmp_path / "ud"
    n = ai_memory.seed_from_backtests(datadir=d, data_source="synthetic", user_data=home, offline=True)
    hist = pd.read_parquet(ai_memory.memory_dir(home) / "history_trades.parquet")
    assert n == len(hist) > 50 and hist["strategy"].nunique() >= 4
