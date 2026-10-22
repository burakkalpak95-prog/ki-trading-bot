"""Phase 6: Dashboard-Datenzugriff und App laufen ohne Fehler – mit leeren und mit gefüllten Daten."""

from pathlib import Path

import pandas as pd
import pytest

from kitb import killswitch, paths
from kitb.dashboard import data as D
from kitb.data.synthetic import write_synthetic_dataset
from kitb.registry import Registry

APP = Path(D.__file__).with_name("app.py")


@pytest.fixture
def home(tmp_path, monkeypatch):
    ud = tmp_path / "user_data"
    ud.mkdir()
    monkeypatch.setattr(paths, "USER_DATA", ud)
    monkeypatch.setattr(paths, "REGISTRY_DB", ud / "kitb.sqlite")
    return ud


def _run_app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_app_runs_without_any_data(home):
    at = _run_app()
    assert any("Live-Trading" in m.label for m in at.metric)


def test_app_runs_with_data_and_kill_switch(home):
    write_synthetic_dataset(home / "data" / "synthetic", ["BTC/USDT", "ETH/USDT"], ["1h", "1d"],
                            "2024-01-01", "2024-03-01", seed=1)
    Registry(paths.REGISTRY_DB).log_source_event("binance", "probe", "failed", {"error": "403"})
    killswitch.activate("Test", "Burak")
    at = _run_app()
    assert any("NOT-STOPP AKTIV" in e.value for e in at.error)
    assert not D.errors().empty


def test_price_status_flags_stale():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC"), "close": [1.0, 2, 3]})
    fresh = D.price_status(df, "1h", now=pd.Timestamp("2024-01-01 03:30", tz="UTC"))
    stale = D.price_status(df, "1h", now=pd.Timestamp("2024-01-02", tz="UTC"))
    assert not fresh["stale"] and stale["stale"] and fresh["last"] == 3.0


def test_registry_table_whitelist(home):
    with pytest.raises(ValueError):
        D.registry_table("sqlite_master")
    assert D.registry_table("decisions").empty  # DB fehlt -> leer, kein Absturz


def test_kill_switch_reset_requires_human(home):
    killswitch.activate("x", "Burak")
    from kitb.models import GovernanceError

    with pytest.raises(GovernanceError):
        killswitch.reset("system")
    killswitch.reset("Burak")
    assert not killswitch.is_active()


def _fake_daily(price_above: bool):
    def fetch(exchange_id, pair, limit=210):
        dates = pd.date_range("2026-01-01", periods=210, freq="D", tz="UTC")
        close = [100.0] * 209 + [120.0 if price_above else 80.0]
        return pd.DataFrame({"date": dates, "close": close})
    return fetch


def test_trend_view_uses_only_closed_candles():
    df = _fake_daily(True)("binance", "BTC/USDT")
    v = D.trend_view(df)
    assert v["sma"] == 100.0 and v["above_now"] is True and v["above_at_last_close"] is False


def test_overview_shows_positions_and_plain_text(home):
    import sqlite3

    db = home / "paper" / "trend_d1" / "tradesv3.dryrun.sqlite"
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE trades (id INTEGER, pair TEXT, is_open INTEGER, open_date TEXT, close_date TEXT, "
                "open_rate REAL, close_rate REAL, amount REAL, stake_amount REAL, close_profit REAL, "
                "close_profit_abs REAL, realized_profit REAL, exit_reason TEXT, fee_open REAL, fee_close REAL)")
    con.execute("INSERT INTO trades VALUES (1,'BTC/USDT',1,'2026-10-22 22:12',NULL,100,NULL,10,1000,NULL,NULL,NULL,"
                "NULL,0.001,NULL)")
    con.commit()
    con.close()
    (db.parent / "freqtrade.log").write_text("heartbeat\n")
    o = D.overview("trend_d1", 10000, fetch=_fake_daily(True))
    assert o["running"] is True and len(o["positions"]) == 1
    assert o["positions"][0]["Gewinn/Verlust (USDT)"] == 200.0 and o["equity"] == 10200.0
    assert "bleibt investiert" in D.explain_market("BTC/USDT", o["markets"]["BTC/USDT"], True)
    assert "kauft" in D.explain_market("ETH/USDT", o["markets"]["ETH/USDT"], False)


def test_overview_survives_without_internet(home):
    def broken(*a, **k):
        raise OSError("offline")
    o = D.overview("trend_d1", 10000, fetch=broken)
    assert o["running"] is False and o["fetch_error"] and o["positions"] == []
    assert "nicht abrufbar" in D.explain_market("BTC/USDT", o["markets"]["BTC/USDT"], False)


def test_performance_compares_bot_with_holding():
    dates = pd.date_range("2026-10-01", periods=10, freq="D", tz="UTC")
    daily = {"BTC/USDT": pd.DataFrame({"date": dates, "close": [100.0] * 5 + [110.0] * 5}),
             "ETH/USDT": pd.DataFrame({"date": dates, "close": [100.0] * 10})}
    trades = pd.DataFrame([{"pair": "BTC/USDT", "is_open": 0, "open_date": "2026-10-01 00:00",
                            "close_date": "2026-10-07 00:00", "open_rate": 100.0, "close_rate": 110.0, "amount": 10.0,
                            "stake_amount": 1000.0, "close_profit_abs": 100.0, "exit_reason": "exit_signal"}])
    p = D.performance(trades, daily, 10000)
    assert p["bot_return"] == 0.01 and p["hold_return"] == 0.05
    assert "zu früh" in p["verdict"] and p["winrate"] == 1.0
    t = D.trades_table(trades, {})
    assert t.iloc[0]["Ergebnis (USDT)"] == 100.0 and t.iloc[0]["Grund"] == "unter 200-Tage-Linie"
