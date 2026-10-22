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
