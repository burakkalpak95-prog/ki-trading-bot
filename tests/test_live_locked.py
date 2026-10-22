"""Phase 8: Live-Trading ist nicht aktivierbar – weder per Konfiguration noch per Umgebungsvariable."""

import copy

import pytest

from kitb import live, paths
from kitb.config import load_freqtrade_config, load_project_config
from kitb.live import LiveTradingLocked, OrderRequest
from kitb.offline import STATIC_SPOT_MARKETS, _market
from kitb.safety import SafetyError, assert_run_mode_allowed, check_freqtrade_config


def test_adapter_refuses_everything():
    adapter = live.get_adapter()
    order = OrderRequest("BTC/USDT", "buy", 0.01, None, "id-1")
    for call in (lambda: adapter.place_order(order), lambda: adapter.cancel_order("1", "BTC/USDT"),
                 adapter.open_orders, adapter.balances, adapter.reconcile):
        with pytest.raises(LiveTradingLocked):
            call()


@pytest.mark.parametrize("env", ["KITB_ALLOW_LIVE", "KITB_LIVE", "LIVE_TRADING", "FREQTRADE__DRY_RUN"])
def test_env_vars_cannot_unlock(monkeypatch, env):
    monkeypatch.setenv(env, "1")
    with pytest.raises(SafetyError):
        assert_run_mode_allowed("live")
    assert isinstance(live.get_adapter(), live.LockedLiveAdapter)


def test_config_cannot_unlock():
    cfg = copy.deepcopy(load_freqtrade_config())
    cfg["dry_run"] = False
    p = load_project_config()
    assert check_freqtrade_config(cfg, p["allowed_exchanges"], p["allowed_pairs"])
    with pytest.raises(SafetyError):
        assert_run_mode_allowed("live")


def test_min_order_check():
    m = _market("BTC/USDT", STATIC_SPOT_MARKETS["BTC/USDT"])
    assert live.check_min_order(m, 0.001, 30000) == []
    assert live.check_min_order(m, 0.000001, 30000)  # zu klein
    assert live.check_min_order({**m, "spot": False}, 1, 30000)


def test_risk_confirmation_tracks_changes(tmp_path, monkeypatch):
    assert not live.risk_limits_confirmed(tmp_path)
    live.confirm_risk_limits("Burak", tmp_path)
    assert live.risk_limits_confirmed(tmp_path)
    monkeypatch.setattr(live, "_risk_hash", lambda: "geändert")
    assert not live.risk_limits_confirmed(tmp_path)


def test_readiness_never_reports_live_possible(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "REGISTRY_DB", tmp_path / "r.sqlite")
    r = live.readiness(tmp_path)
    assert r["live_trading_possible"] is False
    assert r["checks"]["Live-Modus im Code freigeschaltet"] is False


def test_cli_has_no_live_trade_command():
    from kitb.cli import build_parser

    sub = next(a for a in build_parser()._actions if a.dest == "command")
    live_parser = sub.choices["live"]
    actions = next(a for a in live_parser._actions if a.dest == "action")
    assert set(actions.choices) == {"readiness", "confirm-risk"}
