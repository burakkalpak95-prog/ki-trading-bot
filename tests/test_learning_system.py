"""Phase 4: Modellregister, Freigabe/Rollback, Entscheidungslog, Drift, Erklärungen."""

import numpy as np
import pandas as pd
import pytest

from kitb import models
from kitb.decisions import build_decisions, store_decisions
from kitb.drift import feature_drift, performance_drift, psi
from kitb.explain import parse, top_contributions
from kitb.models import GovernanceError
from kitb.registry import Registry

GOOD = {"total_trades": 80, "profit_factor": 1.4, "max_drawdown_account": 0.1, "profit_total": 0.12,
        "excess_vs_buy_and_hold": 0.05, "p_value": 0.01}


def _cand(reg, bot="bot_a", **metric_overrides):
    return models.register_candidate(reg, bot=bot, identifier=f"{bot}-x", freqaimodel="M", overlay={"freqai": {}},
                                     data_fingerprints={"BTC/USDT": "f"}, eval_timerange="20240101-20240201",
                                     metrics={**GOOD, **metric_overrides}, resources={"wall_seconds": 3})


@pytest.fixture
def reg(tmp_path):
    return Registry(tmp_path / "r.sqlite")


def test_versions_increment_per_bot(reg):
    assert models.get(reg, _cand(reg))["version"] == 1
    assert models.get(reg, _cand(reg))["version"] == 2
    assert models.get(reg, _cand(reg, bot="bot_b"))["version"] == 1


def test_criteria_pass_and_fail(reg):
    good = _cand(reg)
    assert models.evaluate_criteria(reg, good, reference_metrics={"profit_total": 0.0}, false_positive_rate=0.4)["passed"]
    bad = _cand(reg, total_trades=5, p_value=0.4)
    res = models.evaluate_criteria(reg, bad, reference_metrics={"profit_total": 0.0})
    assert not res["passed"] and not res["checks"]["min_trades"] and not res["checks"]["max_mean_profit_pvalue"]
    assert models.get(reg, bad)["status"] == "failed"


def test_missing_reference_fails_closed(reg):
    vid = _cand(reg)
    assert not models.evaluate_criteria(reg, vid, reference_metrics=None)["checks"]["beats_reference"]


@pytest.mark.parametrize("actor", ["", "system", "auto", "model", "bot_a", "agent-risk", None])
def test_only_humans_can_approve(reg, actor):
    vid = _cand(reg)
    models.evaluate_criteria(reg, vid, reference_metrics={"profit_total": 0})
    with pytest.raises(GovernanceError):
        models.approve(reg, vid, by=actor)


def test_failed_version_cannot_be_approved_or_activated(reg):
    vid = _cand(reg, profit_total=-0.1)
    models.evaluate_criteria(reg, vid, reference_metrics={"profit_total": 0})
    with pytest.raises(GovernanceError):
        models.approve(reg, vid, by="Burak")
    with pytest.raises(GovernanceError):
        models.activate(reg, vid, by="Burak")


def test_activate_and_rollback(reg):
    v1 = _cand(reg)
    models.evaluate_criteria(reg, v1, reference_metrics={"profit_total": 0})
    models.approve(reg, v1, by="Burak")
    models.activate(reg, v1, by="Burak")
    v2 = _cand(reg, profit_total=0.2)
    assert models.evaluate_criteria(reg, v2, reference_metrics={"profit_total": 0})["checks"]["beats_active_version"]
    models.approve(reg, v2, by="Burak")
    models.activate(reg, v2, by="Burak")
    assert models.active(reg, "bot_a")["id"] == v2 and models.get(reg, v1)["status"] == "retired"
    assert models.rollback(reg, "bot_a", by="Burak") == v1
    assert models.active(reg, "bot_a")["id"] == v1
    events = [e["event"] for e in reg.query("SELECT event FROM model_events ORDER BY id")]
    assert "rolled_back" in events and events.count("activated") == 3


def test_new_version_must_beat_active(reg):
    v1 = _cand(reg)
    models.evaluate_criteria(reg, v1, reference_metrics={"profit_total": 0})
    models.approve(reg, v1, by="Burak")
    models.activate(reg, v1, by="Burak")
    worse = _cand(reg, profit_total=0.05)
    assert not models.evaluate_criteria(reg, worse, reference_metrics={"profit_total": 0})["passed"]


def test_rollback_without_history_fails(reg):
    with pytest.raises(GovernanceError):
        models.rollback(reg, "bot_a", by="Burak")


def test_explain_roundtrip():
    texts = top_contributions(np.array([[0.1, -0.5, 0.2], [0.0, 0.01, -0.3]]), ["a", "b", "c"], k=2)
    assert texts[0] == "b:-0.500;c:+0.200"
    assert parse(texts[1]) == {"c": -0.3, "b": 0.01}


def test_build_and_store_decisions(reg):
    dates = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    sig = pd.DataFrame({"date": dates, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 5.0,
                        "up": [0.7, 0.65, 0.8, 0.9, 0.1], "do_predict": 1, "kitb_explain": "f1:+0.300;f2:-0.100"})
    trades = [
        {"pair": "BTC/USDT", "open_date": "2024-01-01 01:00:00+00:00", "profit_abs": -3.0, "profit_ratio": -0.01,
         "amount": 1.0, "open_rate": 100, "close_rate": 99, "fee_open": 0.0017, "fee_close": 0.0017,
         "exit_reason": "stop_loss", "stake_amount": 100, "stop_loss_ratio": -0.06},
        {"pair": "BTC/USDT", "open_date": "2024-01-01 03:00:00+00:00", "profit_abs": 2.0, "profit_ratio": 0.02,
         "amount": 1.0, "open_rate": 100, "close_rate": 102, "fee_open": 0.0017, "fee_close": 0.0017},
    ]
    regimes = {"BTC/USDT": pd.DataFrame({"valid_from": [pd.Timestamp("2023-12-31", tz="UTC")],
                                         "trend": ["up"], "volatility": ["low"]})}
    decs = build_decisions({"BTC/USDT": sig}, trades, strategy="S", timeframe="1h", entry_proba=0.6,
                           regimes_by_pair=regimes)
    assert decs[0]["proba_up"] == 0.7  # Signalkerze 00:00 für Einstieg 01:00
    assert decs[0]["decided_at"] == "2024-01-01T01:00:00+00:00"
    assert decs[0]["model_error"] == "false_positive" and decs[1]["model_error"] is None
    assert decs[0]["features"] == {"top_contributions": {"f1": 0.3, "f2": -0.1}}
    assert decs[0]["market"]["trend"] == "up"
    assert decs[0]["outcome"]["fees_abs"] == pytest.approx(100 * 0.0017 + 99 * 0.0017)
    assert store_decisions(reg, decs, source="backtest", run_ref="x", bot="bot_a", model_identifier="i") == 2
    assert len(reg.fetch("decisions")) == 2


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 5000)
    assert psi(base, rng.normal(0, 1, 5000)) < 0.1
    assert psi(base, rng.normal(1.5, 1, 5000)) > 0.25


def test_feature_drift_on_volatility_jump():
    rng = np.random.default_rng(1)
    n = 24 * 120
    vol = np.where(np.arange(n) < n - 24 * 30, 0.005, 0.03)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1, n) * vol))
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"), "close": close,
                       "high": close * 1.002, "low": close * 0.998, "volume": 100.0})
    res = feature_drift(df, reference_days=90, recent_days=30)
    assert res["features"]["volatility_24"]["level"] == "significant" and res["drift"] == "significant"


def test_performance_drift():
    ts = pd.date_range("2024-01-01", periods=100, freq="1D", tz="UTC")
    good = pd.DataFrame({"decided_at": ts, "win": [True] * 70 + [False] * 30, "model_error": [None] * 100})
    assert performance_drift(good)["status"] == "drift"  # die letzten 30 sind alle Verluste
    stable = pd.DataFrame({"decided_at": ts, "win": [True, False] * 50, "model_error": [None] * 100})
    assert performance_drift(stable)["status"] == "stable"
    assert performance_drift(stable.iloc[:20])["status"] == "insufficient_data"
