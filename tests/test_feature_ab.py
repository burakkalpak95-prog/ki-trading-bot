"""Vorher/Nachher-Test KI 1: Schalter im Overlay, Urteil verlangt genug Trades und einen klaren Unterschied."""

import json

from kitb import feature_ab


def _r(trades, profit, winrate=0.5):
    m = {k: None for k in feature_ab.KEYS}
    m.update(total_trades=trades, profit_total=profit, winrate=winrate)
    return {"metrics": m, "resources": {"wall_seconds": 3600}}


def test_overlay_sets_switch(tmp_path):
    for extra in (True, False):
        cfg = json.loads(feature_ab._overlay(tmp_path, f"x-{extra}", extra).read_text(encoding="utf-8"))
        assert cfg["freqai"]["kitb"]["extra_features"] is extra and cfg["freqai"]["identifier"] == f"x-{extra}"


def test_verdict_and_render():
    assert "Zu wenige" in feature_ab.verdict(_r(10, 0.5), _r(100, 0.1))
    assert "Kein klarer" in feature_ab.verdict(_r(100, 0.11), _r(100, 0.10))
    assert "besser" in feature_ab.verdict(_r(100, 0.20), _r(100, 0.10))
    assert "schlechter" in feature_ab.verdict(_r(100, 0.0), _r(100, 0.10))
    md = feature_ab.render(_r(100, 0.2), _r(90, 0.1), "20240101-20251104", ["BTC/USDT"])
    assert "| Trades | 100 | 90 |" in md and "20.00%" in md


def test_sweep_overlay_and_render(tmp_path):
    v = feature_ab.SWEEP["1h_ziel_3pct_24h"]
    cfg = json.loads(feature_ab._overlay(tmp_path, "s", True, v).read_text(encoding="utf-8"))["freqai"]
    assert cfg["kitb"]["label_threshold"] == 0.03 and cfg["kitb"]["entry_proba"] == 0.65
    assert cfg["feature_parameters"]["label_period_candles"] == 24
    h = json.loads(feature_ab._overlay(tmp_path, "h", True, feature_ab.SWEEP["1h_halten_12h"]).read_text(encoding="utf-8"))
    assert h["freqai"]["kitb"]["max_hold_candles"] == 12 and h["freqai"]["kitb"]["exit_proba"] == 0.0
    assert cfg["feature_parameters"]["include_timeframes"] == ["1h", "4h", "1d"]
    r = _r(100, -0.1)
    r["trades"] = {"avg_profit": -0.002, "exit_reasons": {"stop_loss": {"n": 40, "avg_profit": -0.06},
                                                          "exit_signal": {"n": 60, "avg_profit": 0.01}}}
    md = feature_ab.render_sweep({"heute_15m": r}, "20240101-20251104")
    assert "| heute_15m | 100 |" in md and "40%" in md and "stop_loss 40×" in md


def test_trade_summary():
    from kitb.ml import trade_summary

    s = trade_summary([{"profit_ratio": 0.02, "exit_reason": "roi", "trade_duration": 60},
                       {"profit_ratio": -0.06, "exit_reason": "stop_loss", "trade_duration": 30}])
    assert abs(s["avg_profit"] + 0.02) < 1e-9 and s["exit_reasons"]["stop_loss"]["n"] == 1 and trade_summary([]) == {}
