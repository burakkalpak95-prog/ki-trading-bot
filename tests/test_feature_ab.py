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
