import numpy as np
import pandas as pd
import pytest

from kitb.data.synthetic import generate_hourly, resample
from kitb.evaluation import (attribute_trades_to_regimes, chronological_split, classify_regimes,
                             regime_summary, rolling_windows)

T = pd.Timestamp


def test_split_is_chronological_and_complete():
    segs = chronological_split(T("2023-01-01", tz="UTC"), T("2025-01-01", tz="UTC"),
                               {"train": 0.6, "validation": 0.2, "test": 0.2})
    assert [s.name for s in segs] == ["train", "validation", "test"]
    assert segs[0].start == T("2023-01-01", tz="UTC") and segs[-1].end == T("2025-01-01", tz="UTC")
    for a, b in zip(segs, segs[1:]):
        assert a.end == b.start  # lückenlos, ohne Überlappung
    assert segs[1].timerange == f"{segs[1].start:%Y%m%d}-{segs[1].end:%Y%m%d}"


def test_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        chronological_split(T("2023-01-01"), T("2024-01-01"), {"a": 0.5, "b": 0.6})


def test_rolling_windows():
    wins = rolling_windows(T("2023-01-01"), T("2023-12-31"), 90, 90)
    assert len(wins) == 4
    assert all(w.end - w.start == pd.Timedelta(days=90) for w in wins)


def test_regimes_are_causal():
    daily = resample(generate_hourly("BTC/USDT", "2022-01-01", "2024-01-01", seed=5), "1d")
    full = classify_regimes(daily)
    part = classify_regimes(daily.iloc[:400])
    merged = part.merge(full, on="date", suffixes=("_p", "_f"))
    assert (merged["trend_p"] == merged["trend_f"]).all()
    assert (merged["volatility_p"] == merged["volatility_f"]).all()
    assert (full["valid_from"] == full["date"] + pd.Timedelta(days=1)).all()
    assert set(full["trend"]) <= {"up", "down", "sideways"}


def test_regime_classification_on_clear_trends():
    dates = pd.date_range("2023-01-01", periods=200, freq="1D", tz="UTC")
    up = pd.DataFrame({"date": dates, "close": np.linspace(100, 300, 200)})
    down = pd.DataFrame({"date": dates, "close": np.linspace(300, 100, 200)})
    assert (classify_regimes(up)["trend"] == "up").all()
    assert (classify_regimes(down)["trend"] == "down").all()


def test_trade_attribution_uses_only_known_regime():
    dates = pd.date_range("2023-01-01", periods=3, freq="1D", tz="UTC")
    reg = pd.DataFrame({"date": dates, "trend": ["up", "down", "sideways"], "volatility": ["low"] * 3,
                        "valid_from": dates + pd.Timedelta(days=1)})
    trades = [  # Einstieg am 2.1. 12:00 -> nur Phase vom 1.1. bekannt (gültig ab 2.1. 00:00)
        {"pair": "BTC/USDT", "open_date": "2023-01-02 12:00:00+00:00", "profit_abs": 5.0, "profit_ratio": 0.01},
        {"pair": "BTC/USDT", "open_date": "2023-01-01 05:00:00+00:00", "profit_abs": -1.0, "profit_ratio": -0.01},
    ]
    att = attribute_trades_to_regimes(trades, {"BTC/USDT": reg}).sort_values("open_date")
    assert list(att["trend"]) == ["unknown", "up"]
    summary = regime_summary(att, min_trades=2)
    assert {r["trend"] for r in summary} == {"unknown", "up"} and all(r["low_sample"] for r in summary)
