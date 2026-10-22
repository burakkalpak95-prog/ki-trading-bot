"""Baseline darf keine Zukunftsdaten nutzen: Signale auf abgeschnittenen Daten
müssen identisch zu den Signalen auf dem vollen Datensatz sein."""

import importlib.util

import pandas as pd

from kitb import paths
from kitb.data.synthetic import generate_hourly


def load_strategy():
    spec = importlib.util.spec_from_file_location(
        "baseline", paths.USER_DATA / "strategies" / "BaselineEmaCross.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BaselineEmaCross({"timeframe": "1h", "stake_currency": "USDT"} | {"runmode": None})


def signals(strategy, df):
    meta = {"pair": "BTC/USDT"}
    out = strategy.populate_indicators(df.copy(), meta)
    out = strategy.populate_entry_trend(out, meta)
    out = strategy.populate_exit_trend(out, meta)
    return out[["enter_long", "exit_long"]].fillna(0)


def test_no_lookahead():
    strat = load_strategy()
    full = generate_hourly("BTC/USDT", "2024-01-01", "2024-04-01", seed=3)
    full_sig = signals(strat, full)
    assert full_sig["enter_long"].sum() > 0  # Test ist sonst wertlos
    for cut in (500, 1000, 1700):
        part_sig = signals(strat, full.iloc[:cut])
        pd.testing.assert_frame_equal(part_sig, full_sig.iloc[:cut])


def test_baseline_is_long_only_spot():
    strat = load_strategy()
    assert strat.can_short is False
    assert strat.stoploss < 0
