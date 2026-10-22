"""Keine Strategie darf Zukunftsdaten nutzen: Signale auf abgeschnittenen Daten
müssen identisch zu den Signalen auf dem vollen Datensatz sein."""

import importlib.util

import pandas as pd
import pytest

from kitb import paths
from kitb.config import load_project_config
from kitb.data.synthetic import generate_hourly

STRATEGIES = load_project_config()["evaluation"]["strategies"]


def load_strategy(name):
    spec = importlib.util.spec_from_file_location(name, paths.USER_DATA / "strategies" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)({"timeframe": "1h", "stake_currency": "USDT", "runmode": None})


def signals(strategy, df):
    meta = {"pair": "BTC/USDT"}
    out = strategy.populate_indicators(df.copy(), meta)
    out = strategy.populate_entry_trend(out, meta)
    out = strategy.populate_exit_trend(out, meta)
    for col in ("enter_long", "exit_long"):
        if col not in out:
            out[col] = 0
    return out[["enter_long", "exit_long"]].fillna(0).astype(int)


@pytest.fixture(scope="module")
def data():
    return generate_hourly("BTC/USDT", "2023-01-01", "2023-07-01", seed=3)


@pytest.mark.parametrize("name", STRATEGIES)
def test_no_lookahead(name, data):
    strat = load_strategy(name)
    full_sig = signals(strat, data)
    assert full_sig["enter_long"].sum() > 0, "Strategie erzeugt keine Signale – Test wäre wertlos"
    for cut in (1500, 2500, 3500):
        pd.testing.assert_frame_equal(signals(strat, data.iloc[:cut]), full_sig.iloc[:cut])


@pytest.mark.parametrize("name", STRATEGIES)
def test_long_only_spot_with_stoploss(name):
    strat = load_strategy(name)
    assert strat.can_short is False
    assert -0.2 <= strat.stoploss < 0
    assert strat.timeframe == "1h"
    assert strat.minimal_roi == {"0": 100.0}  # Ausstieg nur über eindeutige Regeln / Stoploss


@pytest.mark.parametrize("name", STRATEGIES)
def test_startup_covers_indicators(name, data):
    """Nach startup_candle_count dürfen die Signalspalten keine Lücken aus Indikator-Warmup haben."""
    strat = load_strategy(name)
    out = strat.populate_indicators(data.copy(), {"pair": "BTC/USDT"})
    ind_cols = [c for c in out.columns if c not in data.columns]
    assert out[ind_cols].iloc[strat.startup_candle_count:].notna().all().all()
