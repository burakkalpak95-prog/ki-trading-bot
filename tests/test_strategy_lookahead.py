"""Keine Strategie darf Zukunftsdaten nutzen: Signale auf abgeschnittenen Daten
müssen identisch zu den Signalen auf dem vollen Datensatz sein."""

import pandas as pd
import pytest

from kitb.strategies import compute_signals, load_strategy
from kitb.config import load_project_config
from kitb.data.synthetic import generate_hourly

STRATEGIES = load_project_config()["evaluation"]["strategies"]


def signals(strategy, df):
    return compute_signals(strategy, df)[["enter_long", "exit_long"]]


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
