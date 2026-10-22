"""Tagesstrategien, einheitliche Aufwärmphase, faire Benchmark-Kennzahlen und schärferes Agenten-Kriterium."""

import pandas as pd
import pytest

from kitb.backtest import equal_weight_max_drawdown
from kitb.compare import return_to_drawdown
from kitb.config import load_project_config
from kitb.data.synthetic import generate_hourly, resample
from kitb.indicators import STARTUP_CANDLES_1D, STARTUP_CANDLES_1H
from kitb.strategies import compute_signals, load_strategy

EV = load_project_config()["evaluation"]
DAILY = EV["daily_strategies"]
HOURLY = EV["strategies"] + list(load_project_config()["agents"]["pairs_to_evaluate"])


@pytest.fixture(scope="module")
def daily():
    return resample(generate_hourly("BTC/USDT", "2021-01-01", "2023-01-01", seed=5), "1d")


@pytest.mark.parametrize("name", HOURLY)
def test_hourly_strategies_share_startup(name):
    # Sonst hängt das Ergebnis einer Strategie davon ab, mit welchen anderen sie im selben Lauf steht.
    assert load_strategy(name).startup_candle_count == STARTUP_CANDLES_1H


@pytest.mark.parametrize("name", DAILY)
def test_daily_strategy_contract(name):
    strat = load_strategy(name)
    assert strat.timeframe == "1d" and strat.can_short is False
    assert strat.startup_candle_count == STARTUP_CANDLES_1D
    assert -0.2 <= strat.stoploss < 0 and strat.minimal_roi == {"0": 100.0}


@pytest.mark.parametrize("name", DAILY)
def test_daily_no_lookahead(name, daily):
    strat = load_strategy(name)
    full = compute_signals(strat, daily)[["enter_long", "exit_long"]]
    assert full["enter_long"].sum() > 0, "Strategie erzeugt keine Signale – Test wäre wertlos"
    for cut in (300, 450, 600):
        pd.testing.assert_frame_equal(compute_signals(strat, daily.iloc[:cut])[["enter_long", "exit_long"]],
                                      full.iloc[:cut])


def test_equal_weight_drawdown():
    idx = pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC")
    a = pd.Series([100, 50, 50, 100], index=idx)   # −50 %
    b = pd.Series([10, 10, 10, 10], index=idx)     # flach
    assert equal_weight_max_drawdown({"A": a, "B": b}) == pytest.approx(0.25)


def test_return_to_drawdown():
    assert return_to_drawdown(0.10, 0.05) == pytest.approx(2.0)
    assert return_to_drawdown(0.10, 0) is None
    assert return_to_drawdown(None, 0.1) is None


def test_agent_criterion_requires_train():
    assert load_project_config()["agents"]["benefit_criteria"]["require_train_not_worse"] is True
