"""Phase 5: Agentenrahmen ist azyklisch, deterministisch, ohne Netz und kann Risiko nicht erhöhen."""

import inspect

import pandas as pd
import pytest

import kitb.agents.core as core
from kitb.agents import AGENT_COLUMNS, Agent, Coordinator, CycleError, default_agents
from kitb.config import load_project_config
from kitb.data.synthetic import generate_hourly
from kitb.strategies import load_strategy
from test_strategy_lookahead import signals


@pytest.fixture(scope="module")
def data():
    return generate_hourly("BTC/USDT", "2023-01-01", "2023-07-01", seed=9)


def test_each_agent_runs_once_in_dependency_order(data):
    coord = Coordinator(default_agents())
    out = coord.run(data)
    assert set(AGENT_COLUMNS) <= set(out.columns)
    assert all(n == 1 for n in coord.calls.values()) and len(coord.calls) == 5
    assert coord.order.index("risk") > coord.order.index("trend")
    assert coord.order[-1] == "coordinator"


def test_cycles_are_rejected():
    f = lambda df: {}
    with pytest.raises(CycleError):
        Coordinator([Agent("a", "A", ("b",), (), f), Agent("b", "B", ("a",), (), f)])
    with pytest.raises(ValueError):
        Coordinator([Agent("a", "A", ("missing",), (), f)])


def test_agent_output_contract_enforced(data):
    bad = Agent("x", "X", (), ("agent_x",), lambda df: {"other": df["close"]})
    with pytest.raises(ValueError):
        Coordinator([bad]).run(data)


def test_agents_make_no_external_calls():
    src = inspect.getsource(core)
    for forbidden in ("import requests", "import ccxt", "urllib", "socket", "anthropic", "openai", "http"):
        assert forbidden not in src


def test_agents_are_causal(data):
    full = Coordinator(default_agents()).run(data)
    part = Coordinator(default_agents()).run(data.iloc[:2500])
    pd.testing.assert_frame_equal(part[AGENT_COLUMNS], full[AGENT_COLUMNS].iloc[:2500])


@pytest.mark.parametrize("gated,base", list(load_project_config()["agents"]["pairs_to_evaluate"].items()))
def test_gate_only_removes_entries_and_keeps_risk(gated, base, data):
    g, b = load_strategy(gated), load_strategy(base)
    sg, sb = signals(g, data), signals(b, data)
    assert ((sg["enter_long"] == 1) <= (sb["enter_long"] == 1)).all()  # Teilmenge
    pd.testing.assert_series_equal(sg["exit_long"], sb["exit_long"])
    assert g.stoploss == b.stoploss and g.minimal_roi == b.minimal_roi


@pytest.mark.parametrize("gated", list(load_project_config()["agents"]["pairs_to_evaluate"]))
def test_gated_strategies_have_no_lookahead(gated, data):
    strat = load_strategy(gated)
    full = signals(strat, data)
    for cut in (1500, 3000):
        pd.testing.assert_frame_equal(signals(strat, data.iloc[:cut]), full.iloc[:cut])


def test_agents_disabled_by_default():
    assert load_project_config()["agents"]["enabled"] is False
