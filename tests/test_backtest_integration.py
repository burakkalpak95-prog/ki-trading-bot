"""Integrationstest: synthetische Daten -> Validierung -> Freqtrade-Backtest (offline) -> SQLite.

Läuft ohne Internet dank statischer Marktdefinitionen (kitb.offline).
"""

import pytest

from kitb.backtest import DataQualityError, run_backtest
from kitb.data.storage import load_ohlcv_raw, store_ohlcv
from kitb.data.synthetic import write_synthetic_dataset
from kitb.registry import Registry


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic")
    write_synthetic_dataset(d, ["BTC/USDT", "ETH/USDT"], ["1h"], "2024-01-01", "2024-07-01", seed=11)
    return d


def _run(datadir, tmp_path, name):
    return run_backtest(datadir=datadir, data_source="synthetic", offline=True,
                        registry=Registry(tmp_path / "reg.sqlite"), results_dir=tmp_path / name)


def test_backtest_runs_and_is_reproducible(synth_dir, tmp_path):
    a = _run(synth_dir, tmp_path, "a")
    b = _run(synth_dir, tmp_path, "b")
    m = a["metrics"]
    assert m["total_trades"] > 0
    assert m["effective_fee_per_side"] == pytest.approx(0.0017)
    assert m["total_cost_abs_incl_slippage"] > 0
    assert m["synthetic_data"] is True
    assert set(m["buy_and_hold"]) == {"BTC/USDT", "ETH/USDT", "equal_weight_net_return", "equal_weight_max_drawdown"}
    assert 0 <= m["buy_and_hold"]["equal_weight_max_drawdown"] < 1
    for key in ("profit_total", "max_drawdown_account", "profit_factor", "winrate", "total_trades", "exposure"):
        assert a["metrics"][key] == b["metrics"][key], key
    assert a["data_fingerprints"] == b["data_fingerprints"]
    runs = Registry(tmp_path / "reg.sqlite").fetch("backtest_runs")
    assert len(runs) == 2 and all(r["synthetic"] == 1 for r in runs)


def test_fees_reduce_result(synth_dir, tmp_path, monkeypatch):
    from kitb import backtest as bt
    from kitb.costs import CostModel

    with_costs = _run(synth_dir, tmp_path, "costs")["metrics"]
    monkeypatch.setattr(bt.CostModel, "from_project_config",
                        classmethod(lambda cls, _: CostModel(0.0, 0.0, 0.0)))
    no_costs = _run(synth_dir, tmp_path, "free")["metrics"]
    assert with_costs["total_trades"] == no_costs["total_trades"]
    assert with_costs["profit_total_abs"] < no_costs["profit_total_abs"]


def test_corrupt_data_blocks_backtest(synth_dir, tmp_path):
    bad = tmp_path / "bad"
    write_synthetic_dataset(bad, ["BTC/USDT", "ETH/USDT"], ["1h"], "2024-01-01", "2024-03-01", seed=1)
    df = load_ohlcv_raw(bad, "BTC/USDT", "1h")
    df.loc[100, "high"] = df.loc[100, "low"] / 2
    store_ohlcv(bad, "BTC/USDT", "1h", df)
    with pytest.raises(DataQualityError):
        _run(bad, tmp_path, "x")


def test_offline_markets_refused_for_real_data(tmp_path):
    real = tmp_path / "binance"
    real.mkdir()
    with pytest.raises(ValueError):
        run_backtest(datadir=real, data_source="binance", offline=True)
