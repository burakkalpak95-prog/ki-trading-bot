import pytest

from kitb.config import load_project_config
from kitb.costs import CostModel
from kitb.registry import Registry
from kitb.retry import with_backoff


def test_effective_fee_includes_slippage_and_spread():
    cm = CostModel(fee_per_side=0.001, slippage_per_side=0.0005, half_spread_per_side=0.0002)
    assert cm.effective_fee_per_side == pytest.approx(0.0017)
    assert cm.round_trip_cost == pytest.approx(0.0034)


def test_project_costs_load_and_are_nonzero():
    cm = CostModel.from_project_config(load_project_config())
    assert cm.fee_per_side > 0 and cm.slippage_per_side > 0


@pytest.mark.parametrize("bad", [-0.001, 0.05, 1.0])
def test_cost_bounds(bad):
    with pytest.raises(ValueError):
        CostModel(fee_per_side=bad, slippage_per_side=0, half_spread_per_side=0)


def test_backoff_retries_then_succeeds():
    calls, sleeps = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("x")
        return "ok"

    assert with_backoff(flaky, retries=4, base_seconds=1.0, retry_on=(ConnectionError,),
                        sleep=sleeps.append) == "ok"
    assert sleeps == [1.0, 2.0]


def test_backoff_gives_up():
    sleeps = []

    def broken():
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        with_backoff(broken, retries=2, base_seconds=0.5, retry_on=(ConnectionError,), sleep=sleeps.append)
    assert sleeps == [0.5, 1.0]


def test_backoff_does_not_retry_other_errors():
    sleeps = []
    with pytest.raises(ValueError):
        with_backoff(lambda: (_ for _ in ()).throw(ValueError()), retries=3, base_seconds=1,
                     retry_on=(ConnectionError,), sleep=sleeps.append)
    assert sleeps == []


def test_registry_roundtrip(tmp_path):
    reg = Registry(tmp_path / "r.sqlite")
    reg.log_source_event("binance", "probe", "failed", {"error": "403"})
    reg.add_backtest(strategy="S", timeframe="1h", timerange=None, data_source="synthetic", synthetic=True,
                     data_fingerprints={"BTC/USDT": "abc"}, config_sha256="x", cost_model={"fee": 1},
                     freqtrade_version="v", result_file=None, metrics={"profit_total": 0.1})
    assert reg.fetch("source_events")[0]["status"] == "failed"
    run = reg.fetch("backtest_runs")[0]
    assert run["synthetic"] == 1 and '"profit_total": 0.1' in run["metrics"]
    with pytest.raises(ValueError):
        reg.fetch("sqlite_master")
