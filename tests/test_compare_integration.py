import json

import pytest

from kitb.compare import compare_strategies
from kitb.data.synthetic import write_synthetic_dataset
from kitb.registry import Registry


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth_cmp")
    write_synthetic_dataset(d, ["BTC/USDT", "ETH/USDT"], ["1h", "1d"], "2023-01-01", "2024-01-01", seed=21)
    return d


def test_compare_runs_all_and_keeps_test_untouched(synth, tmp_path):
    reg = Registry(tmp_path / "r.sqlite")
    strategies = ["BaselineEmaCross", "BreakoutDonchian"]
    rep = compare_strategies(datadir=synth, data_source="synthetic", strategies=strategies, offline=True,
                             registry=reg, results_dir=tmp_path / "res", report_dir=tmp_path / "rep")
    assert rep["test_set_used"] is False
    assert set(rep["results"]) == set(strategies)
    for strat in strategies:
        assert set(rep["results"][strat]["segments"]) == {"train", "validation"}
        assert rep["results"][strat]["window_stability"]["windows"] >= 1
    runs = reg.fetch("backtest_runs", 100)
    assert all(json.loads(r["metrics"])["segment"] != "test" for r in runs)
    assert not [e for e in reg.fetch("source_events", 100) if e["action"] == "test_set_used"]
    md = (tmp_path / "rep" / f"compare-{rep['compare_id']}" / "report.md").read_text()
    assert "SYNTHETISCHE DATEN" in md and "BreakoutDonchian" in md


def test_include_test_is_logged(synth, tmp_path):
    reg = Registry(tmp_path / "r.sqlite")
    rep = compare_strategies(datadir=synth, data_source="synthetic", strategies=["BaselineEmaCross"],
                             include_test=True, with_windows=False, offline=True, registry=reg,
                             results_dir=tmp_path / "res", report_dir=tmp_path / "rep")
    assert "test" in rep["results"]["BaselineEmaCross"]["segments"]
    assert [e for e in reg.fetch("source_events", 100) if e["action"] == "test_set_used"]


def test_identical_conditions_across_strategies(synth, tmp_path):
    """Alle Strategien eines Laufs nutzen dieselben Daten (Fingerprint) und dieselben Kosten."""
    reg = Registry(tmp_path / "r.sqlite")
    compare_strategies(datadir=synth, data_source="synthetic", strategies=["BaselineEmaCross", "TrendEmaAdx"],
                       with_windows=False, offline=True, registry=reg, results_dir=tmp_path / "res",
                       report_dir=tmp_path / "rep")
    runs = reg.fetch("backtest_runs", 100)
    by_segment = {}
    for r in runs:
        seg = json.loads(r["metrics"])["segment"]
        by_segment.setdefault(seg, set()).add((r["data_fingerprints"], r["cost_model"], r["config_sha256"]))
    assert all(len(v) == 1 for v in by_segment.values())
