"""Phase 3: KI-Bots unter identischen Bedingungen, ohne Datenleck, reproduzierbar."""

import json

import numpy as np
import pandas as pd
import pytest

from kitb.config import load_json, load_project_config
from kitb.data.synthetic import write_synthetic_dataset
from kitb.ml import FREQAI_SHARED, build_overlay, run_bot

PROJECT = load_project_config()
SHARED = load_json(FREQAI_SHARED)["freqai"]


def test_overlays_share_all_conditions_except_features_and_model(tmp_path):
    a = json.loads(build_overlay("bot_a", "a", tmp_path).read_text())["freqai"]
    b = json.loads(build_overlay("bot_b", "b", tmp_path).read_text())["freqai"]
    for key in ("train_period_days", "backtest_period_days", "data_split_parameters", "kitb"):
        assert a[key] == b[key] == SHARED[key]
    assert a["feature_parameters"]["label_period_candles"] == b["feature_parameters"]["label_period_candles"]
    assert a["model_training_parameters"] != b["model_training_parameters"]


def test_no_train_test_leakage_config():
    """Ohne Mischen liegt das interne Eval-Set am Ende des Trainingsfensters. Ist es länger als der
    Label-Horizont, reichen Trainingslabels nicht in den Backtest-Zeitraum."""
    split = SHARED["data_split_parameters"]
    assert split["shuffle"] is False
    train_candles = SHARED["train_period_days"] * 24
    assert split["test_size"] * train_candles > SHARED["feature_parameters"]["label_period_candles"]
    assert SHARED["kitb"]["label_threshold"] > 2 * 0.0017  # Label-Schwelle über Round-Trip-Kosten


def test_label_definition():
    from kitb.ml_strategy import KitbFreqaiBase

    class Dummy:
        class_names = []

    strat = KitbFreqaiBase.__new__(KitbFreqaiBase)
    strat.freqai_info = {"feature_parameters": {"label_period_candles": 2},
                         "kitb": {"label_threshold": 0.01, "entry_proba": 0.6, "exit_proba": 0.45}}
    strat.freqai = Dummy()
    df = pd.DataFrame({"close": [100, 100, 102, 100, 99, 99]})
    out = strat.set_freqai_targets(df, {})
    assert list(out["&s-up_or_down"][:4]) == ["up", "down", "down", "down"]
    assert out["&s-up_or_down"][4:].isna().all()
    assert strat.freqai.class_names == ["down", "up"]


@pytest.fixture(scope="module")
def ml_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("ml")
    data = root / "synthetic"
    write_synthetic_dataset(data, ["BTC/USDT", "ETH/USDT"], ["1h", "1d"], "2022-01-01", "2023-07-01", seed=5)
    ud = root / "user_data"
    ud.mkdir()
    return data, ud


@pytest.mark.parametrize("bot", ["bot_a", "bot_b"])
def test_bot_runs_walk_forward_and_measures_resources(bot, ml_env, tmp_path):
    data, ud = ml_env
    r = run_bot(bot, timerange="20230501-20230701", datadir=data, data_source="synthetic", offline=True,
                registry_path=None, results_dir=tmp_path, user_data_dir=ud)
    m = next(iter(r["strategies"].values()))["metrics"]
    assert m["bot"] == bot and m["synthetic_data"] is True
    res = r["resources"]
    assert res["wall_seconds"] > 0 and res["peak_rss_mb"] > 0 and res["model_disk_kb"] > 0
    assert res["trainings"] >= 2 * 2  # 2 Paare x mind. 2 Walk-Forward-Fenster


def test_bot_a_is_reproducible_and_cleans_up(ml_env, tmp_path):
    data, ud = ml_env
    kw = dict(timerange="20230601-20230701", datadir=data, data_source="synthetic", offline=True,
              registry_path=None, results_dir=tmp_path, user_data_dir=ud, keep_models=False)
    r1, r2 = run_bot("bot_a", **kw), run_bot("bot_a", **kw)
    m1, m2 = (next(iter(r["strategies"].values()))["metrics"] for r in (r1, r2))
    for k in ("profit_total", "total_trades", "winrate"):
        assert m1[k] == m2[k]
    assert not (ud / "models" / r1["identifier"]).exists()
