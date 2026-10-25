"""Stil-KIs (Burak 25.10.2026): der Stil eines Regel-Bots schlägt vor, die KI entscheidet aus dem Gedächtnis."""

import numpy as np
import pandas as pd
import pytest

from kitb import rule_learning


class _Base:
    def advise_entry(self, df, metadata):
        df["enter_long"] = 0
        df["enter_tag"] = ""
        df.loc[df.index[-1], ["enter_long", "enter_tag"]] = (1, "ema_cross_up")
        return df


class _Style(rule_learning.StyleAiMixin, _Base):
    timeframe = "15m"
    stoploss = -0.10

    def __init__(self, on=True, explore=0.2):
        self.config = {"kitb_style_ai": on, "bot_name": "ema_cross_15m", "kitb_practice": {"explore_share": explore}}

    def _is_live_like(self):
        return True


def _df():
    return pd.DataFrame({"date": pd.date_range("2026-10-25", periods=5, freq="15min", tz="UTC"),
                         "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})


@pytest.fixture
def learn(monkeypatch):
    calls = {}

    def chance(bot, feats, user_data=None, sims=True):
        calls["sims"] = sims
        return calls["allow"], 0.4, "Test"

    monkeypatch.setattr("kitb.self_learning.win_chance", chance)
    monkeypatch.setattr("kitb.self_learning.stop_for", lambda bot, pair, when, user_data=None, sims=True: 4)
    monkeypatch.setattr("kitb.shared_memory.market_snapshot", lambda df, m, pair=None: pd.DataFrame({"rsi_14": [50.0] * len(df)}))
    return calls


def test_style_suggestion_taken_when_ai_agrees(learn):
    learn["allow"] = True
    df = _Style().advise_entry(_df(), {"pair": "BTC/USDT"})
    assert df["enter_long"].iloc[-1] == 1 and df["enter_tag"].iloc[-1] == "signal: ema_cross_up · stop:4"
    assert learn["sims"] is False  # Stil-KIs lernen ohne Simulationen (Arbeitsspeicher)


def test_rejected_suggestion_dropped_or_taken_as_learning_trade(learn):
    learn["allow"] = False
    assert _Style(explore=0.0).advise_entry(_df(), {"pair": "BTC/USDT"})["enter_long"].iloc[-1] == 0
    df = _Style(explore=1.0).advise_entry(_df(), {"pair": "BTC/USDT"})
    assert df["enter_long"].iloc[-1] == 1 and df["enter_tag"].iloc[-1].startswith("lern: ema_cross_up")


def test_without_flag_strategy_unchanged(learn):
    learn["allow"] = False
    df = _Style(on=False).advise_entry(_df(), {"pair": "BTC/USDT"})
    assert df["enter_long"].iloc[-1] == 1 and df["enter_tag"].iloc[-1] == "ema_cross_up"


def test_learned_stop_never_looser_than_hard_limit():
    class T:
        enter_tag = "signal: x · stop:6"
        leverage = 1.0

    s = _Style()
    s.stoploss = -0.06
    sl = s.custom_stoploss("BTC/USDT", T(), None, 1.0, 0.0)
    assert np.isclose(sl, 0.06)
    T.enter_tag = "ema_cross_up"
    assert s.custom_stoploss("BTC/USDT", T(), None, 1.0, 0.0) is None


def test_fleet_has_16_ais_and_trend_d1_stays():
    import re

    from kitb.config import load_project_config

    project = load_project_config()
    compose = open("docker-compose.yml", encoding="utf-8").read()
    running = set(re.findall(r'command: \["paper", "run", "--bot", "(\w+)"\]', compose))
    styles = {b for b in running if project.get("paper_strategies", {}).get(b, {}).get("mode") == "explore"}
    learning = {b for b in running if project.get("paper_strategies", {}).get(b, {}).get("mode") == "learning"}
    ais = {b for b in running if b in project["bots"]} | learning | styles
    assert len(ais) == 16 and "trend_btc_d1" in styles and "trend_d1" in running and "trend_d1" not in ais
