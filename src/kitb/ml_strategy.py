"""Gemeinsame Basis der KI-Bots (FreqAI).

Beide Bots teilen Ziel (Label), Ein-/Ausstiegsregeln, Stoploss und Kapital-/Kostenbedingungen.
Sie unterscheiden sich nur in Features (``feature_engineering_*``) und Modell (``freqaimodel``).

Label: Steigt der Schlusskurs in den nächsten ``label_period_candles`` Kerzen um mehr als
``label_threshold`` (Standard 0,6 % > Round-Trip-Kosten 0,34 %)? -> Klasse ``up``, sonst ``down``.
Das Label nutzt bewusst Zukunftsdaten (Lernziel); FreqAI trainiert nur auf Daten **vor** dem
jeweiligen Testfenster. Mit ``shuffle=false`` und ``test_size`` > Label-Horizont liegen die letzten
Trainingslabels nicht im Backtest-Zeitraum (siehe tests/test_ml.py).
"""

import numpy as np
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class KitbFreqaiBase(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False
    minimal_roi = {"0": 100.0}
    stoploss = -0.06
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 200
    order_types = {"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False}

    LABEL = "&s-up_or_down"

    def _kitb(self) -> dict:
        return self.freqai_info.get("kitb", {"label_threshold": 0.006, "entry_proba": 0.6, "exit_proba": 0.45})

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour / 23
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        horizon = self.freqai_info["feature_parameters"]["label_period_candles"]
        future_return = dataframe["close"].shift(-horizon) / dataframe["close"] - 1
        self.freqai.class_names = ["down", "up"]
        dataframe[self.LABEL] = np.where(future_return > self._kitb()["label_threshold"], "up", "down")
        dataframe.loc[future_return.isna(), self.LABEL] = np.nan
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.freqai.start(dataframe, metadata, self)

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[(df["do_predict"] == 1) & (df["up"] > self._kitb()["entry_proba"]) & (df["volume"] > 0),
               ["enter_long", "enter_tag"]] = (1, "ml_up")
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[(df["up"] < self._kitb()["exit_proba"]) & (df["volume"] > 0),
               ["exit_long", "exit_tag"]] = (1, "ml_weak")
        return df
