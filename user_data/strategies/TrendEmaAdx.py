"""Trendfolge: Rücksetzer im bestätigten Aufwärtstrend kaufen.

Einstieg: EMA(50) > EMA(200) UND ADX(14) > 25 UND Close kreuzt EMA(20) von unten.
Ausstieg: Close kreuzt EMA(50) von oben. Stoploss −8 %.
"""

from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IStrategy
from kitb.risk import RiskManagedMixin
from kitb.rule_learning import StyleAiMixin
from kitb import indicators as ind
from kitb.indicators import STARTUP_CANDLES_1H


class TrendEmaAdx(StyleAiMixin, RiskManagedMixin, IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False
    minimal_roi = {"0": 100.0}
    stoploss = -0.08
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = STARTUP_CANDLES_1H
    order_types = {"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ind.add_ema(dataframe, 20, 50, 200)
        ind.add_adx(dataframe, 14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["ema_50"] > dataframe["ema_200"])
            & (dataframe["adx_14"] > 25)
            & qtpylib.crossed_above(dataframe["close"], dataframe["ema_20"])
            & (dataframe["volume"] > 0),
            ["enter_long", "enter_tag"],
        ] = (1, "trend_pullback")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            qtpylib.crossed_below(dataframe["close"], dataframe["ema_50"]) & (dataframe["volume"] > 0),
            ["exit_long", "exit_tag"],
        ] = (1, "below_ema50")
        return dataframe
