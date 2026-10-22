"""Momentum: MACD-Signal mit RSI-Filter.

Einstieg: MACD kreuzt Signallinie von unten UND MACD > 0 UND 50 < RSI(14) < 70.
Ausstieg: MACD kreuzt Signallinie von oben ODER RSI(14) > 80. Stoploss −6 %.
"""

from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IStrategy
from kitb.risk import RiskManagedMixin
from kitb import indicators as ind


class MomentumMacdRsi(RiskManagedMixin, IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False
    minimal_roi = {"0": 100.0}
    stoploss = -0.06
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 200
    order_types = {"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ind.add_macd(dataframe)
        ind.add_rsi(dataframe, 14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            qtpylib.crossed_above(dataframe["macd"], dataframe["macd_signal"])
            & (dataframe["macd"] > 0)
            & dataframe["rsi_14"].between(50, 70, inclusive="neither")
            & (dataframe["volume"] > 0),
            ["enter_long", "enter_tag"],
        ] = (1, "macd_up")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (qtpylib.crossed_below(dataframe["macd"], dataframe["macd_signal"]) | (dataframe["rsi_14"] > 80))
            & (dataframe["volume"] > 0),
            ["exit_long", "exit_tag"],
        ] = (1, "macd_down_or_rsi_high")
        return dataframe
