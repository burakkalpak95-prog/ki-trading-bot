"""Langsamer Trendfolger auf Tageskerzen: 200-Tage-Durchschnitt (Lehrbuch-Regel, NICHT optimiert).

Einstieg: Tagesschluss über SMA(200). Ausstieg: Tagesschluss unter SMA(200). Stoploss −10 % wie die Baseline.
Hypothese: Wenige Trades, dadurch geringe Kosten; investiert nur in Aufwärtsphasen und meidet lange Abwärtsphasen.
"""

from pandas import DataFrame

from freqtrade.strategy import IStrategy
from kitb.risk import RiskManagedMixin
from kitb import indicators as ind
from kitb.indicators import STARTUP_CANDLES_1D


class DailyTrendSma(RiskManagedMixin, IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1d"
    can_short = False
    minimal_roi = {"0": 100.0}
    stoploss = -0.10
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = STARTUP_CANDLES_1D
    order_types = {"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ind.add_sma(dataframe, 200)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["close"] > dataframe["sma_200"]) & (dataframe["volume"] > 0),
            ["enter_long", "enter_tag"],
        ] = (1, "above_sma200")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["close"] < dataframe["sma_200"]) & (dataframe["volume"] > 0),
            ["exit_long", "exit_tag"],
        ] = (1, "below_sma200")
        return dataframe
