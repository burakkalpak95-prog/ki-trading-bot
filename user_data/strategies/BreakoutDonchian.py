"""Breakout: Donchian-Kanal mit Volumenbestätigung.

Einstieg: Close > Hoch der vorherigen 20 Kerzen UND Volumen > 1,5 × SMA(20) des Volumens.
Ausstieg: Close < Tief der vorherigen 10 Kerzen. Stoploss −8 %.
"""

from pandas import DataFrame

from freqtrade.strategy import IStrategy
from kitb.risk import RiskManagedMixin
from kitb import indicators as ind


class BreakoutDonchian(RiskManagedMixin, IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False
    minimal_roi = {"0": 100.0}
    stoploss = -0.08
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 100
    order_types = {"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ind.add_donchian(dataframe, 20, 10)
        ind.add_volume_sma(dataframe, 20)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["close"] > dataframe["dc_high_20"])
            & (dataframe["volume"] > 1.5 * dataframe["vol_sma_20"]),
            ["enter_long", "enter_tag"],
        ] = (1, "donchian_breakout")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["close"] < dataframe["dc_low_10"]) & (dataframe["volume"] > 0),
            ["exit_long", "exit_tag"],
        ] = (1, "donchian_exit")
        return dataframe
