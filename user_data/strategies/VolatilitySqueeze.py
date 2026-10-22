"""Volatilität: Ausbruch nach einer Volatilitäts-Kontraktion („Squeeze“).

Squeeze: Bollinger-Bandbreite der Vorkerze im untersten 20 %-Bereich der letzten 500 Kerzen.
Einstieg: Squeeze UND Close kreuzt oberes Bollinger-Band von unten UND Close > EMA(50).
Ausstieg: Close < Bollinger-Mittellinie. Stoploss −6 %.
"""

from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IStrategy
from kitb.risk import RiskManagedMixin
from kitb import indicators as ind


class VolatilitySqueeze(RiskManagedMixin, IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False
    minimal_roi = {"0": 100.0}
    stoploss = -0.06
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 520
    order_types = {"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ind.add_bollinger(dataframe, 20, 2.0)
        ind.add_ema(dataframe, 50)
        dataframe["bbw_rank"] = ind.rolling_percentile_rank(dataframe["bb_width"], 500)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["bbw_rank"].shift(1) < 0.2)
            & qtpylib.crossed_above(dataframe["close"], dataframe["bb_upper"])
            & (dataframe["close"] > dataframe["ema_50"])
            & (dataframe["volume"] > 0),
            ["enter_long", "enter_tag"],
        ] = (1, "squeeze_breakout")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["close"] < dataframe["bb_mid"]) & (dataframe["volume"] > 0),
            ["exit_long", "exit_tag"],
        ] = (1, "below_mid")
        return dataframe
