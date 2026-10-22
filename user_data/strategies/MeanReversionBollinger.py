"""Mean Reversion: Überverkauft unter dem unteren Bollinger-Band, nur im übergeordneten Aufwärtstrend.

Einstieg: Close < unteres Bollinger-Band(20, 2) UND RSI(14) < 30 UND Close > EMA(200).
Ausstieg: Close kreuzt Bollinger-Mittellinie von unten. Stoploss −7 %.
"""

from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IStrategy
from kitb.risk import RiskManagedMixin
from kitb import indicators as ind
from kitb.indicators import STARTUP_CANDLES_1H


class MeanReversionBollinger(RiskManagedMixin, IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False
    minimal_roi = {"0": 100.0}
    stoploss = -0.07
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = STARTUP_CANDLES_1H
    order_types = {"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ind.add_bollinger(dataframe, 20, 2.0)
        ind.add_rsi(dataframe, 14)
        ind.add_ema(dataframe, 200)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["close"] < dataframe["bb_lower"])
            & (dataframe["rsi_14"] < 30)
            & (dataframe["close"] > dataframe["ema_200"])
            & (dataframe["volume"] > 0),
            ["enter_long", "enter_tag"],
        ] = (1, "bb_oversold")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            qtpylib.crossed_above(dataframe["close"], dataframe["bb_mid"]) & (dataframe["volume"] > 0),
            ["exit_long", "exit_tag"],
        ] = (1, "back_to_mean")
        return dataframe
