"""Zeitreihen-Momentum auf Tageskerzen (feste Parameter, NICHT optimiert).

Einstieg: Tagesschluss höher als vor 90 Tagen. Ausstieg: Tagesschluss niedriger als vor 90 Tagen.
Stoploss −10 % wie die Baseline. Hypothese: Krypto-Trends halten über Wochen bis Monate an.
"""

from pandas import DataFrame

from freqtrade.strategy import IStrategy
from kitb.risk import RiskManagedMixin
from kitb.indicators import STARTUP_CANDLES_1D

LOOKBACK_DAYS = 90


class DailyMomentum(RiskManagedMixin, IStrategy):
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
        dataframe["mom_90"] = dataframe["close"] / dataframe["close"].shift(LOOKBACK_DAYS) - 1
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["mom_90"] > 0) & (dataframe["volume"] > 0),
            ["enter_long", "enter_tag"],
        ] = (1, "momentum_up")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["mom_90"] < 0) & (dataframe["volume"] > 0),
            ["exit_long", "exit_tag"],
        ] = (1, "momentum_down")
        return dataframe
