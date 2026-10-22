"""Bot A – effizient: wenige, gut interpretierbare Features; Modell: logistische Regression."""

import talib.abstract as ta
from pandas import DataFrame

from kitb.ml_strategy import KitbFreqaiBase


class BotAEfficient(KitbFreqaiBase):
    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: dict,
                                       **kwargs) -> DataFrame:
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)
        dataframe["%-natr-period"] = ta.NATR(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe["%-pct_change"] = dataframe["close"].pct_change()
        dataframe["%-dist_ema50"] = dataframe["close"] / ta.EMA(dataframe, timeperiod=50) - 1
        return dataframe
