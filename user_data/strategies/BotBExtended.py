"""Bot B – erweitert: mehr Features (Trend, Momentum, Volatilität, Volumen), Mehrfach-Zeitrahmen
und BTC/ETH-Querbezug; Modell: LightGBM. Prüft, ob mehr Komplexität messbaren Mehrwert bringt."""

import talib.abstract as ta
from pandas import DataFrame

from kitb.ml_strategy import KitbFreqaiBase


class BotBExtended(KitbFreqaiBase):
    startup_candle_count = 300

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: dict,
                                       **kwargs) -> DataFrame:
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)
        dataframe["%-natr-period"] = ta.NATR(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        bb = ta.BBANDS(dataframe, timeperiod=period, nbdevup=2.0, nbdevdn=2.0)
        dataframe["%-bb_width-period"] = (bb["upperband"] - bb["lowerband"]) / bb["middleband"]
        dataframe["%-bb_pos-period"] = (dataframe["close"] - bb["lowerband"]) / (bb["upperband"] - bb["lowerband"])
        dataframe["%-dist_ema-period"] = dataframe["close"] / ta.EMA(dataframe, timeperiod=period) - 1
        dataframe["%-rel_volume-period"] = dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe["%-pct_change"] = dataframe["close"].pct_change()
        macd = ta.MACD(dataframe)
        dataframe["%-macd_hist_norm"] = macd["macdhist"] / dataframe["close"]
        dataframe["%-dist_ema200"] = dataframe["close"] / ta.EMA(dataframe, timeperiod=200) - 1
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe = super().feature_engineering_standard(dataframe, metadata, **kwargs)
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek / 6
        return dataframe
