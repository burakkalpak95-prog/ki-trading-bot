"""KI-Trader – die KI handelt selbst (Paper-Lernmodus, Burak 23.10.2026).

Modell: LightGBM über FreqAI. Merkmale wie Bot B plus die aktuellen Signale aller sechs 1h-Regel-Bots als
Zusatzinformation (die KI lernt selbst, welchen Bot-Signalen sie wann traut). Im Lernmodus übt sie bewusst auch mit
unsicheren Trades und schließt Positionen spätestens nach dem Lernhorizont, damit laufend Ergebnisse entstehen.
Risikolimits gelten unverändert (RiskManagedMixin).
"""

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from kitb import indicators as ind
from kitb.ml_strategy import KitbFreqaiBase


def rule_signals(df: DataFrame) -> DataFrame:
    """Einstiegssignale der Regel-Bots (gleiche Bedingungen wie in deren Strategien), als 0/1-Merkmale."""
    d = df.copy()
    ind.add_ema(d, 20, 50, 200)
    ind.add_adx(d, 14)
    ind.add_macd(d)
    ind.add_rsi(d, 14)
    ind.add_bollinger(d, 20, 2.0)
    ind.add_donchian(d, 20, 10)
    ind.add_volume_sma(d, 20)
    ema_fast, ema_slow = ta.EMA(d, timeperiod=12), ta.EMA(d, timeperiod=26)
    bbw_rank = ind.rolling_percentile_rank(d["bb_width"], 500)
    out = DataFrame(index=df.index)
    out["%-sig_ema_cross"] = qtpylib.crossed_above(ema_fast, ema_slow).astype(float)
    out["%-sig_trend_adx"] = ((d["ema_50"] > d["ema_200"]) & (d["adx_14"] > 25)
                              & qtpylib.crossed_above(d["close"], d["ema_20"])).astype(float)
    out["%-sig_macd_rsi"] = (qtpylib.crossed_above(d["macd"], d["macd_signal"]) & (d["macd"] > 0)
                             & d["rsi_14"].between(50, 70, inclusive="neither")).astype(float)
    out["%-sig_bollinger"] = ((d["close"] < d["bb_lower"]) & (d["rsi_14"] < 30)
                              & (d["close"] > d["ema_200"])).astype(float)
    out["%-sig_donchian"] = ((d["close"] > d["dc_high_20"]) & (d["volume"] > 1.5 * d["vol_sma_20"])).astype(float)
    out["%-sig_squeeze"] = ((bbw_rank.shift(1) < 0.2) & qtpylib.crossed_above(d["close"], d["bb_upper"])
                            & (d["close"] > d["ema_50"])).astype(float)
    # Signal in den letzten 6 Stunden (Bots reagieren oft verzögert)
    for c in list(out.columns):
        out[c.replace("%-sig_", "%-sig6_")] = out[c].rolling(6, min_periods=1).max()
    return out


class KiTrader(KitbFreqaiBase):
    startup_candle_count = 500

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: dict,
                                       **kwargs) -> DataFrame:
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)
        dataframe["%-natr-period"] = ta.NATR(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-dist_ema-period"] = dataframe["close"] / ta.EMA(dataframe, timeperiod=period) - 1
        dataframe["%-rel_volume-period"] = dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe["%-pct_change"] = dataframe["close"].pct_change()
        dataframe["%-dist_ema200"] = dataframe["close"] / ta.EMA(dataframe, timeperiod=200) - 1
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe = super().feature_engineering_standard(dataframe, metadata, **kwargs)
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek / 6
        sig = rule_signals(dataframe)
        for c in sig.columns:
            dataframe[c] = sig[c]
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df = super().populate_entry_trend(df, metadata)
        # Grund ergänzen: welche Regel-Bots gerade (letzte 6 h) ein Signal hatten – Zusatzinformation der KI.
        sig = rule_signals(df)
        names = {"%-sig6_ema_cross": "EMA", "%-sig6_trend_adx": "ADX", "%-sig6_macd_rsi": "MACD",
                 "%-sig6_bollinger": "Bollinger", "%-sig6_donchian": "Donchian", "%-sig6_squeeze": "Squeeze"}
        entered = df["enter_long"] == 1
        if entered.any():
            bots = sig.loc[entered, list(names)].apply(
                lambda r: ", ".join(names[c] for c in names if r[c] > 0) or "keiner", axis=1)
            df.loc[entered, "enter_tag"] = df.loc[entered, "enter_tag"] + " · Bots: " + bots
        return df
