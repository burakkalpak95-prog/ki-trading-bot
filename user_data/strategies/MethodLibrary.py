"""Methoden-Bibliothek: bekannte Trading-Methoden als Kandidaten für Datensammler-Bots (Burak, 23.10.2026).

Auftrag: „alle Trading-Methoden abdecken, die in der Vergangenheit mindestens 60 % erfolgreiche Trades hatten“.
Jede Methode hier ist nur ein Kandidat. Bot wird sie erst, wenn ``kitb methods scan`` auf echten Daten ab 2022
(16 Coins, mit Gebühren) mindestens 60 % Treffer UND Gewinn nach Gebühren zeigt – in einem frühen und in einem späteren
Zeitraum (sonst ist es Zufall/Überanpassung). Alle nur Long (Spot ohne Hebel), alle mit Stoploss und Risikomodul.
Alle Indikatoren nur aus vergangenen und aktuellen Kerzen.
"""

import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IStrategy
from kitb.risk import RiskManagedMixin


class _Method(RiskManagedMixin, IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False
    minimal_roi = {"0": 100.0}
    stoploss = -0.06
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 300
    order_types = {"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False}
    TAG = "methode"

    def entry(self, d: DataFrame):  # -> bool-Serie
        raise NotImplementedError

    def exit(self, d: DataFrame):
        raise NotImplementedError

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        d = dataframe
        d["ema9"], d["ema10"], d["ema20"], d["ema21"] = (ta.EMA(d, timeperiod=p) for p in (9, 10, 20, 21))
        d["ema50"], d["ema55"], d["ema200"] = (ta.EMA(d, timeperiod=p) for p in (50, 55, 200))
        d["sma5"], d["sma50"], d["sma200"] = (ta.SMA(d, timeperiod=p) for p in (5, 50, 200))
        d["rsi2"], d["rsi14"] = ta.RSI(d, timeperiod=2), ta.RSI(d, timeperiod=14)
        d["atr"] = ta.ATR(d, timeperiod=14)
        d["adx"] = ta.ADX(d, timeperiod=14)
        return d

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[self.entry(dataframe) & (dataframe["volume"] > 0), ["enter_long", "enter_tag"]] = (1, self.TAG)
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[self.exit(dataframe) & (dataframe["volume"] > 0), ["exit_long", "exit_tag"]] = (1, "ziel")
        return dataframe


# --- Rückkehr zum Mittelwert (meist hohe Trefferquote, kleine Gewinne) ---

class MethodRsi2(_Method):
    """Connors RSI(2): im Aufwärtstrend (über SMA 200) bei RSI(2) < 10 kaufen, über SMA 5 verkaufen."""
    TAG = "rsi2"

    def entry(self, d):
        return (d["close"] > d["sma200"]) & (d["rsi2"] < 10)

    def exit(self, d):
        return d["close"] > d["sma5"]


class MethodWilliamsR(_Method):
    """Williams %R(14) unter −90 im Aufwärtstrend kaufen, über −20 verkaufen."""
    TAG = "williams_r"

    def populate_indicators(self, dataframe, metadata):
        d = super().populate_indicators(dataframe, metadata)
        d["willr"] = ta.WILLR(d, timeperiod=14)
        return d

    def entry(self, d):
        return (d["close"] > d["ema200"]) & (d["willr"] < -90)

    def exit(self, d):
        return d["willr"] > -20


class MethodCci(_Method):
    """CCI(20) unter −150 im Aufwärtstrend kaufen, über 0 verkaufen."""
    TAG = "cci"

    def populate_indicators(self, dataframe, metadata):
        d = super().populate_indicators(dataframe, metadata)
        d["cci"] = ta.CCI(d, timeperiod=20)
        return d

    def entry(self, d):
        return (d["close"] > d["ema200"]) & (d["cci"] < -150)

    def exit(self, d):
        return d["cci"] > 0


class MethodMfi(_Method):
    """Money Flow Index(14) unter 20 (Ausverkauf mit Volumen) kaufen, über 60 verkaufen."""
    TAG = "mfi"

    def populate_indicators(self, dataframe, metadata):
        d = super().populate_indicators(dataframe, metadata)
        d["mfi"] = ta.MFI(d, timeperiod=14)
        return d

    def entry(self, d):
        return (d["close"] > d["ema200"]) & (d["mfi"] < 20)

    def exit(self, d):
        return d["mfi"] > 60


class MethodStochRsi(_Method):
    """Stoch-RSI: %K kreuzt %D unter 20 nach oben (im Aufwärtstrend), Ausstieg bei %K > 80."""
    TAG = "stoch_rsi"

    def populate_indicators(self, dataframe, metadata):
        d = super().populate_indicators(dataframe, metadata)
        r = d["rsi14"]
        k = (r - r.rolling(14).min()) / (r.rolling(14).max() - r.rolling(14).min()).replace(0, np.nan) * 100
        d["srsi_k"] = k.rolling(3).mean()
        d["srsi_d"] = d["srsi_k"].rolling(3).mean()
        return d

    def entry(self, d):
        return (d["close"] > d["ema200"]) & (d["srsi_k"] < 20) & qtpylib.crossed_above(d["srsi_k"], d["srsi_d"])

    def exit(self, d):
        return d["srsi_k"] > 80


class MethodZScore(_Method):
    """Z-Score des Kurses (50 Kerzen) unter −2 kaufen, bei Rückkehr zum Mittel (Z > 0) verkaufen."""
    TAG = "zscore"

    def populate_indicators(self, dataframe, metadata):
        d = super().populate_indicators(dataframe, metadata)
        m, s = d["close"].rolling(50).mean(), d["close"].rolling(50).std()
        d["z"] = (d["close"] - m) / s.replace(0, np.nan)
        return d

    def entry(self, d):
        return (d["close"] > d["ema200"]) & (d["z"] < -2)

    def exit(self, d):
        return d["z"] > 0


class MethodBollingerBounce(_Method):
    """Unteres Bollinger-Band (20, 2) unterschritten im Aufwärtstrend, Ausstieg an der Mittellinie (ohne RSI-Filter)."""
    TAG = "bb_bounce"

    def populate_indicators(self, dataframe, metadata):
        d = super().populate_indicators(dataframe, metadata)
        bb = qtpylib.bollinger_bands(qtpylib.typical_price(d), window=20, stds=2)
        d["bb_lower"], d["bb_mid"] = bb["lower"], bb["mid"]
        return d

    def entry(self, d):
        return (d["close"] > d["ema200"]) & (d["close"] < d["bb_lower"])

    def exit(self, d):
        return d["close"] > d["bb_mid"]


class MethodThreeDown(_Method):
    """Drei fallende Schlusskurse in Folge im Aufwärtstrend kaufen, beim ersten höheren Schluss verkaufen."""
    TAG = "three_down"

    def entry(self, d):
        down = d["close"] < d["close"].shift(1)
        return (d["close"] > d["sma200"]) & down & down.shift(1) & down.shift(2)

    def exit(self, d):
        return d["close"] > d["close"].shift(1)


class MethodVwapReversion(_Method):
    """Kurs mehr als 2 ATR unter dem gleitenden VWAP (96 Kerzen) kaufen, am VWAP verkaufen."""
    TAG = "vwap"

    def populate_indicators(self, dataframe, metadata):
        d = super().populate_indicators(dataframe, metadata)
        tp = qtpylib.typical_price(d)
        d["vwap"] = (tp * d["volume"]).rolling(96).sum() / d["volume"].rolling(96).sum().replace(0, np.nan)
        return d

    def entry(self, d):
        return (d["close"] > d["ema200"]) & (d["close"] < d["vwap"] - 2 * d["atr"])

    def exit(self, d):
        return d["close"] > d["vwap"]


# --- Trendfolge / Ausbruch (meist niedrigere Trefferquote, größere Gewinne) ---

class MethodKeltner(_Method):
    """Schluss über dem oberen Keltner-Kanal (EMA 20 + 2 ATR), Ausstieg unter EMA 20."""
    TAG = "keltner"

    def entry(self, d):
        return qtpylib.crossed_above(d["close"], d["ema20"] + 2 * d["atr"])

    def exit(self, d):
        return d["close"] < d["ema20"]


class MethodSupertrend(_Method):
    """Supertrend (10, 3): Wechsel auf Aufwärtstrend kaufen, Wechsel auf Abwärtstrend verkaufen."""
    TAG = "supertrend"

    def populate_indicators(self, dataframe, metadata):
        d = super().populate_indicators(dataframe, metadata)
        atr = ta.ATR(d, timeperiod=10).to_numpy()
        hl2 = ((d["high"] + d["low"]) / 2).to_numpy()
        close = d["close"].to_numpy()
        up, dn = hl2 - 3 * atr, hl2 + 3 * atr
        trend = np.ones(len(d))
        for i in range(1, len(d)):
            if not np.isfinite(atr[i - 1]):
                continue
            if close[i - 1] > up[i - 1]:
                up[i] = max(up[i], up[i - 1])
            if close[i - 1] < dn[i - 1]:
                dn[i] = min(dn[i], dn[i - 1])
            trend[i] = 1 if close[i] > dn[i - 1] else (-1 if close[i] < up[i - 1] else trend[i - 1])
        d["st_trend"] = trend
        return d

    def entry(self, d):
        return (d["st_trend"] == 1) & (d["st_trend"].shift(1) == -1)

    def exit(self, d):
        return d["st_trend"] == -1


class MethodIchimoku(_Method):
    """Ichimoku: Kurs über der Wolke und Tenkan kreuzt Kijun nach oben; Ausstieg unter Kijun."""
    TAG = "ichimoku"

    def populate_indicators(self, dataframe, metadata):
        d = super().populate_indicators(dataframe, metadata)
        mid = lambda n: (d["high"].rolling(n).max() + d["low"].rolling(n).min()) / 2  # noqa: E731
        d["tenkan"], d["kijun"] = mid(9), mid(26)
        d["span_a"] = ((d["tenkan"] + d["kijun"]) / 2).shift(26)
        d["span_b"] = mid(52).shift(26)
        return d

    def entry(self, d):
        cloud_top = d[["span_a", "span_b"]].max(axis=1)
        return (d["close"] > cloud_top) & qtpylib.crossed_above(d["tenkan"], d["kijun"])

    def exit(self, d):
        return d["close"] < d["kijun"]


class MethodParabolicSar(_Method):
    """Parabolic SAR springt unter den Kurs bei ADX > 20; Ausstieg, wenn er wieder darüber springt."""
    TAG = "psar"

    def populate_indicators(self, dataframe, metadata):
        d = super().populate_indicators(dataframe, metadata)
        d["sar"] = ta.SAR(d, acceleration=0.02, maximum=0.2)
        return d

    def entry(self, d):
        return (d["sar"] < d["close"]) & (d["sar"].shift(1) > d["close"].shift(1)) & (d["adx"] > 20)

    def exit(self, d):
        return d["sar"] > d["close"]


class MethodTripleEma(_Method):
    """EMA 9 > 21 > 55 neu ausgerichtet kaufen, EMA 9 unter 21 verkaufen."""
    TAG = "triple_ema"

    def entry(self, d):
        aligned = (d["ema9"] > d["ema21"]) & (d["ema21"] > d["ema55"])
        return aligned & ~aligned.shift(1, fill_value=False)

    def exit(self, d):
        return d["ema9"] < d["ema21"]


class MethodHeikinAshi(_Method):
    """Heikin-Ashi: nach roter Kerze zwei grüne ohne unteren Docht (starker Schub) kaufen, erste rote verkaufen."""
    TAG = "heikin_ashi"

    def populate_indicators(self, dataframe, metadata):
        d = super().populate_indicators(dataframe, metadata)
        ha = qtpylib.heikinashi(d)
        d["ha_open"], d["ha_close"], d["ha_low"] = ha["open"], ha["close"], ha["low"]
        return d

    def entry(self, d):
        green = d["ha_close"] > d["ha_open"]
        strong = green & (d["ha_low"] >= d["ha_open"])
        return strong & strong.shift(1, fill_value=False) & ~green.shift(2, fill_value=True) & (d["close"] > d["ema50"])

    def exit(self, d):
        return d["ha_close"] < d["ha_open"]


class MethodVolumeBreakout(_Method):
    """Volumen > 3× Durchschnitt und Schluss über dem 20-Kerzen-Hoch; Ausstieg unter EMA 10."""
    TAG = "volume_breakout"

    def entry(self, d):
        return (d["volume"] > 3 * d["volume"].rolling(20).mean()) & (d["close"] > d["high"].shift(1).rolling(20).max())

    def exit(self, d):
        return d["close"] < d["ema10"]


class MethodTurtle(_Method):
    """Turtle: Schluss über dem 55-Kerzen-Hoch kaufen, unter dem 20-Kerzen-Tief verkaufen."""
    TAG = "turtle"
    stoploss = -0.10

    def entry(self, d):
        return d["close"] > d["high"].shift(1).rolling(55).max()

    def exit(self, d):
        return d["close"] < d["low"].shift(1).rolling(20).min()


class MethodInsideBar(_Method):
    """Innenstab-Ausbruch: Kerze innerhalb der Vorkerze, dann Schluss über deren Hoch (über EMA 50); Ausstieg < EMA 20."""
    TAG = "inside_bar"

    def entry(self, d):
        inside = (d["high"].shift(1) < d["high"].shift(2)) & (d["low"].shift(1) > d["low"].shift(2))
        return inside & (d["close"] > d["high"].shift(2)) & (d["close"] > d["ema50"])

    def exit(self, d):
        return d["close"] < d["ema20"]


class MethodGoldenCross(_Method):
    """Golden Cross (SMA 50 über SMA 200) kaufen, Death Cross verkaufen."""
    TAG = "golden_cross"
    stoploss = -0.15

    def entry(self, d):
        return qtpylib.crossed_above(d["sma50"], d["sma200"])

    def exit(self, d):
        return qtpylib.crossed_below(d["sma50"], d["sma200"])


METHODS = [MethodRsi2, MethodWilliamsR, MethodCci, MethodMfi, MethodStochRsi, MethodZScore, MethodBollingerBounce,
           MethodThreeDown, MethodVwapReversion, MethodKeltner, MethodSupertrend, MethodIchimoku, MethodParabolicSar,
           MethodTripleEma, MethodHeikinAshi, MethodVolumeBreakout, MethodTurtle, MethodInsideBar, MethodGoldenCross]
