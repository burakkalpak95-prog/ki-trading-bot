"""Gemeinsame, kausale Indikatoren für alle Strategien (nur vergangene und aktuelle Kerze).

Zentral gehalten, damit Strategien dieselben Berechnungen nutzen und Indikatoren
nicht mehrfach mit leicht unterschiedlichen Parametern entstehen.
"""

import pandas as pd
import talib.abstract as ta

# Einheitliche Aufwärmphase aller 1h-Regel-Strategien. Freqtrade lädt bei --strategy-list den Vorlauf für das
# Maximum aller Strategien; EMAs hängen vom Startpunkt ab. Mit einem gemeinsamen Wert liefert jede Strategie
# unabhängig von der Zusammenstellung eines Laufs dieselben Signale.
STARTUP_CANDLES_1H = 520
# Dasselbe für Tagesstrategien (längster Indikator: SMA 200 Tage).
STARTUP_CANDLES_1D = 200


def add_ema(df: pd.DataFrame, *periods: int) -> pd.DataFrame:
    for p in periods:
        df[f"ema_{p}"] = ta.EMA(df, timeperiod=p)
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df[f"rsi_{period}"] = ta.RSI(df, timeperiod=period)
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd = ta.MACD(df, fastperiod=fast, slowperiod=slow, signalperiod=signal)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd["macd"], macd["macdsignal"], macd["macdhist"]
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df[f"atr_{period}"] = ta.ATR(df, timeperiod=period)
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df[f"adx_{period}"] = ta.ADX(df, timeperiod=period)
    return df


def add_bollinger(df: pd.DataFrame, period: int = 20, stds: float = 2.0) -> pd.DataFrame:
    bb = ta.BBANDS(df, timeperiod=period, nbdevup=stds, nbdevdn=stds)
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = bb["upperband"], bb["middleband"], bb["lowerband"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    return df


def add_donchian(df: pd.DataFrame, entry_period: int = 20, exit_period: int = 10) -> pd.DataFrame:
    """Kanal aus den **vorherigen** Kerzen (shift(1)), damit die aktuelle Kerze den Kanal nicht selbst setzt."""
    df[f"dc_high_{entry_period}"] = df["high"].rolling(entry_period).max().shift(1)
    df[f"dc_low_{exit_period}"] = df["low"].rolling(exit_period).min().shift(1)
    return df


def add_volume_sma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df[f"vol_sma_{period}"] = df["volume"].rolling(period).mean()
    return df


def rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Rang des aktuellen Werts innerhalb der letzten ``window`` Werte (0..1), nur Vergangenheit."""
    return series.rolling(window).rank(pct=True)


def add_sma(df: pd.DataFrame, *periods: int) -> pd.DataFrame:
    for p in periods:
        df[f"sma_{p}"] = ta.SMA(df, timeperiod=p)
    return df
