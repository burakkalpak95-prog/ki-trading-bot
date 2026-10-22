"""Deterministische, klar gekennzeichnete **synthetische** OHLCV-Daten für Offline-Tests.

Diese Daten sind KEINE Marktdaten. Sie dienen nur dazu, Pipeline, Validierung
und Backtest ohne Internetzugang reproduzierbar zu prüfen. Ergebnisse auf diesen
Daten sagen nichts über reale Profitabilität aus.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from kitb.data.storage import store_ohlcv

SYNTHETIC_MARKER = "SYNTHETIC_DATA_README.txt"

# Startpreise und Jahresvolatilität grob an BTC/ETH angelehnt (Annahme, nicht kalibriert).
DEFAULT_PARAMS = {
    "BTC/USDT": {"start": 30000.0, "annual_vol": 0.60, "annual_drift": 0.10, "base_volume": 1500.0},
    "ETH/USDT": {"start": 2000.0, "annual_vol": 0.75, "annual_drift": 0.05, "base_volume": 15000.0},
}


def generate_hourly(pair: str, start: str, end: str, seed: int) -> pd.DataFrame:
    params = DEFAULT_PARAMS[pair]
    dates = pd.date_range(start=start, end=end, freq="1h", tz="UTC", inclusive="left")
    n = len(dates)
    # Seed je Paar ableiten, damit Paare unabhängig, aber reproduzierbar sind.
    rng = np.random.default_rng([seed, sum(map(ord, pair))])
    dt = 1 / (365 * 24)
    sigma = params["annual_vol"] * np.sqrt(dt)
    mu = (params["annual_drift"] - 0.5 * params["annual_vol"] ** 2) * dt
    # Regimewechsel (ruhig/volatil), damit die Daten nicht völlig gleichförmig sind.
    regime = np.repeat(rng.choice([0.6, 1.0, 1.8], size=n // 500 + 1), 500)[:n]
    log_ret = mu + sigma * regime * rng.standard_normal(n)
    close = params["start"] * np.exp(np.cumsum(log_ret))
    open_ = np.concatenate([[params["start"]], close[:-1]])
    wick = np.abs(rng.standard_normal((2, n))) * sigma * regime * 0.5
    high = np.maximum(open_, close) * np.exp(wick[0])
    low = np.minimum(open_, close) * np.exp(-wick[1])
    volume = params["base_volume"] * regime * rng.lognormal(0, 0.4, n)
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low,
                         "close": close, "volume": volume})


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = {"4h": "4h", "1d": "1D"}[timeframe]
    out = (df.set_index("date")
             .resample(rule, label="left", closed="left")
             .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
             .dropna()
             .reset_index())
    return out


def write_synthetic_dataset(datadir: Path, pairs: list[str], timeframes: list[str],
                            start: str, end: str, seed: int = 42) -> dict[tuple[str, str], int]:
    """Schreibt synthetische Daten im Freqtrade-Format und legt eine Kennzeichnungsdatei ab."""
    datadir.mkdir(parents=True, exist_ok=True)
    (datadir / SYNTHETIC_MARKER).write_text(
        "ACHTUNG: Dieser Ordner enthält SYNTHETISCHE Kursdaten (Zufallsprozess, Seed "
        f"{seed}).\nKeine echten Marktdaten. Nur für Offline-Tests der Pipeline.\n",
        encoding="utf-8",
    )
    written = {}
    for pair in pairs:
        hourly = generate_hourly(pair, start, end, seed)
        for tf in timeframes:
            df = hourly if tf == "1h" else resample(hourly, tf)
            store_ohlcv(datadir, pair, tf, df)
            written[(pair, tf)] = len(df)
    return written


def is_synthetic_dir(datadir: Path) -> bool:
    return (datadir / SYNTHETIC_MARKER).exists()
