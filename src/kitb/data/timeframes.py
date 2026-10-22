import pandas as pd

_UNITS = {"m": "min", "h": "h", "d": "D", "w": "W"}


def to_timedelta(timeframe: str) -> pd.Timedelta:
    """'1h' -> Timedelta(1h). Unterstützt m, h, d, w."""
    value, unit = int(timeframe[:-1]), timeframe[-1]
    if unit not in _UNITS:
        raise ValueError(f"Unbekanntes Zeitintervall: {timeframe}")
    return pd.Timedelta(value, _UNITS[unit])
