"""Lesen/Schreiben von OHLCV-Daten über die Datahandler von Freqtrade.

Damit liegen eigene Daten (z. B. synthetische Offline-Daten) im selben Format
und Pfadschema wie mit ``freqtrade download-data`` geladene Daten.
"""

from pathlib import Path

import pandas as pd
from freqtrade.data.history import get_datahandler
from freqtrade.enums import CandleType

DATA_FORMAT = "parquet"


def store_ohlcv(datadir: Path, pair: str, timeframe: str, df: pd.DataFrame) -> None:
    datadir.mkdir(parents=True, exist_ok=True)
    get_datahandler(datadir, DATA_FORMAT).ohlcv_store(pair, timeframe, df, CandleType.SPOT)


def load_ohlcv_raw(datadir: Path, pair: str, timeframe: str) -> pd.DataFrame:
    """Lädt Daten **ohne** Auffüllen von Lücken (für die Validierung)."""
    return get_datahandler(datadir, DATA_FORMAT).ohlcv_load(
        pair, timeframe, CandleType.SPOT, fill_missing=False, drop_incomplete=False, warn_no_data=False
    )
