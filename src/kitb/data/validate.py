"""Validierung von OHLCV-Daten (Freqtrade-Format: date, open, high, low, close, volume).

Fehler (ERROR) machen einen Datensatz für Backtests unbrauchbar, Warnungen
(WARNING) sind zu prüfen, blockieren aber nicht. Die Daten müssen **ungefüllt**
geladen werden (Freqtrade ``fill_missing=False``), sonst werden Lücken verdeckt.
"""

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from kitb.data.timeframes import to_timedelta

COLUMNS = ["date", "open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]
MAX_EXAMPLES = 5


@dataclass
class Issue:
    severity: str  # "ERROR" | "WARNING"
    check: str
    message: str
    count: int = 0
    examples: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    pair: str
    timeframe: str
    source: str
    rows: int
    first_candle: str | None
    last_candle: str | None
    fingerprint: str
    issues: list[Issue]

    @property
    def ok(self) -> bool:
        return not any(i.severity == "ERROR" for i in self.issues)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ok"] = self.ok
        return d


def fingerprint(df: pd.DataFrame) -> str:
    """Stabiler Hash über den Inhalt – macht Backtests auf exakt diesen Daten nachvollziehbar."""
    if df.empty:
        return hashlib.sha256(b"").hexdigest()
    hashed = pd.util.hash_pandas_object(df[COLUMNS], index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _fmt(ts: Any) -> str:
    return pd.Timestamp(ts).isoformat()


def validate_ohlcv(
    df: pd.DataFrame,
    pair: str,
    timeframe: str,
    source: str,
    *,
    now: pd.Timestamp | None = None,
    max_abs_log_return: float = 0.3,
    max_staleness_candles: int | None = None,
) -> ValidationReport:
    """Prüft einen OHLCV-DataFrame. ``now`` aktiviert die Prüfungen auf
    unvollständige und veraltete Kerzen (bei rein historischen Analysen weglassen)."""
    issues: list[Issue] = []

    missing_cols = [c for c in COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(Issue("ERROR", "schema", f"Fehlende Spalten: {missing_cols}"))
        return ValidationReport(pair, timeframe, source, len(df), None, None, "", issues)

    if df.empty:
        issues.append(Issue("ERROR", "empty", "Keine Daten vorhanden."))
        return ValidationReport(pair, timeframe, source, 0, None, None, fingerprint(df), issues)

    dates = df["date"]
    if not isinstance(dates.dtype, pd.DatetimeTZDtype) or str(dates.dt.tz) != "UTC":
        issues.append(Issue("ERROR", "timezone", f"Zeitstempel sind nicht UTC (dtype={dates.dtype})."))
        dates = pd.to_datetime(dates, utc=True)

    tf = to_timedelta(timeframe)

    if not dates.is_monotonic_increasing:
        issues.append(Issue("ERROR", "order", "Zeitstempel sind nicht aufsteigend sortiert."))

    dup_mask = dates.duplicated(keep=False)
    if dup_mask.any():
        dups = dates[dup_mask].drop_duplicates()
        issues.append(Issue("ERROR", "duplicates", "Doppelte Zeitstempel.", int(dup_mask.sum()),
                            [_fmt(t) for t in dups[:MAX_EXAMPLES]]))

    epoch = pd.Timestamp(0, tz="UTC")
    misaligned = ((dates - epoch) % tf) != pd.Timedelta(0)
    if misaligned.any():
        issues.append(Issue("ERROR", "alignment", f"Zeitstempel nicht auf {timeframe}-Raster.",
                            int(misaligned.sum()), [_fmt(t) for t in dates[misaligned][:MAX_EXAMPLES]]))

    unique_sorted = dates.drop_duplicates().sort_values()
    deltas = unique_sorted.diff().iloc[1:]
    gap_mask = deltas > tf
    if gap_mask.any():
        missing = int(((deltas[gap_mask] / tf) - 1).sum())
        starts = unique_sorted.shift(1).iloc[1:][gap_mask]
        examples = [f"nach {_fmt(s)}: {int(d / tf) - 1} Kerzen" for s, d in
                    zip(starts[:MAX_EXAMPLES], deltas[gap_mask][:MAX_EXAMPLES])]
        issues.append(Issue("WARNING", "gaps", f"{missing} fehlende Kerzen in {int(gap_mask.sum())} Lücken.",
                            missing, examples))

    nan_mask = df[PRICE_COLUMNS + ["volume"]].isna().any(axis=1)
    if nan_mask.any():
        issues.append(Issue("ERROR", "nan", "Leere Werte (NaN).", int(nan_mask.sum()),
                            [_fmt(t) for t in dates[nan_mask][:MAX_EXAMPLES]]))

    nonpos = (df[PRICE_COLUMNS] <= 0).any(axis=1)
    if nonpos.any():
        issues.append(Issue("ERROR", "non_positive_price", "Preise <= 0.", int(nonpos.sum()),
                            [_fmt(t) for t in dates[nonpos][:MAX_EXAMPLES]]))

    neg_vol = df["volume"] < 0
    if neg_vol.any():
        issues.append(Issue("ERROR", "negative_volume", "Negatives Volumen.", int(neg_vol.sum()),
                            [_fmt(t) for t in dates[neg_vol][:MAX_EXAMPLES]]))

    zero_vol = df["volume"] == 0
    if zero_vol.any():
        issues.append(Issue("WARNING", "zero_volume", "Kerzen ohne Volumen (evtl. aufgefüllte Lücken).",
                            int(zero_vol.sum()), [_fmt(t) for t in dates[zero_vol][:MAX_EXAMPLES]]))

    hi, lo, op, cl = df["high"], df["low"], df["open"], df["close"]
    bad_ohlc = (hi < np.maximum(op, cl)) | (lo > np.minimum(op, cl)) | (hi < lo)
    if bad_ohlc.any():
        issues.append(Issue("ERROR", "ohlc_consistency", "High/Low widersprechen Open/Close.",
                            int(bad_ohlc.sum()), [_fmt(t) for t in dates[bad_ohlc][:MAX_EXAMPLES]]))

    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.log(cl / cl.shift(1)).abs()
    outliers = log_ret > max_abs_log_return
    if outliers.any():
        issues.append(Issue("WARNING", "outlier_return",
                            f"Kursänderung > {max_abs_log_return:.0%} (log) zur Vorkerze – prüfen.",
                            int(outliers.sum()), [_fmt(t) for t in dates[outliers][:MAX_EXAMPLES]]))

    last_open = dates.max()
    if now is not None:
        now = pd.Timestamp(now).tz_convert("UTC")
        last_close = last_open + tf
        if last_close > now:
            issues.append(Issue("ERROR", "incomplete_candle",
                                "Letzte Kerze ist noch nicht abgeschlossen und darf nicht verwendet werden.",
                                1, [_fmt(last_open)]))
        if max_staleness_candles is not None and now - last_close > tf * max_staleness_candles:
            issues.append(Issue("WARNING", "stale",
                                f"Daten veraltet: letzte Kerze schloss {_fmt(last_close)}.", 1, [_fmt(last_close)]))

    return ValidationReport(pair, timeframe, source, len(df), _fmt(dates.min()), _fmt(last_open),
                            fingerprint(df), issues)


def drop_incomplete_candles(df: pd.DataFrame, timeframe: str, now: pd.Timestamp) -> pd.DataFrame:
    """Entfernt Kerzen, die zum Zeitpunkt ``now`` noch nicht geschlossen sind."""
    tf = to_timedelta(timeframe)
    return df[df["date"] + tf <= pd.Timestamp(now).tz_convert("UTC")].reset_index(drop=True)
