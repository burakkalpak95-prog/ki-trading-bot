import numpy as np
import pandas as pd

from kitb.data.validate import drop_incomplete_candles, fingerprint, validate_ohlcv


def checks(report, severity=None):
    return {i.check for i in report.issues if severity is None or i.severity == severity}


def test_clean_data_passes(clean_df):
    rep = validate_ohlcv(clean_df, "BTC/USDT", "1h", "test")
    assert rep.ok and rep.issues == [] and rep.rows == 48


def test_duplicates_detected(clean_df):
    df = pd.concat([clean_df, clean_df.iloc[[5]]]).sort_values("date").reset_index(drop=True)
    rep = validate_ohlcv(df, "BTC/USDT", "1h", "test")
    assert not rep.ok and "duplicates" in checks(rep, "ERROR")


def test_gaps_detected(clean_df):
    df = clean_df.drop(index=[10, 11, 12]).reset_index(drop=True)
    rep = validate_ohlcv(df, "BTC/USDT", "1h", "test")
    gap = next(i for i in rep.issues if i.check == "gaps")
    assert gap.count == 3 and rep.ok  # Lücken sind Warnung, kein Abbruch


def test_unsorted_detected(clean_df):
    df = clean_df.iloc[::-1].reset_index(drop=True)
    assert "order" in checks(validate_ohlcv(df, "X", "1h", "t"), "ERROR")


def test_non_utc_detected(clean_df):
    df = clean_df.copy()
    df["date"] = df["date"].dt.tz_localize(None)
    assert "timezone" in checks(validate_ohlcv(df, "X", "1h", "t"), "ERROR")


def test_misaligned_detected(clean_df):
    df = clean_df.copy()
    df.loc[3, "date"] = df.loc[3, "date"] + pd.Timedelta(minutes=7)
    assert "alignment" in checks(validate_ohlcv(df, "X", "1h", "t"), "ERROR")


def test_bad_values_detected(clean_df):
    df = clean_df.copy()
    df.loc[1, "close"] = -5
    df.loc[2, "high"] = df.loc[2, "low"] - 1
    df.loc[3, "volume"] = -1
    df.loc[4, "open"] = np.nan
    found = checks(validate_ohlcv(df, "X", "1h", "t"), "ERROR")
    assert {"non_positive_price", "ohlc_consistency", "negative_volume", "nan"} <= found


def test_outlier_and_zero_volume_are_warnings(clean_df):
    df = clean_df.copy()
    df.loc[20, ["open", "high", "low", "close"]] = [500, 1000, 400, 1000]
    df.loc[21, "volume"] = 0
    rep = validate_ohlcv(df, "X", "1h", "t", max_abs_log_return=0.3)
    assert {"outlier_return", "zero_volume"} <= checks(rep, "WARNING")


def test_incomplete_last_candle_is_error(clean_df):
    last_open = clean_df["date"].iloc[-1]
    now = last_open + pd.Timedelta(minutes=30)
    rep = validate_ohlcv(clean_df, "X", "1h", "t", now=now)
    assert "incomplete_candle" in checks(rep, "ERROR")
    trimmed = drop_incomplete_candles(clean_df, "1h", now)
    assert len(trimmed) == 47
    assert validate_ohlcv(trimmed, "X", "1h", "t", now=now).ok


def test_stale_data_warning(clean_df):
    now = clean_df["date"].iloc[-1] + pd.Timedelta(hours=10)
    rep = validate_ohlcv(clean_df, "X", "1h", "t", now=now, max_staleness_candles=3)
    assert "stale" in checks(rep, "WARNING")


def test_missing_columns_and_empty():
    assert not validate_ohlcv(pd.DataFrame({"date": []}), "X", "1h", "t").ok
    empty = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    assert "empty" in checks(validate_ohlcv(empty, "X", "1h", "t"))


def test_fingerprint_stable_and_sensitive(clean_df):
    assert fingerprint(clean_df) == fingerprint(clean_df.copy())
    other = clean_df.copy()
    other.loc[0, "close"] += 0.01
    assert fingerprint(other) != fingerprint(clean_df)
