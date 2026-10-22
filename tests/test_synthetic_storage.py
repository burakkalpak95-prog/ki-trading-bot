from kitb.data.storage import load_ohlcv_raw
from kitb.data.synthetic import generate_hourly, is_synthetic_dir, write_synthetic_dataset
from kitb.data.validate import fingerprint, validate_ohlcv


def test_generation_is_deterministic():
    a = generate_hourly("BTC/USDT", "2024-01-01", "2024-02-01", seed=7)
    b = generate_hourly("BTC/USDT", "2024-01-01", "2024-02-01", seed=7)
    c = generate_hourly("BTC/USDT", "2024-01-01", "2024-02-01", seed=8)
    assert fingerprint(a) == fingerprint(b) != fingerprint(c)


def test_write_load_validate_roundtrip(tmp_path):
    written = write_synthetic_dataset(tmp_path, ["BTC/USDT", "ETH/USDT"], ["1h", "1d"],
                                      "2024-01-01", "2024-03-01", seed=1)
    assert is_synthetic_dir(tmp_path)
    assert written[("BTC/USDT", "1h")] == 60 * 24 and written[("BTC/USDT", "1d")] == 60
    for pair in ("BTC/USDT", "ETH/USDT"):
        for tf in ("1h", "1d"):
            df = load_ohlcv_raw(tmp_path, pair, tf)
            assert len(df) == written[(pair, tf)]
            rep = validate_ohlcv(df, pair, tf, "synthetic")
            assert rep.ok, rep.issues
    assert sorted(p.name for p in tmp_path.glob("*.parquet")) == [
        "BTC_USDT-1d.parquet", "BTC_USDT-1h.parquet", "ETH_USDT-1d.parquet", "ETH_USDT-1h.parquet"]


def test_trim_incomplete_rewrites_store(tmp_path, monkeypatch):
    import pandas as pd

    from kitb import cli

    write_synthetic_dataset(tmp_path, ["BTC/USDT"], ["1h"], "2024-01-01", "2024-01-03", seed=1)
    last_open = load_ohlcv_raw(tmp_path, "BTC/USDT", "1h")["date"].iloc[-1]
    fake_now = last_open + pd.Timedelta(minutes=20)
    monkeypatch.setattr(cli.pd.Timestamp, "now", classmethod(lambda cls, tz=None: fake_now))
    cli._trim_incomplete(tmp_path, ["BTC/USDT"], ["1h"])
    assert load_ohlcv_raw(tmp_path, "BTC/USDT", "1h")["date"].iloc[-1] == last_open - pd.Timedelta(hours=1)
