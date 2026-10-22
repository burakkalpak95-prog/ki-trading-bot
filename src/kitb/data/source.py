"""Erreichbarkeit öffentlicher Börsendaten prüfen und historische Daten über Freqtrade laden."""

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import ccxt

from kitb.retry import with_backoff

RETRYABLE = (ccxt.NetworkError, ccxt.DDoSProtection, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable)


def probe_exchange(exchange_id: str, pairs: list[str], *, retries: int, backoff_base_seconds: float,
                   timeout_ms: int) -> dict[str, Any]:
    """Öffentliche Endpunkte ohne API-Key abfragen: Märkte, Ticker, letzte 1h-Kerzen.

    Gibt immer ein Ergebnis-Dict zurück (``ok`` False bei Fehlern), wirft nicht.
    """
    result: dict[str, Any] = {"exchange": exchange_id, "ok": False, "pairs": {}}
    started = time.time()
    try:
        ex = getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": timeout_ms})
        call = lambda f: with_backoff(f, retries=retries, base_seconds=backoff_base_seconds,
                                      retry_on=RETRYABLE)
        markets = call(ex.load_markets)
        for pair in pairs:
            market = markets.get(pair)
            if not market or not market.get("spot") or not market.get("active", True):
                result["pairs"][pair] = {"available": False}
                continue
            ticker = call(lambda: ex.fetch_ticker(pair))
            candles = call(lambda: ex.fetch_ohlcv(pair, "1h", limit=3))
            result["pairs"][pair] = {
                "available": True,
                "last": ticker.get("last"),
                "ticker_time_utc": ticker.get("datetime"),
                "last_candle_open_ms": candles[-1][0] if candles else None,
            }
        result["ok"] = all(p.get("available") for p in result["pairs"].values())
    except Exception as e:  # Ergebnis wird protokolliert, nicht verschluckt
        result["error"] = f"{type(e).__name__}: {e}"
    result["duration_s"] = round(time.time() - started, 2)
    return result


def download_with_freqtrade(config_path: Path, exchange_id: str, pairs: list[str], timeframes: list[str],
                            datadir: Path, timerange: str | None, days: int | None) -> subprocess.CompletedProcess:
    """Ruft ``freqtrade download-data`` auf (Rate-Limit/Retry übernimmt Freqtrade/ccxt)."""
    cmd = [sys.executable, "-m", "freqtrade", "download-data", "-c", str(config_path),
           "--exchange", exchange_id, "--pairs", *pairs, "--timeframes", *timeframes,
           "--data-format-ohlcv", "parquet", "--datadir", str(datadir)]
    if timerange:
        cmd += ["--timerange", timerange]
    elif days:
        cmd += ["--days", str(days)]
    return subprocess.run(cmd, capture_output=True, text=True)
