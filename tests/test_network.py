"""Echte Netzwerktests – standardmäßig übersprungen. Ausführen mit: pytest -m network"""

import pytest

from kitb.data.source import probe_exchange


@pytest.mark.network
@pytest.mark.parametrize("exchange", ["binance"])
def test_public_market_data_reachable(exchange):
    res = probe_exchange(exchange, ["BTC/USDT", "ETH/USDT"], retries=2, backoff_base_seconds=1, timeout_ms=15000)
    assert res["ok"], res
