"""Offline-Modus: statische Marktdefinitionen statt ``load_markets()``.

Freqtrade lädt beim Start die Marktliste der Börse – auch im Backtest. Ohne
Internetzugang (z. B. in CI oder einer gesperrten Cloud-Umgebung) ersetzt dieser
Kontextmanager das Laden durch feste Spot-Markteinträge für BTC/USDT und ETH/USDT.

Nur für Backtests auf **synthetischen** Daten gedacht. Präzisionen und
Mindestgrößen sind Annahmen und können von der echten Börse abweichen.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from unittest import mock

STATIC_SPOT_MARKETS = {
    "BTC/USDT": {"base": "BTC", "price_tick": 0.01, "amount_step": 0.00001},
    "ETH/USDT": {"base": "ETH", "price_tick": 0.01, "amount_step": 0.0001},
}


def _market(symbol: str, spec: dict) -> dict:
    return {
        "id": symbol.replace("/", ""), "symbol": symbol, "base": spec["base"], "quote": "USDT",
        "baseId": spec["base"], "quoteId": "USDT", "settle": None, "settleId": None,
        "type": "spot", "spot": True, "margin": False, "swap": False, "future": False,
        "option": False, "contract": False, "linear": None, "inverse": None, "contractSize": None,
        "active": True, "taker": 0.001, "maker": 0.001,
        "precision": {"amount": spec["amount_step"], "price": spec["price_tick"]},
        "limits": {"amount": {"min": spec["amount_step"], "max": 9000.0},
                   "price": {"min": spec["price_tick"], "max": 1_000_000.0},
                   "cost": {"min": 5.0, "max": None}, "leverage": {"min": None, "max": None}},
        "info": {"kitb_offline": True},
    }


@contextmanager
def offline_markets() -> Iterator[None]:
    from freqtrade.exchange.exchange import Exchange

    markets = {s: _market(s, spec) for s, spec in STATIC_SPOT_MARKETS.items()}

    def fake_reload(self, force: bool = False, *, load_leverage_tiers: bool = True) -> None:
        self._api_async.set_markets(markets)
        self._api.set_markets(markets)
        self._markets = self._api.markets
        self._last_markets_refresh = 1

    with mock.patch.object(Exchange, "reload_markets", fake_reload):
        yield
