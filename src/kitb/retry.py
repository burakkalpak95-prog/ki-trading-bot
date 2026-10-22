"""Wiederholungslogik mit exponentiellem Backoff für eigene Netzwerkaufrufe.

Freqtrade selbst hat eigene Retry- und Rate-Limit-Logik (ccxt ``enableRateLimit``);
diese Hilfe wird nur für die Quellen-Prüfung (``kitb probe``) verwendet.
"""

import logging
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
log = logging.getLogger(__name__)


def with_backoff(fn: Callable[[], T], *, retries: int, base_seconds: float,
                 retry_on: tuple[type[BaseException], ...],
                 sleep: Callable[[float], None] = time.sleep) -> T:
    attempt = 0
    while True:
        try:
            return fn()
        except retry_on as e:
            if attempt >= retries:
                raise
            delay = base_seconds * (2 ** attempt)
            log.warning("Versuch %d fehlgeschlagen (%s: %s) – neuer Versuch in %.1fs",
                        attempt + 1, type(e).__name__, e, delay)
            sleep(delay)
            attempt += 1
