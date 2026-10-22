"""Sicherheitsprüfungen für Phase 1.

In Phase 1 sind ausschließlich die Modi ``data`` und ``backtest`` erlaubt.
Paper-Trading (Freqtrade ``trade`` mit dry_run) und Live-Trading sind gesperrt –
unabhängig von Umgebungsvariablen oder Konfigurationswerten. Eine Freigabe
erfordert eine Codeänderung in einer späteren, ausdrücklich freigegebenen Phase.
"""

from collections.abc import Iterable
from typing import Any

ALLOWED_RUN_MODES = frozenset({"data", "backtest"})
BLOCKED_FREQTRADE_COMMANDS = frozenset({"trade", "webserver"})
SECRET_KEYS = ("key", "secret", "password", "uid", "walletAddress", "privateKey")


class SafetyError(RuntimeError):
    """Konfiguration oder Aufruf verletzt eine Phase-1-Sicherheitsregel."""


def assert_run_mode_allowed(mode: str) -> None:
    if mode not in ALLOWED_RUN_MODES:
        raise SafetyError(
            f"Modus '{mode}' ist in Phase 1 gesperrt. Erlaubt: {sorted(ALLOWED_RUN_MODES)}. "
            "Paper- und Live-Trading werden erst nach ausdrücklicher Freigabe implementiert."
        )


def assert_freqtrade_command_allowed(command: str) -> None:
    if command in BLOCKED_FREQTRADE_COMMANDS:
        raise SafetyError(f"Freqtrade-Befehl '{command}' ist in Phase 1 gesperrt.")


def check_freqtrade_config(
    config: dict[str, Any],
    allowed_exchanges: Iterable[str],
    allowed_pairs: Iterable[str],
) -> list[str]:
    """Gibt eine Liste aller Verstöße zurück (leer = sicher)."""
    problems: list[str] = []

    if config.get("dry_run") is not True:
        problems.append("dry_run muss explizit true sein.")
    if config.get("trading_mode", "spot") != "spot":
        problems.append("trading_mode muss 'spot' sein (keine Futures/Margin).")
    if config.get("margin_mode", "") not in ("", None):
        problems.append("margin_mode muss leer sein.")
    if "leverage" in config or "liquidation_buffer" in config:
        problems.append("Hebel-Parameter sind nicht erlaubt.")
    if config.get("initial_state", "stopped") != "stopped":
        problems.append("initial_state muss 'stopped' sein.")
    if config.get("force_entry_enable", False):
        problems.append("force_entry_enable muss false sein.")

    exchange = config.get("exchange", {})
    if exchange.get("name") not in set(allowed_exchanges):
        problems.append(f"Börse '{exchange.get('name')}' ist nicht freigegeben.")
    for key in SECRET_KEYS:
        if exchange.get(key):
            problems.append(f"exchange.{key} ist gesetzt – in Phase 1 keine Zugangsdaten.")
    pairs = set(exchange.get("pair_whitelist", []))
    not_allowed = pairs - set(allowed_pairs)
    if not_allowed:
        problems.append(f"Nicht freigegebene Märkte: {sorted(not_allowed)}.")

    if config.get("telegram", {}).get("enabled"):
        problems.append("Telegram muss deaktiviert sein.")
    api = config.get("api_server", {})
    if api.get("enabled"):
        problems.append("api_server muss in Phase 1 deaktiviert sein.")
    if api.get("listen_ip_address", "127.0.0.1") not in ("127.0.0.1", "localhost"):
        problems.append("api_server darf nur an 127.0.0.1 gebunden werden.")
    return problems


def assert_safe_freqtrade_config(
    config: dict[str, Any],
    allowed_exchanges: Iterable[str],
    allowed_pairs: Iterable[str],
) -> None:
    problems = check_freqtrade_config(config, allowed_exchanges, allowed_pairs)
    if problems:
        raise SafetyError("Unsichere Konfiguration:\n- " + "\n- ".join(problems))
