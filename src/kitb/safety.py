"""Sicherheitsprüfungen.

Erlaubte Modi: ``data``, ``backtest`` und (ab Phase 7) ``paper`` – Paper-Trading ausschließlich als
Freqtrade-Dry-Run ohne Zugangsdaten, gestartet über ``kitb paper``. **Live-Trading ist gesperrt** –
unabhängig von Umgebungsvariablen oder Konfigurationswerten. Eine Freigabe erfordert eine
Codeänderung nach separater Nutzerfreigabe und unabhängiger Prüfung (Phase 8).
"""

from collections.abc import Iterable
from typing import Any

ALLOWED_RUN_MODES = frozenset({"data", "backtest", "paper"})
BLOCKED_FREQTRADE_COMMANDS = frozenset({"trade", "webserver"})
PAPER_ONLY_COMMANDS = frozenset({"trade"})
SECRET_KEYS = ("key", "secret", "password", "uid", "walletAddress", "privateKey")


class SafetyError(RuntimeError):
    """Konfiguration oder Aufruf verletzt eine Phase-1-Sicherheitsregel."""


def assert_run_mode_allowed(mode: str) -> None:
    if mode not in ALLOWED_RUN_MODES:
        raise SafetyError(
            f"Modus '{mode}' ist gesperrt. Erlaubt: {sorted(ALLOWED_RUN_MODES)}. "
            "Live-Trading erst nach separater Freigabe und unabhängiger Prüfung."
        )


def assert_freqtrade_command_allowed(command: str, mode: str | None = None) -> None:
    if command in PAPER_ONLY_COMMANDS and mode == "paper":
        return
    if command in BLOCKED_FREQTRADE_COMMANDS:
        raise SafetyError(f"Freqtrade-Befehl '{command}' ist gesperrt (nur 'trade' im Paper-Modus über kitb).")


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


def check_paper_config(config: dict[str, Any], allowed_exchanges: Iterable[str],
                       allowed_pairs: Iterable[str], paper_root: str) -> list[str]:
    """Wie ``check_freqtrade_config``, aber für den laufenden Dry-Run-Bot (initial_state darf 'running' sein)."""
    cfg = {**config, "initial_state": "stopped"}
    problems = check_freqtrade_config(cfg, allowed_exchanges, allowed_pairs)
    db_url = str(config.get("db_url", ""))
    if "dryrun" not in db_url or paper_root not in db_url:
        problems.append("db_url muss eine Dry-Run-Datenbank unter user_data/paper sein.")
    if config.get("initial_state") not in ("running", "stopped"):
        problems.append("initial_state ungültig.")
    return problems
