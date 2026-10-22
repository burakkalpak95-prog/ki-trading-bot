"""Phase 8 (vorbereitet, GESPERRT): austauschbare Schnittstelle für späteres Live-Trading.

Es gibt keinen Code-Pfad, der echtes Geld handelt. ``LockedLiveAdapter`` verweigert jede Aktion;
der Laufmodus ``live`` ist in ``kitb.safety`` nicht erlaubt; Umgebungsvariablen oder
Konfigurationswerte können das nicht ändern. Eine Aktivierung erfordert:

1. ausdrückliche, separate Freigabe durch den Nutzer,
2. unabhängige technische und risikobezogene Prüfung,
3. eine bewusste Codeänderung (eigener Adapter, Tests, Review) – nicht nur Konfiguration.

``readiness()`` listet, welche Voraussetzungen erfüllt sind – zur Vorbereitung, nicht zur Aktivierung.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from kitb import killswitch, paths
from kitb.risk import RISK_CONFIG, RiskLimits
from kitb.safety import ALLOWED_RUN_MODES, SafetyError

RISK_CONFIRMATION = "risk_confirmation.json"
SECRET_ENV_VARS = ("KITB_EXCHANGE_KEY", "KITB_EXCHANGE_SECRET")


class LiveTradingLocked(SafetyError):
    pass


@dataclass(frozen=True)
class OrderRequest:
    pair: str
    side: str        # nur "buy"/"sell" im Spotmarkt
    amount: float
    price: float | None
    client_order_id: str  # Idempotenz-Schlüssel gegen doppelte Orders


class LiveTradingAdapter(Protocol):
    """Schnittstelle, die ein künftiger Live-Adapter erfüllen muss."""

    def place_order(self, order: OrderRequest) -> dict[str, Any]: ...
    def cancel_order(self, order_id: str, pair: str) -> None: ...
    def open_orders(self) -> list[dict[str, Any]]: ...
    def balances(self) -> dict[str, float]: ...
    def reconcile(self) -> dict[str, Any]: ...  # Abgleich lokaler Zustand <-> Börse nach Verbindungsabbruch


class LockedLiveAdapter:
    """Einzige vorhandene Implementierung: verweigert alles."""

    def _deny(self, *_: Any, **__: Any) -> Any:
        raise LiveTradingLocked("Live-Trading ist gesperrt (Phase 8 nur vorbereitet).")

    place_order = cancel_order = open_orders = balances = reconcile = _deny


def get_adapter() -> LiveTradingAdapter:
    if "live" not in ALLOWED_RUN_MODES:
        return LockedLiveAdapter()
    raise LiveTradingLocked("Kein freigegebener Live-Adapter implementiert.")


def check_min_order(market: dict[str, Any], amount: float, price: float) -> list[str]:
    """Prüft Mindestmenge und Mindestwert (ccxt-Marktstruktur)."""
    problems = []
    limits = market.get("limits", {})
    min_amount = (limits.get("amount") or {}).get("min")
    min_cost = (limits.get("cost") or {}).get("min")
    if min_amount is not None and amount < min_amount:
        problems.append(f"Menge {amount} < Minimum {min_amount}")
    if min_cost is not None and amount * price < min_cost:
        problems.append(f"Orderwert {amount * price:.2f} < Minimum {min_cost}")
    if not market.get("spot", False):
        problems.append("Kein Spotmarkt")
    return problems


def _risk_hash() -> str:
    return hashlib.sha256(Path(RISK_CONFIG).read_bytes()).hexdigest()


def confirm_risk_limits(by: str, user_data_dir: Path | None = None) -> Path:
    """Ein Mensch bestätigt die aktuell konfigurierten Risikolimits (Hash der Datei)."""
    from kitb.models import _human

    actor = _human(by)
    RiskLimits.load()  # validiert
    p = (user_data_dir or paths.USER_DATA) / RISK_CONFIRMATION
    p.write_text(json.dumps({"risk_sha256": _risk_hash(), "by": actor,
                             "at": datetime.now(UTC).isoformat()}, indent=2), encoding="utf-8")
    return p


def risk_limits_confirmed(user_data_dir: Path | None = None) -> bool:
    p = (user_data_dir or paths.USER_DATA) / RISK_CONFIRMATION
    try:
        return json.loads(p.read_text(encoding="utf-8"))["risk_sha256"] == _risk_hash()
    except (OSError, ValueError, KeyError):
        return False


def readiness(user_data_dir: Path | None = None) -> dict[str, Any]:
    from kitb import models
    from kitb.registry import Registry

    reg = Registry(paths.REGISTRY_DB)
    paper_trades = 0
    for bot in ("bot_a", "bot_b"):
        db = (user_data_dir or paths.USER_DATA) / "paper" / bot / "tradesv3.dryrun.sqlite"
        if db.exists():
            from kitb.dashboard.data import paper_trades as pt

            paper_trades += len(pt(bot, user_data_dir))
    items = {
        "Live-Modus im Code freigeschaltet": "live" in ALLOWED_RUN_MODES,
        "Eigener, geprüfter Live-Adapter vorhanden": False,
        "Separate Nutzerfreigabe dokumentiert": False,
        "Unabhängige technische/risikobezogene Prüfung dokumentiert": False,
        "Risikolimits ausdrücklich bestätigt (aktueller Stand)": risk_limits_confirmed(user_data_dir),
        "Aktive, freigegebene Modellversion (mind. ein Bot)": any(models.active(reg, b) for b in ("bot_a", "bot_b")),
        "Langzeit-Paper-Trading: mind. 100 Trades insgesamt": paper_trades >= 100,
        "Not-Stopp funktionsfähig (Datei schreibbar)": os.access((user_data_dir or paths.USER_DATA), os.W_OK),
        "Not-Stopp derzeit nicht aktiv": not killswitch.is_active(user_data_dir),
        "Zugangsdaten nur über Umgebungsvariablen (nicht in Dateien)": True,
        "API-Key ohne Auszahlungsrecht (manuell bei der Börse prüfen)": False,
    }
    return {"live_trading_possible": False, "checks": items,
            "note": "Auch wenn alle Punkte erfüllt sind, bleibt Live-Trading gesperrt, bis Code und Freigabe vorliegen."}
