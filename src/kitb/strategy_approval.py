"""Menschliche Freigabe von Regel-Strategien für Paper-Trading (Gegenstück zur Modellfreigabe in ``kitb.models``).

Regel-Strategien haben keine Modellversion. Damit trotzdem kein Bot ohne bewusste Entscheidung handelt, braucht jede
Paper-Strategie aus ``config/project.json → paper_strategies`` eine namentliche, protokollierte Freigabe.
"""

from typing import Any

from kitb.config import load_project_config
from kitb.models import GovernanceError, _human
from kitb.registry import Registry


def _spec(paper_id: str) -> dict[str, Any]:
    specs = load_project_config().get("paper_strategies", {})
    if paper_id not in specs:
        raise GovernanceError(f"Unbekannte Paper-Strategie: {paper_id}")
    return specs[paper_id]


def approve(reg: Registry, paper_id: str, *, by: str, reason: str) -> int:
    actor = _human(by)
    if not reason or not reason.strip():
        raise GovernanceError("Für die Freigabe ist eine Begründung nötig (--reason).")
    return reg.insert("strategy_approvals", {"paper_id": paper_id, "strategy": _spec(paper_id)["strategy"],
                                             "event": "approved", "actor": actor, "reason": reason.strip()})


def revoke(reg: Registry, paper_id: str, *, by: str, reason: str = "") -> int:
    actor = _human(by)
    return reg.insert("strategy_approvals", {"paper_id": paper_id, "strategy": _spec(paper_id)["strategy"],
                                             "event": "revoked", "actor": actor, "reason": reason.strip()})


def current(reg: Registry, paper_id: str) -> dict[str, Any] | None:
    """Gültige Freigabe: das letzte Ereignis ist 'approved' und gilt für die aktuell konfigurierte Strategie."""
    rows = reg.query("SELECT * FROM strategy_approvals WHERE paper_id = ? ORDER BY id DESC LIMIT 1", (paper_id,))
    if not rows or rows[0]["event"] != "approved" or rows[0]["strategy"] != _spec(paper_id)["strategy"]:
        return None
    return rows[0]
