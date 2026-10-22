"""Versionierte Modelle mit kontrollierter Freigabe und Rollback.

Lebenszyklus: candidate -> (Prüfung) passed | failed -> approved (nur durch einen Menschen)
-> active (genau eine je Bot) -> retired. Rollback aktiviert die zuletzt aktive, freigegebene Version.

Kein Code-Pfad erlaubt es einem Modell oder einem automatischen Lauf, eine Version freizugeben
oder zu aktivieren: ``approve`` und ``activate`` verlangen einen menschlichen Akteur (``by``).
"""

import hashlib
import json
from typing import Any

from kitb import paths
from kitb.config import load_json
from kitb.registry import Registry

RELEASE_CRITERIA = paths.CONFIG_DIR / "release_criteria.json"
AUTOMATED_ACTORS = {"", "system", "auto", "model", "kitb", "bot", "agent"}


class GovernanceError(RuntimeError):
    pass


def _human(by: str | None) -> str:
    if not by or by.strip().lower() in AUTOMATED_ACTORS or by.strip().lower().startswith(("bot", "agent", "model")):
        raise GovernanceError("Freigabe/Aktivierung nur durch einen namentlich genannten Menschen (--by NAME).")
    return f"human:{by.strip()}"


def register_candidate(reg: Registry, *, bot: str, identifier: str, freqaimodel: str, overlay: dict,
                       data_fingerprints: dict, eval_timerange: str, metrics: dict, resources: dict) -> int:
    cfg = json.dumps(overlay, sort_keys=True)
    rows = reg.query("SELECT COALESCE(MAX(version), 0) AS v FROM model_versions WHERE bot = ?", (bot,))
    version = rows[0]["v"] + 1
    vid = reg.insert("model_versions", {
        "bot": bot, "version": version, "identifier": identifier, "freqaimodel": freqaimodel, "config": cfg,
        "config_sha256": hashlib.sha256(cfg.encode()).hexdigest(),
        "data_fingerprints": json.dumps(data_fingerprints, sort_keys=True), "eval_timerange": eval_timerange,
        "metrics": json.dumps(metrics, default=str, sort_keys=True), "resources": json.dumps(resources, default=str),
        "criteria": None, "status": "candidate", "approved_by": None})
    _event(reg, vid, "registered", "system", {"version": version})
    return vid


def get(reg: Registry, version_id: int) -> dict[str, Any]:
    rows = reg.query("SELECT * FROM model_versions WHERE id = ?", (version_id,))
    if not rows:
        raise GovernanceError(f"Modellversion {version_id} existiert nicht.")
    row = rows[0]
    for k in ("config", "data_fingerprints", "metrics", "resources", "criteria"):
        row[k] = json.loads(row[k]) if row[k] else None
    return row


def active(reg: Registry, bot: str) -> dict[str, Any] | None:
    rows = reg.query("SELECT id FROM model_versions WHERE bot = ? AND status = 'active'", (bot,))
    return get(reg, rows[0]["id"]) if rows else None


def evaluate_criteria(reg: Registry, version_id: int, *, reference_metrics: dict | None = None,
                      false_positive_rate: float | None = None, criteria: dict | None = None) -> dict[str, Any]:
    """Prüft die dokumentierten Kriterien. Ergebnis wird gespeichert; Status -> passed/failed."""
    crit = criteria or load_json(RELEASE_CRITERIA)
    mv = get(reg, version_id)
    m = mv["metrics"]
    checks: dict[str, bool] = {
        "min_trades": m.get("total_trades", 0) >= crit["min_trades"],
        "min_profit_factor": (m.get("profit_factor") or 0) >= crit["min_profit_factor"],
        "max_drawdown": (m.get("max_drawdown_account") or 1) <= crit["max_drawdown"],
        "min_profit_total": (m.get("profit_total") or -1) > crit["min_profit_total"],
        "max_wall_seconds": (mv["resources"] or {}).get("wall_seconds", 0) <= crit["max_wall_seconds"],
        # t-Test auf mittleren Gewinn je Trade (Freqtrade); laut Freqtrade-Doku eher optimistisch.
        "max_mean_profit_pvalue": m.get("p_value") is not None and m["p_value"] <= crit["max_mean_profit_pvalue"],
    }
    if crit.get("must_beat_buy_and_hold"):
        checks["beats_buy_and_hold"] = (m.get("excess_vs_buy_and_hold") or -1) > 0
    if crit.get("must_beat_reference"):
        checks["beats_reference"] = (reference_metrics is not None and
                                     (m.get("profit_total") or -1) > (reference_metrics.get("profit_total") or 0))
    if false_positive_rate is not None:
        checks["max_false_positive_rate"] = false_positive_rate <= crit["max_false_positive_rate"]
    if crit.get("must_beat_active_version"):
        act = active(reg, mv["bot"])
        checks["beats_active_version"] = act is None or (
            (m.get("profit_total") or -1) > (act["metrics"].get("profit_total") or 0))
    passed = all(checks.values())
    result = {"passed": passed, "checks": checks, "criteria": crit,
              "low_sample": m.get("total_trades", 0) < crit["min_trades"]}
    reg.execute("UPDATE model_versions SET criteria = ?, status = ? WHERE id = ? AND status IN ('candidate','passed','failed')",
                (json.dumps(result), "passed" if passed else "failed", version_id))
    _event(reg, version_id, "evaluated", "system", result)
    return result


def approve(reg: Registry, version_id: int, *, by: str, override_reason: str | None = None) -> None:
    actor = _human(by)
    mv = get(reg, version_id)
    if mv["status"] != "passed" and not override_reason:
        raise GovernanceError(f"Version {version_id} hat Status '{mv['status']}' – nur 'passed' ist freigebbar "
                              "(oder bewusst mit --override-reason).")
    reg.execute("UPDATE model_versions SET status = 'approved', approved_by = ? WHERE id = ?", (actor, version_id))
    _event(reg, version_id, "approved", actor, {"override_reason": override_reason})


def activate(reg: Registry, version_id: int, *, by: str) -> None:
    actor = _human(by)
    mv = get(reg, version_id)
    if mv["status"] not in ("approved", "retired") or not mv["approved_by"]:
        raise GovernanceError("Nur freigegebene Versionen können aktiviert werden.")
    current = active(reg, mv["bot"])
    if current:
        reg.execute("UPDATE model_versions SET status = 'retired' WHERE id = ?", (current["id"],))
        _event(reg, current["id"], "retired", actor, {"replaced_by": version_id})
    reg.execute("UPDATE model_versions SET status = 'active' WHERE id = ?", (version_id,))
    _event(reg, version_id, "activated", actor, {"previous": current["id"] if current else None})


def rollback(reg: Registry, bot: str, *, by: str) -> int:
    """Aktiviert die zuletzt aktive, freigegebene Vorgängerversion."""
    actor = _human(by)
    current = active(reg, bot)
    if not current:
        raise GovernanceError(f"Für {bot} ist keine Version aktiv.")
    prev = reg.query(
        "SELECT e.model_version_id AS id FROM model_events e JOIN model_versions v ON v.id = e.model_version_id "
        "WHERE v.bot = ? AND e.event = 'activated' AND e.model_version_id != ? AND v.approved_by IS NOT NULL "
        "ORDER BY e.id DESC LIMIT 1", (bot, current["id"]))
    if not prev:
        raise GovernanceError("Keine frühere freigegebene Version für einen Rollback vorhanden.")
    activate(reg, prev[0]["id"], by=by)
    _event(reg, current["id"], "rolled_back", actor, {"to": prev[0]["id"]})
    return prev[0]["id"]


def _event(reg: Registry, vid: int, event: str, actor: str, details: dict) -> None:
    reg.insert("model_events", {"model_version_id": vid, "event": event, "actor": actor,
                                "details": json.dumps(details, default=str)})
