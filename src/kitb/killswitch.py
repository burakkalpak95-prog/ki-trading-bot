"""Not-Stopp (Kill Switch) als Datei – funktioniert auch, wenn Dashboard oder Datenbank ausfallen.

Ist ``user_data/KILL_SWITCH`` vorhanden, verweigert das Risikomodul jeden neuen Einstieg und die
Paper-Trading-Steuerung stoppt laufende Bots. Zurücksetzen nur durch einen Menschen.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from kitb import paths


def path(user_data_dir: Path | None = None) -> Path:
    return (user_data_dir or paths.USER_DATA) / "KILL_SWITCH"


def activate(reason: str, by: str, user_data_dir: Path | None = None) -> Path:
    p = path(user_data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"activated_at": datetime.now(UTC).isoformat(), "reason": reason, "by": by}),
                 encoding="utf-8")
    return p


def is_active(user_data_dir: Path | None = None) -> bool:
    return path(user_data_dir).exists()


def info(user_data_dir: Path | None = None) -> dict | None:
    p = path(user_data_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"reason": "unlesbar – gilt trotzdem als aktiv"}


def reset(by: str, user_data_dir: Path | None = None) -> None:
    from kitb.models import _human

    _human(by)  # nur Menschen dürfen zurücksetzen
    path(user_data_dir).unlink(missing_ok=True)
