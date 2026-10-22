"""Laden der Projekt- und Freqtrade-Konfiguration (JSON, ohne Kommentare)."""

import json
from pathlib import Path
from typing import Any

from kitb import paths


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data.pop("$comment", None)
    return data


def load_project_config(path: Path | None = None) -> dict[str, Any]:
    return load_json(path or paths.PROJECT_CONFIG)


def load_freqtrade_config(path: Path | None = None) -> dict[str, Any]:
    return load_json(path or paths.FREQTRADE_BASE_CONFIG)
