"""Zentrale Pfade. Alles relativ zum Projektverzeichnis (überschreibbar per KITB_HOME)."""

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("KITB_HOME", Path(__file__).resolve().parents[2]))
CONFIG_DIR = PROJECT_ROOT / "config"
USER_DATA = PROJECT_ROOT / "user_data"
DATA_DIR = USER_DATA / "data"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
RESULTS_DIR = USER_DATA / "backtest_results"
REGISTRY_DB = USER_DATA / "kitb.sqlite"

FREQTRADE_BASE_CONFIG = CONFIG_DIR / "freqtrade.base.json"
PROJECT_CONFIG = CONFIG_DIR / "project.json"
