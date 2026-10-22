"""BaselineEmaCross mit vorgeschaltetem Agenten-Filter (nur zur Nutzenprüfung in Phase 5)."""

import importlib.util
from pathlib import Path

from kitb.agents.gate import AgentGateMixin

_spec = importlib.util.spec_from_file_location("_baseline", Path(__file__).with_name("BaselineEmaCross.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class AgentGatedBaseline(AgentGateMixin, _mod.BaselineEmaCross):
    startup_candle_count = 400
