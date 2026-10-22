"""Nutzennachweis für Agenten: gleiche Strategie mit und ohne Agenten-Filter unter identischen Bedingungen."""

import json
from pathlib import Path
from typing import Any

from kitb.compare import compare_strategies
from kitb.config import load_project_config
from kitb.registry import Registry


def evaluate_agents(*, datadir: Path, data_source: str, offline: bool = False, registry: Registry | None = None,
                    results_dir: Path | None = None, report_dir: Path | None = None) -> dict[str, Any]:
    cfg = load_project_config()["agents"]
    pairs = cfg["pairs_to_evaluate"]
    crit = cfg["benefit_criteria"]
    strategies = [s for pair in pairs.items() for s in pair]
    report = compare_strategies(datadir=datadir, data_source=data_source, strategies=strategies, offline=offline,
                                registry=registry, results_dir=results_dir, report_dir=report_dir)
    res = report["results"]
    verdicts = {}
    for gated, base in pairs.items():
        g, b = res[gated]["segments"]["validation"], res[base]["segments"]["validation"]
        gw, bw = res[gated]["windows"], res[base]["windows"]
        better = sum(x["profit_total"] > y["profit_total"] for x, y in zip(gw, bw))
        share = better / len(gw) if gw else 0.0
        checks = {
            "profit_improved": (g["profit_total"] - b["profit_total"]) > crit["min_profit_improvement"],
            "drawdown_not_worse": (not crit["max_drawdown_not_worse"])
            or (g["max_drawdown_account"] or 0) <= (b["max_drawdown_account"] or 0),
            "windows_better": share >= crit["min_share_windows_better"],
            "enough_trades_kept": b["total_trades"] == 0
            or g["total_trades"] / b["total_trades"] >= crit["min_trades_kept_ratio"],
            "sample_ok": not g["low_sample_warning"],
        }
        verdicts[gated] = {"base": base, "checks": checks, "benefit_proven": all(checks.values()),
                           "validation_profit": {"with_agents": g["profit_total"], "without": b["profit_total"]},
                           "validation_drawdown": {"with_agents": g["max_drawdown_account"],
                                                   "without": b["max_drawdown_account"]},
                           "share_windows_better": round(share, 3)}
    proven = any(v["benefit_proven"] for v in verdicts.values())
    summary = {"compare_id": report["compare_id"], "synthetic_data": report["synthetic_data"], "verdicts": verdicts,
               "recommendation": ("Nutzen in mindestens einem Fall belegt – Aktivierung (agents.enabled) "
                                  "nach Prüfung auf echten Daten möglich." if proven else
                                  "Kein belegter Nutzen – Agenten bleiben deaktiviert (einfachere Architektur)."),
               "report_dir": report["report_dir"]}
    Path(report["report_dir"], "agents_verdict.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if registry:
        registry.log_source_event("evaluation", "agents_evaluated", "ok", summary)
    return summary
