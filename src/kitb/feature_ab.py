"""Vorher/Nachher-Test für KI 1 (Burak, 23.10.2026: „1 aber setze es jetzt sofort um“).

Frage: Trifft KI 1 mit den Zusatzdaten vom 23.10. (Regel-/Methoden-Signale, Orderfluss, Kontext, Formel-Faktoren)
bessere Entscheidungen als ohne? Zwei Backtests mit identischem Modell, Coins, Zeitraum und Kosten; einziger
Unterschied ist ``freqai.kitb.extra_features``. Der Test schreibt nichts in das Gedächtnis der laufenden KIs
(eigener user_data-Ordner).

Hinweise (Unsicher):
- Übungstrades ohne Signal laufen nur im Paper-Lernmodus, nicht im Backtest; verglichen wird die Signal-Qualität.
- Long/Short-Verhältnisse und Open Interest gibt es nur für die letzten ~180 Tage; davor sind sie im Backtest 0.
- Ein einzelner Zeitraum: ein Unterschied unter ~2 Prozentpunkten oder wenige Trades sind kein Beleg.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

from kitb import paths
from kitb.config import load_project_config

BOT = "ki_trader"
TIMERANGE = "20240101-20251104"  # Test-Abschnitt ab 05.11.2025 bleibt unberührt
KEYS = ["total_trades", "winrate", "profit_total", "profit_factor", "sharpe", "max_drawdown_account",
        "excess_vs_buy_and_hold"]
LABELS = {"total_trades": "Trades", "winrate": "Trefferquote", "profit_total": "Ergebnis nach Gebühren",
          "profit_factor": "Profitfaktor", "sharpe": "Sharpe", "max_drawdown_account": "Max. Drawdown",
          "excess_vs_buy_and_hold": "Vorsprung zu Kaufen+Halten"}


def _overlay(out: Path, identifier: str, extra: bool) -> Path:
    from kitb.ml import build_overlay

    p = build_overlay(BOT, identifier, out)
    cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg["freqai"].setdefault("kitb", {})["extra_features"] = extra
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return p


def run_variant(extra: bool, *, tmp: Path, cfg: Path, timerange: str, datadir: Path, data_source: str,
                offline: bool) -> dict[str, Any]:
    from kitb.ml import run_isolated

    spec = load_project_config()["bots"][BOT]
    identifier = f"ab-{BOT}-{'mit' if extra else 'ohne'}"
    res = run_isolated(
        strategies=[spec["strategy"]], timeframe=spec.get("timeframe", "15m"), timerange=timerange, datadir=datadir,
        data_source=data_source, offline=offline, registry_path=None, results_dir=tmp / "results", config_path=cfg,
        extra_config_files=[_overlay(tmp / "ud" / "freqai_configs", identifier, extra)],
        extra_argv=["--freqaimodel", spec["freqaimodel"], "--freqaimodel-path", str(paths.USER_DATA / "freqaimodels")],
        user_data_dir=tmp / "ud", extra_metrics={"bot": BOT, "variant": "mit" if extra else "ohne"})
    m = next(iter(res["strategies"].values()))["metrics"]
    return {"metrics": {k: m.get(k) for k in KEYS}, "resources": res["resources"]}


def verdict(mit: dict[str, Any], ohne: dict[str, Any]) -> str:
    a, b = mit["metrics"], ohne["metrics"]
    if min(a["total_trades"] or 0, b["total_trades"] or 0) < 30:
        return "Zu wenige Trades für ein Urteil."
    d = (a["profit_total"] or 0) - (b["profit_total"] or 0)
    if abs(d) < 0.02:
        return f"Kein klarer Unterschied ({d:+.1%} Ergebnis): die Zusatzdaten bringen messbar nichts."
    return (f"Mit Zusatzdaten {'besser' if d > 0 else 'schlechter'} ({d:+.1%} Ergebnis, Trefferquote "
            f"{(a['winrate'] or 0) - (b['winrate'] or 0):+.1%}).")


def render(mit: dict[str, Any], ohne: dict[str, Any], timerange: str, pairs: list[str]) -> str:
    rows = ["| Kennzahl | mit Zusatzdaten | ohne |", "|---|---|---|"]
    for k in KEYS:
        f = (lambda v: "—" if v is None else f"{v:.0f}") if k == "total_trades" else \
            (lambda v: "—" if v is None else f"{v:.2%}") if k in ("winrate", "profit_total", "max_drawdown_account",
                                                                   "excess_vs_buy_and_hold") else \
            (lambda v: "—" if v is None else f"{v:.2f}")
        rows.append(f"| {LABELS[k]} | {f(mit['metrics'][k])} | {f(ohne['metrics'][k])} |")
    return "\n".join([f"# Vorher/Nachher-Test KI 1 ({timerange}, {len(pairs)} Coins)", "", verdict(mit, ohne), "",
                      *rows, "",
                      f"Laufzeit: mit {mit['resources']['wall_seconds'] / 3600:.1f} h, "
                      f"ohne {ohne['resources']['wall_seconds'] / 3600:.1f} h.", "",
                      "Unsicher: ein Zeitraum, keine Übungstrades im Backtest, Long/Short-Daten erst ab ~180 Tagen."])


def run(*, datadir: Path, data_source: str, timerange: str = TIMERANGE, offline: bool = False,
        out_dir: Path | None = None) -> dict[str, Any]:
    from freqtrade.configuration.directory_operations import create_userdata_dir

    from kitb.methods import _config

    project = load_project_config()
    pairs = project.get("learning_pairs") or project["allowed_pairs"]
    out_dir = out_dir or (paths.USER_DATA / "reports" / "feature-ab")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        create_userdata_dir(str(tmp / "ud"), create_dir=True)
        cfg = _config(tmp, pairs)
        for extra in (False, True):  # „ohne“ zuerst: schneller, erstes Ergebnis früher da
            results["mit" if extra else "ohne"] = run_variant(extra, tmp=tmp, cfg=cfg, timerange=timerange,
                                                              datadir=datadir, data_source=data_source,
                                                              offline=offline)
            (out_dir / "feature-ab.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    report = render(results["mit"], results["ohne"], timerange, pairs)
    (out_dir / "feature-ab.md").write_text(report, encoding="utf-8")
    return {"report": str(out_dir / "feature-ab.md"), "verdict": verdict(results["mit"], results["ohne"])}
