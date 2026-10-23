"""Phase 3: Bot A (effizient) gegen Bot B (erweitert) unter identischen Bedingungen.

Identisch für beide Bots und die regelbasierte Referenz: Daten, Zeitraum, Kosten, Startkapital,
Positionsgrößen, max. offene Trades, Stoploss, Label, Ein-/Ausstiegsschwellen sowie das
FreqAI-Walk-Forward-Schema (Training auf ``train_period_days`` vor jedem Testfenster von
``backtest_period_days``, danach Neutraining).

Jeder Bot läuft in einem eigenen Prozess, damit Laufzeit und Spitzen-Arbeitsspeicher
sauber messbar sind.
"""

import json
import resource
import shutil
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from kitb import paths
from kitb.config import load_json, load_project_config
from kitb.registry import Registry

FREQAI_SHARED = paths.CONFIG_DIR / "freqai.shared.json"


def build_overlay(bot: str, identifier: str, out_dir: Path) -> Path:
    """Gemeinsame FreqAI-Konfiguration + bot-spezifische Features/Modellparameter."""
    project = load_project_config()
    spec = project["bots"][bot]
    shared = load_json(FREQAI_SHARED)
    freqai = json.loads(json.dumps(shared["freqai"]))
    freqai["identifier"] = identifier
    freqai["feature_parameters"].update(spec["feature_parameters"])
    freqai["model_training_parameters"] = spec["model_training_parameters"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"freqai-{identifier}.json"
    path.write_text(json.dumps({"freqai": freqai}, indent=2), encoding="utf-8")
    return path


def _regimes(datadir: Path) -> dict:
    from kitb.compare import _daily
    from kitb.config import load_freqtrade_config
    from kitb.evaluation import classify_regimes

    out = {}
    for pair in load_freqtrade_config()["exchange"]["pair_whitelist"]:
        try:
            out[pair] = classify_regimes(_daily(datadir, pair))
        except Exception:  # Marktphase ist Zusatzinfo, kein Abbruchgrund
            out[pair] = None
    return {k: v for k, v in out.items() if v is not None}


def _model_dir_size(user_data_dir: Path, identifier: str) -> int:
    d = user_data_dir / "models" / identifier
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.exists() else 0


def _count_trainings(user_data_dir: Path, identifier: str) -> int:
    d = user_data_dir / "models" / identifier
    return len(list(d.glob("sub-train-*"))) if d.exists() else 0


def _worker(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Läuft im Kindprozess (spawn): ein Freqtrade-Backtest + Ressourcenmessung."""
    from kitb.backtest import run_freqtrade_backtest

    registry_path = kwargs.pop("registry_path")
    decisions_spec = kwargs.pop("decisions_spec", None)
    t0 = time.perf_counter()
    cpu0 = resource.getrusage(resource.RUSAGE_SELF)
    res = run_freqtrade_backtest(registry=Registry(registry_path) if registry_path else None, **kwargs)
    cpu1 = resource.getrusage(resource.RUSAGE_SELF)
    res["resources"] = {
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "cpu_seconds": round((cpu1.ru_utime + cpu1.ru_stime) - (cpu0.ru_utime + cpu0.ru_stime), 2),
        "peak_rss_mb": round(cpu1.ru_maxrss / 1024, 1),  # Linux: KiB
    }
    for name, s in res["strategies"].items():
        trades = s.pop("trades", [])
        if decisions_spec:
            from kitb.decisions import build_decisions, load_signals

            s["decisions"] = build_decisions(load_signals(Path(res["result_file"]), name), trades,
                                             strategy=name, timeframe=kwargs["timeframe"],
                                             entry_proba=decisions_spec["entry_proba"],
                                             regimes_by_pair=decisions_spec.get("regimes"))
    return res


def run_isolated(**kwargs: Any) -> dict[str, Any]:
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
        return pool.submit(_worker, kwargs).result()


def run_bot(bot: str, *, timerange: str, datadir: Path, data_source: str, offline: bool,
            registry_path: Path | None, results_dir: Path | None = None,
            user_data_dir: Path | None = None, keep_models: bool = True,
            compare_id: str = "") -> dict[str, Any]:
    project = load_project_config()
    spec = project["bots"][bot]
    user_data_dir = user_data_dir or paths.USER_DATA
    identifier = f"{bot}-{compare_id or uuid.uuid4().hex[:8]}"
    overlay = build_overlay(bot, identifier, user_data_dir / "freqai_configs")
    regimes = _regimes(datadir)
    res = run_isolated(
        export="signals",
        strategies=[spec["strategy"]], timeframe=spec.get("timeframe", "1h"), timerange=timerange, datadir=datadir,
        data_source=data_source, offline=offline, registry_path=registry_path, results_dir=results_dir,
        config_path=None, extra_config_files=[overlay],
        extra_argv=["--freqaimodel", spec["freqaimodel"],
                    "--freqaimodel-path", str(paths.USER_DATA / "freqaimodels")],
        user_data_dir=user_data_dir,
        extra_metrics={"bot": bot, "model": spec["freqaimodel"], "model_identifier": identifier,
                       "compare_id": compare_id},
        decisions_spec={"entry_proba": load_json(FREQAI_SHARED)["freqai"]["kitb"]["entry_proba"],
                        "regimes": regimes})
    res["resources"]["model_disk_kb"] = round(_model_dir_size(user_data_dir, identifier) / 1024, 1)
    res["resources"]["trainings"] = _count_trainings(user_data_dir, identifier)
    res["identifier"] = identifier
    res["overlay"] = json.loads(overlay.read_text(encoding="utf-8"))
    if not keep_models:
        shutil.rmtree(user_data_dir / "models" / identifier, ignore_errors=True)
    return res


ML_TABLE_KEYS = ["total_trades", "p_value", "profit_total", "max_drawdown_account", "profit_factor", "sharpe", "winrate",
                 "exposure", "total_cost_abs_incl_slippage", "excess_vs_buy_and_hold", "low_sample_warning"]


def ml_compare(*, datadir: Path, data_source: str, include_test: bool = False, offline: bool = False,
               registry_path: Path | None = None, results_dir: Path | None = None,
               user_data_dir: Path | None = None, report_dir: Path | None = None,
               keep_models: bool = True) -> dict[str, Any]:
    """Bot A, Bot B und die regelbasierten Strategien auf denselben Abschnitten (Standard: Validation)."""
    from datetime import UTC, datetime

    from kitb.compare import data_span
    from kitb.config import load_freqtrade_config
    from kitb.evaluation import chronological_split

    project = load_project_config()
    pairs = load_freqtrade_config()["exchange"]["pair_whitelist"]
    compare_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    start, end = data_span(datadir, pairs, "1h")
    segments = chronological_split(start, end, project["evaluation"]["split_fractions"])
    evaluated = [s for s in segments if s.name == "validation" or (include_test and s.name == "test")]
    if include_test and registry_path:
        Registry(registry_path).log_source_event("evaluation", "test_set_used", "ok",
                                                 {"compare_id": compare_id, "kind": "ml"})
    shared = load_json(FREQAI_SHARED)["freqai"]
    # Bot B nutzt u. a. EMA(200) auf Tageskerzen: Vorlauf + Trainingsfenster müssen vor dem Backtest liegen.
    if (evaluated[0].start - start).days < shared["kitb"]["min_history_days_before_backtest"]:
        raise ValueError("Zu wenig Historie vor dem Auswertungsabschnitt "
                         f"(< {shared['kitb']['min_history_days_before_backtest']} Tage) – mehr Daten laden.")

    references = project["evaluation"]["strategies"]
    report: dict[str, Any] = {"compare_id": compare_id, "data_source": data_source,
                              "segments": {s.name: s.timerange for s in segments},
                              "test_set_used": include_test, "freqai_shared": shared,
                              "bots": {b: {k: v for k, v in spec.items()} for b, spec in project["bots"].items()},
                              "results": {}}
    reg = Registry(registry_path) if registry_path else None
    for seg in evaluated:
        seg_res: dict[str, Any] = {}
        ref = run_isolated(strategies=references, timeframe="1h", timerange=seg.timerange, datadir=datadir,
                           data_source=data_source, offline=offline, registry_path=registry_path,
                           results_dir=results_dir, config_path=None, user_data_dir=user_data_dir,
                           extra_metrics={"compare_id": compare_id, "segment": seg.name, "kind": "ml_reference"})
        for name, s in ref["strategies"].items():
            seg_res[name] = {"metrics": {k: s["metrics"].get(k) for k in ML_TABLE_KEYS},
                             "resources": {"wall_seconds_all_references": ref["resources"]["wall_seconds"]}}
        reference_name = load_json(paths.CONFIG_DIR / "release_criteria.json").get("must_beat_reference")
        for bot in project["bots"]:
            r = run_bot(bot, timerange=seg.timerange, datadir=datadir, data_source=data_source, offline=offline,
                        registry_path=registry_path, results_dir=results_dir, user_data_dir=user_data_dir,
                        keep_models=keep_models, compare_id=f"{compare_id}-{seg.name}")
            strat = next(iter(r["strategies"].values()))
            m = strat["metrics"]
            seg_res[bot] = {"metrics": {k: m.get(k) for k in ML_TABLE_KEYS}, "resources": r["resources"],
                            "identifier": r["identifier"]}
            report["synthetic_data"] = m["synthetic_data"]
            report.setdefault("buy_and_hold", {})[seg.name] = m["buy_and_hold"]["equal_weight_net_return"]
            decisions = strat.get("decisions", [])
            fp_rate = (sum(d["model_error"] == "false_positive" for d in decisions) / len(decisions)
                       if decisions else None)
            seg_res[bot]["false_positive_rate"] = fp_rate
            if reg and seg.name == "validation":
                from kitb import models
                from kitb.decisions import store_decisions

                vid = models.register_candidate(
                    reg, bot=bot, identifier=r["identifier"], freqaimodel=project["bots"][bot]["freqaimodel"],
                    overlay=r["overlay"], data_fingerprints=r["data_fingerprints"], eval_timerange=seg.timerange,
                    metrics=m, resources=r["resources"])
                store_decisions(reg, decisions, source="backtest", run_ref=r["result_file"], bot=bot,
                                model_identifier=r["identifier"], model_version_id=vid)
                crit = models.evaluate_criteria(
                    reg, vid, false_positive_rate=fp_rate,
                    reference_metrics=ref["strategies"].get(reference_name, {}).get("metrics"))
                seg_res[bot]["model_version_id"] = vid
                seg_res[bot]["release_check"] = crit
        seg_res["verdict"] = verdict(seg_res, references)
        report["results"][seg.name] = seg_res

    report_dir = (report_dir or (user_data_dir or paths.USER_DATA) / "reports") / f"ml-compare-{compare_id}"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (report_dir / "report.md").write_text(render_ml_markdown(report), encoding="utf-8")
    report["report_dir"] = str(report_dir)
    return report


def verdict(seg_res: dict[str, Any], references: list[str]) -> dict[str, Any]:
    """Einfache, vorsichtige Einordnung – kein Signifikanztest."""
    a, b = seg_res["bot_a"]["metrics"], seg_res["bot_b"]["metrics"]
    best_ref = max(references, key=lambda r: seg_res[r]["metrics"]["profit_total"] or -1e9)
    low_sample = a["low_sample_warning"] or b["low_sample_warning"]
    b_better = b["profit_total"] > a["profit_total"]
    return {
        "bot_b_minus_bot_a_profit": b["profit_total"] - a["profit_total"],
        "bot_b_cost_factor_wall_time": round(seg_res["bot_b"]["resources"]["wall_seconds"]
                                             / max(seg_res["bot_a"]["resources"]["wall_seconds"], 1e-6), 2),
        "best_reference": best_ref,
        "bot_a_beats_best_reference": a["profit_total"] > seg_res[best_ref]["metrics"]["profit_total"],
        "bot_b_beats_best_reference": b["profit_total"] > seg_res[best_ref]["metrics"]["profit_total"],
        "statement": (
            "Stichprobe zu klein für eine Aussage." if low_sample else
            ("Bot B liegt vorn – Mehrwert erst nach Test-Abschnitt und Paper-Trading belastbar."
             if b_better else "Bot B bringt in diesem Abschnitt keinen Mehrwert gegenüber Bot A.")),
    }


def _pct(v: Any) -> str:
    return "–" if v is None else f"{v * 100:.2f} %"


def _num(v: Any) -> str:
    return "–" if v is None else f"{v:.2f}"


def render_ml_markdown(report: dict[str, Any]) -> str:
    lines = [f"# KI-Bot-Vergleich {report['compare_id']}", ""]
    if report.get("synthetic_data"):
        lines += ["> **SYNTHETISCHE DATEN** (Zufallsprozess ohne echtes Muster). Ein ML-Vorteil wäre hier ein "
                  "Warnsignal für Datenlecks; erwartet ist kein Vorteil.", ""]
    fq = report["freqai_shared"]
    lines += [f"Walk-Forward: Training {fq['train_period_days']} Tage, Neutraining alle {fq['backtest_period_days']} Tage, "
              f"Label-Horizont {fq['feature_parameters']['label_period_candles']} Kerzen, "
              f"Schwelle {fq['kitb']['label_threshold'] * 100:.2f} %.",
              f"Test-Abschnitt verwendet: **{'ja' if report['test_set_used'] else 'nein (unberührt)'}**", ""]
    for bot, spec in report["bots"].items():
        lines.append(f"- **{bot}** ({spec['strategy']}, {spec['freqaimodel']}): {spec['description']}")
    lines.append("")
    for seg, res in report["results"].items():
        lines += [f"## Abschnitt: {seg} `{report['segments'][seg]}`", "",
                  "| Bot/Strategie | Trades | Netto | Max DD | PF | Sharpe | Trefferquote | Exposure | Kosten | vs. B&H | Stichprobe |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for name, r in res.items():
            if name == "verdict":
                continue
            m = r["metrics"]
            lines.append(f"| {name} | {m['total_trades']} | {_pct(m['profit_total'])} | {_pct(m['max_drawdown_account'])} | "
                         f"{_num(m['profit_factor'])} | {_num(m['sharpe'])} | {_pct(m['winrate'])} | {_pct(m['exposure'])} | "
                         f"{_num(m['total_cost_abs_incl_slippage'])} | {_pct(m['excess_vs_buy_and_hold'])} | "
                         f"{'zu klein' if m['low_sample_warning'] else 'ok'} |")
        lines += ["", f"Buy&Hold (gleichgewichtet, nach Kosten): {_pct(report['buy_and_hold'][seg])}", "",
                  "| Bot | Laufzeit (s) | CPU (s) | Spitzen-RAM (MB) | Modelle auf Disk (KB) | Trainings |",
                  "|---|---:|---:|---:|---:|---:|"]
        for bot in report["bots"]:
            rs = res[bot]["resources"]
            lines.append(f"| {bot} | {rs['wall_seconds']} | {rs['cpu_seconds']} | {rs['peak_rss_mb']} | "
                         f"{rs['model_disk_kb']} | {rs['trainings']} |")
        for bot in report["bots"]:
            rc = res[bot].get("release_check")
            if rc:
                failed = [k for k, ok in rc["checks"].items() if not ok]
                lines.append(f"- {bot}: Modellversion #{res[bot]['model_version_id']} – Freigabekriterien "
                             f"{'erfüllt' if rc['passed'] else 'nicht erfüllt: ' + ', '.join(failed)}")
        lines.append("")
        v = res["verdict"]
        lines += ["", f"**Einordnung:** {v['statement']} Bot B − Bot A: {_pct(v['bot_b_minus_bot_a_profit'])}; "
                      f"Laufzeitfaktor Bot B/Bot A: {v['bot_b_cost_factor_wall_time']}×; "
                      f"beste Regel-Strategie: {v['best_reference']}.", ""]
    lines.append("Hinweis: Der p-Wert (Kriterium max_mean_profit_pvalue) prüft nur, ob der mittlere Trade-Gewinn von 0 verschieden ist – kein Beleg für künftige Gewinne.")
    return "\n".join(lines) + "\n"
