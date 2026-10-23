"""Phase 7: Paper-Trading über Freqtrade-Dry-Run – virtuelles Kapital, kein echtes Geld.

Zwei Arten: KI-Bots (``bots``, brauchen eine freigegebene, aktive Modellversion) und Regel-Strategien
(``paper_strategies``, brauchen eine namentliche Freigabe in ``kitb.strategy_approval``).

- Beide Bots: gleiche Basis-Konfiguration (Startkapital, Positionsgröße, max. offene Trades),
  gleiche Risikolimits (config/risk.json), gleiche Gebühr.
- Spread/Slippage: Freqtrade simuliert Markt-Orders im Dry-Run über das Orderbuch der Börse
  (``get_dry_market_fill_price``, max. 5 % Abweichung); deshalb hier nur die Börsengebühr.
- Persistenz und Wiederanlauf: Freqtrade-Dry-Run-Datenbank je Bot unter ``user_data/paper/<bot>/``.
- Nur die aktive, von einem Menschen freigegebene Modellversion wird verwendet.
- Not-Stopp: Datei ``user_data/KILL_SWITCH`` -> keine Einstiege, Bot beendet sich (RiskManagedMixin).
"""

import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from kitb import killswitch, models, paths, strategy_approval
from kitb.config import load_freqtrade_config, load_project_config
from kitb.registry import Registry
from kitb.safety import (SafetyError, assert_freqtrade_command_allowed, assert_run_mode_allowed,
                         check_paper_config)


def paper_ids() -> list[str]:
    project = load_project_config()
    return list(project["bots"]) + list(project.get("paper_strategies", {}))


def paper_dir(bot: str, user_data_dir: Path | None = None) -> Path:
    return (user_data_dir or paths.USER_DATA) / "paper" / bot


def build_paper_config(bot: str, reg: Registry, user_data_dir: Path | None = None) -> tuple[Path, dict[str, Any]]:
    project = load_project_config()
    if bot in project.get("paper_strategies", {}):
        return _build_rule_config(bot, reg, project, user_data_dir)
    if bot not in project["bots"]:
        raise ValueError(f"Unbekannter Bot: {bot}")
    mv = models.active(reg, bot)
    learning = None
    if mv and mv.get("approved_by"):
        freqai = json.loads(json.dumps(mv["config"]["freqai"]))
        freqai["identifier"] = f"paper-{bot}-v{mv['version']}"
        # Kontrollierter Trainingszyklus: Neutraining im freigegebenen Rhythmus, Vorhersagen verfallen danach.
        freqai["live_retrain_hours"] = freqai["backtest_period_days"] * 24
        freqai["expiration_hours"] = freqai["backtest_period_days"] * 24 * 2
    else:
        learning = strategy_approval.current(reg, bot)
        if not learning:
            raise SafetyError(f"Für {bot} ist weder eine freigegebene, aktive Modellversion vorhanden "
                              "(kitb models approve/activate --by NAME) noch der Lernmodus freigegeben "
                              "(kitb strategies approve --id BOT --by NAME --reason ...).")
        # Lernmodus: gleiche Bot-Einstellungen wie im Vergleich, aber Neutraining in kurzem Takt mit den neuesten Kursen.
        from kitb.ml import FREQAI_SHARED
        from kitb.config import load_json

        spec = project["bots"][bot]
        freqai = json.loads(json.dumps(load_json(FREQAI_SHARED)["freqai"]))
        freqai["feature_parameters"].update(spec["feature_parameters"])
        freqai["model_training_parameters"] = spec["model_training_parameters"]
        # Kennung je Merkmals-Satz und Zeitrahmen: nach einer Änderung (z. B. KI 1 1h -> 15m, 23.10.2026) werden alte
        # Modelle mit anderen Merkmalen nicht mehr geladen ("Pipeline expected ... but got ..."), sondern neu trainiert.
        sig = json.dumps([spec.get("timeframe", "1h"), freqai["feature_parameters"]], sort_keys=True)
        freqai["identifier"] = f"paper-learn-{bot}-{hashlib.sha1(sig.encode()).hexdigest()[:8]}"
        # Üben statt warten (Burak, 23.10.2026): niedrigere Einstiegsschwelle + Anteil Lern-Trades, nur Paper-Lernmodus.
        pl = {**project.get("paper_learning", {}), **spec.get("practice", {})}
        hours = pl.get("retrain_hours", 24)
        freqai["live_retrain_hours"] = hours
        freqai["expiration_hours"] = max(hours * 2, 2)
        kitb = freqai.setdefault("kitb", {})
        kitb["entry_proba"] = pl.get("learning_entry_proba", kitb.get("entry_proba", 0.6))
        kitb["explore_share"] = pl.get("explore_share", 0.0)
        kitb["explore_floor"] = pl.get("explore_floor", 0.45)
        if pl.get("max_hold_candles"):
            kitb["max_hold_candles"] = pl["max_hold_candles"]
        kitb["memory_gate_signals"] = bool(pl.get("memory_gate_signals"))
    d = paper_dir(bot, user_data_dir)
    d.mkdir(parents=True, exist_ok=True)
    cfg = load_freqtrade_config()
    cfg.update({
        "bot_name": bot,
        "dry_run": True,
        "initial_state": "running",
        "db_url": f"sqlite:///{(d / 'tradesv3.dryrun.sqlite').resolve()}",
        "fee": project["costs"]["fee_per_side"],
        "freqai": freqai,
        "kitb_model_version_id": mv["id"] if learning is None else None,
        "kitb_mode": "learning" if learning else "approved_model",
        "kitb_strategy_approval_id": learning["id"] if learning else None,
    })
    if learning and project["bots"][bot].get("timeframe"):
        cfg["timeframe"] = project["bots"][bot]["timeframe"]  # z. B. KI 1 im 15-Minuten-Takt (Burak 23.10.2026)
    pairs = project["allowed_pairs"]
    if learning:
        pairs = _use_learning_pairs(cfg, project)
        _apply_learning_slots(cfg, project)
    problems = check_paper_config(cfg, project["allowed_exchanges"], pairs,
                                  str((user_data_dir or paths.USER_DATA).resolve() / "paper"))
    if problems:
        raise SafetyError("Unsichere Paper-Konfiguration:\n- " + "\n- ".join(problems))
    path = d / "config.json"
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path, cfg


def _apply_learning_slots(cfg: dict[str, Any], project: dict[str, Any]) -> None:
    """KIs im Paper-Lernmodus: mehr gleichzeitige Positionen (Freigabe Burak 23.10.2026, Entscheidungskarte: ein Platz je Lern-Coin, 16). Einsatz je Trade so
    verkleinert, dass alle Positionen zusammen das Gesamteinsatz-Limit (max_total_exposure_fraction) einhalten."""
    from kitb.risk import RiskLimits

    n = int(project.get("paper_learning", {}).get("max_open_positions_learning", 0))
    n = min(n, len(cfg.get("exchange", {}).get("pair_whitelist") or [])) or n  # mehr Plätze als Coins bringen nichts
    if n <= 0 or not cfg.get("dry_run"):
        return
    exposure = RiskLimits.load().max_total_exposure_fraction
    cfg["max_open_trades"] = n
    cfg["stake_amount"] = int(cfg["dry_run_wallet"] * exposure / n)
    cfg["kitb_max_open_positions"] = n


def _use_learning_pairs(cfg: dict[str, Any], project: dict[str, Any]) -> list[str]:
    """Lern-/Sammler-Bots handeln (virtuell) zusätzlich die Lernmärkte; Kandidaten bleiben bei allowed_pairs."""
    pairs = project.get("learning_pairs") or project["allowed_pairs"]
    cfg["exchange"] = {**cfg["exchange"], "pair_whitelist": list(pairs)}
    return list(pairs)


def _build_rule_config(paper_id: str, reg: Registry, project: dict[str, Any],
                       user_data_dir: Path | None) -> tuple[Path, dict[str, Any]]:
    approval = strategy_approval.current(reg, paper_id)
    if not approval:
        raise SafetyError(f"Für {paper_id} liegt keine gültige Freigabe vor "
                          "(kitb strategies approve --id ... --by NAME --reason ...).")
    d = paper_dir(paper_id, user_data_dir)
    d.mkdir(parents=True, exist_ok=True)
    cfg = load_freqtrade_config()
    cfg.update({
        "bot_name": paper_id,
        "dry_run": True,
        "initial_state": "running",
        "db_url": f"sqlite:///{(d / 'tradesv3.dryrun.sqlite').resolve()}",
        "fee": project["costs"]["fee_per_side"],
        "kitb_strategy_approval_id": approval["id"],
        "kitb_mode": project["paper_strategies"][paper_id].get("mode", "candidate"),
        "kitb_ai_filter": bool(project["paper_strategies"][paper_id].get("ai_filter")
                               and project.get("paper_learning", {}).get("apply_ai_filter")),
    })
    cfg.pop("timeframe", None)  # Zeitintervall kommt aus der Strategie (z. B. 1d) ...
    if project["paper_strategies"][paper_id].get("timeframe"):
        # ... außer bei Sammlern mit eigenem Intervall (z. B. 15m: dieselbe Regel, häufiger geprüft)
        cfg["timeframe"] = project["paper_strategies"][paper_id]["timeframe"]
    cfg["kitb_practice"] = project["paper_strategies"][paper_id].get("practice", {})
    pairs = project["allowed_pairs"]
    if cfg["kitb_mode"] in ("explore", "learning"):
        pairs = _use_learning_pairs(cfg, project)
    if cfg["kitb_mode"] == "learning":
        _apply_learning_slots(cfg, project)
    problems = check_paper_config(cfg, project["allowed_exchanges"], pairs,
                                  str((user_data_dir or paths.USER_DATA).resolve() / "paper"))
    if problems:
        raise SafetyError("Unsichere Paper-Konfiguration:\n- " + "\n- ".join(problems))
    path = d / "config.json"
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path, cfg


def command(bot: str, config_path: Path, user_data_dir: Path | None = None) -> list[str]:
    project = load_project_config()
    ud = user_data_dir or paths.USER_DATA
    rule = project.get("paper_strategies", {}).get(bot)
    spec = rule or project["bots"][bot]
    cmd = [sys.executable, "-m", "freqtrade", "trade", "-c", str(config_path), "--userdir", str(ud),
           "--strategy", spec["strategy"], "--strategy-path", str(paths.USER_DATA / "strategies")]
    if not rule:
        cmd += ["--freqaimodel", spec["freqaimodel"], "--freqaimodel-path", str(paths.USER_DATA / "freqaimodels")]
    return cmd + ["--logfile", str(paper_dir(bot, user_data_dir) / "freqtrade.log")]


def round_marker(bot: str, user_data_dir: Path | None = None) -> Path:
    return paper_dir(bot, user_data_dir) / "NEW_ROUND.json"


def rounds_dir(bot: str, user_data_dir: Path | None = None) -> Path:
    return paper_dir(bot, user_data_dir) / "rounds"


def request_new_round(bot: str, reason: str, user_data_dir: Path | None = None) -> None:
    from datetime import UTC, datetime

    p = round_marker(bot, user_data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"reason": reason, "at": datetime.now(UTC).isoformat()}), encoding="utf-8")


def archive_round(bot: str, user_data_dir: Path | None = None) -> int:
    """Aktuelle Dry-Run-DB als abgeschlossene Runde ablegen; die nächste Runde startet wieder mit dem gleichen
    Startkapital (dry_run_wallet). Abgelegte Runden bleiben Teil des KI-Gedächtnisses."""
    marker = round_marker(bot, user_data_dir)
    info = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
    rd = rounds_dir(bot, user_data_dir)
    rd.mkdir(parents=True, exist_ok=True)
    n = len(list(rd.glob("round-*.sqlite"))) + 1
    db = paper_dir(bot, user_data_dir) / "tradesv3.dryrun.sqlite"
    if db.exists():
        db.rename(rd / f"round-{n:03d}.sqlite")
    (rd / f"round-{n:03d}.json").write_text(json.dumps({"round": n, **info}), encoding="utf-8")
    marker.unlink(missing_ok=True)
    Registry(paths.REGISTRY_DB).log_source_event("paper", "new_round", "ok", {"bot": bot, "finished_round": n, **info})
    return n


def _pidfile(bot: str,user_data_dir: Path | None = None) -> Path:
    return paper_dir(bot, user_data_dir) / "bot.pid"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


HEARTBEAT_MAX_AGE_S = 180


def _log_age_seconds(bot: str, user_data_dir: Path | None = None) -> float | None:
    import time

    log_path = paper_dir(bot, user_data_dir) / "freqtrade.log"
    return time.time() - log_path.stat().st_mtime if log_path.exists() else None


def start(bot: str, *, foreground: bool, user_data_dir: Path | None = None) -> int:
    assert_run_mode_allowed("paper")
    assert_freqtrade_command_allowed("trade", mode="paper")
    if killswitch.is_active(user_data_dir):
        raise SafetyError("Not-Stopp ist aktiv – erst prüfen und durch einen Menschen zurücksetzen.")
    reg = Registry(paths.REGISTRY_DB)
    pid_path = _pidfile(bot, user_data_dir)
    if pid_path.exists() and _alive(int(pid_path.read_text())):
        raise SafetyError(f"{bot} läuft bereits (PID {pid_path.read_text()}).")
    cfg_path, cfg = build_paper_config(bot, reg, user_data_dir)
    cmd = command(bot, cfg_path, user_data_dir)
    reg.log_source_event("paper", "start", "ok", {"bot": bot, "model_version_id": cfg.get("kitb_model_version_id"),
                                                  "strategy_approval_id": cfg.get("kitb_strategy_approval_id"),
                                                  "foreground": foreground})
    if foreground:
        while True:
            rc = subprocess.call(cmd)
            # Runde zu Ende (Lern-/Sammler-Bot hat sein Verlustlimit erreicht): archivieren, frisch weiter.
            if not round_marker(bot, user_data_dir).exists() or killswitch.is_active(user_data_dir):
                return rc
            archive_round(bot, user_data_dir)
    proc =subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    pid_path.write_text(str(proc.pid))
    return 0


def stop(bot: str, user_data_dir: Path | None = None) -> bool:
    pid_path = _pidfile(bot, user_data_dir)
    stopped = False
    if pid_path.exists():
        pid = int(pid_path.read_text())
        if _alive(pid):
            os.kill(pid, signal.SIGTERM)  # Freqtrade beendet sich geordnet, Zustand bleibt in der DB
            stopped = True
        pid_path.unlink(missing_ok=True)
    Registry(paths.REGISTRY_DB).log_source_event("paper", "stop", "ok", {"bot": bot, "was_running": stopped})
    return stopped


def status(bot: str, user_data_dir: Path | None = None) -> dict[str, Any]:
    from kitb.dashboard.data import paper_summary, risk_state

    pid_path = _pidfile(bot, user_data_dir)
    heartbeat_age = _log_age_seconds(bot, user_data_dir)
    # Im Docker-Betrieb läuft der Bot in einem eigenen Container (keine PID-Datei, anderer Prozessraum);
    # dann gilt er als laufend, solange sein Log frisch ist (Freqtrade-Heartbeat standardmäßig alle 60 s).
    running = (pid_path.exists() and _alive(int(pid_path.read_text()))) or (
        heartbeat_age is not None and heartbeat_age <= HEARTBEAT_MAX_AGE_S)
    return {"bot": bot, "running": running,
            "log_age_seconds": None if heartbeat_age is None else round(heartbeat_age),
            "kill_switch": killswitch.info(user_data_dir),
            "summary": paper_summary(bot, load_freqtrade_config()["dry_run_wallet"], user_data_dir),
            "risk": risk_state(user_data_dir).get(bot)}


def report(user_data_dir: Path | None = None) -> str:
    """Vergleich aller Paper-Bots aus den Dry-Run-Datenbanken (Markdown, wird unter reports/ gespeichert)."""
    from datetime import UTC, datetime

    import pandas as pd

    from kitb.dashboard.data import paper_trades

    wallet = load_freqtrade_config()["dry_run_wallet"]
    lines = ["# Paper-Trading-Vergleich", "", f"Stand: {datetime.now(UTC):%Y-%m-%d %H:%M} UTC · Startkapital je Bot: {wallet} USDT", "",
             "| Bot | Trades (geschl.) | offen | Netto | Max DD (realisiert) | Trefferquote | Gebühren | erster Trade | Stichprobe |",
             "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for bot in paper_ids():
        t = paper_trades(bot, user_data_dir)
        closed = t[t["is_open"] == 0] if not t.empty else t
        if closed.empty:
            lines.append(f"| {bot} | 0 | {0 if t.empty else int((t['is_open'] == 1).sum())} | – | – | – | – | – | zu klein |")
            continue
        eq = wallet + closed["close_profit_abs"].cumsum()
        dd = float((1 - eq / pd.concat([pd.Series([wallet]), eq]).cummax().iloc[1:]).max())
        fees = float((closed["amount"] * closed["open_rate"] * closed["fee_open"]
                      + closed["amount"] * closed["close_rate"] * closed["fee_close"]).sum())
        lines.append(f"| {bot} | {len(closed)} | {int((t['is_open'] == 1).sum())} | "
                     f"{closed['close_profit_abs'].sum() / wallet:.2%} | {dd:.2%} | "
                     f"{(closed['close_profit_abs'] > 0).mean():.1%} | {fees:.2f} | {t['open_date'].min()} | "
                     f"{'zu klein' if len(closed) < 30 else 'ok'} |")
    lines += ["", "Hinweis: Paper-Trading simuliert Ausführung über das Orderbuch; echte Ausführung kann abweichen. "
              "Eine belastbare Aussage braucht viele Wochen und deutlich mehr als 30 Trades je Bot."]
    text = "\n".join(lines) + "\n"
    out = (user_data_dir or paths.USER_DATA) / "reports" / f"paper-{datetime.now(UTC):%Y%m%d-%H%M%S}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text
