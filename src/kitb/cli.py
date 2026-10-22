"""Kommandozeile ``kitb``. Bietet in Phase 1 nur Daten- und Backtest-Befehle.

Es gibt bewusst keinen Befehl zum Starten von Paper- oder Live-Trading.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from kitb import paths
from kitb.config import load_freqtrade_config, load_project_config
from kitb.registry import Registry
from kitb.safety import SafetyError, assert_run_mode_allowed, check_freqtrade_config


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def cmd_check_safety(args) -> int:
    project = load_project_config()
    problems = check_freqtrade_config(load_freqtrade_config(args.config), project["allowed_exchanges"],
                                      project["allowed_pairs"])
    if problems:
        print("UNSICHER:\n- " + "\n- ".join(problems))
        return 1
    print("OK: Konfiguration erfüllt die Phase-1-Sicherheitsregeln (nur Daten/Backtest, dry_run, Spot).")
    return 0


def cmd_probe(args) -> int:
    from kitb.data.source import probe_exchange

    assert_run_mode_allowed("data")
    project = load_project_config()
    reg = Registry(paths.REGISTRY_DB)
    net = project["network"]
    rc = 0
    for ex in args.exchanges:
        res = probe_exchange(ex, project["allowed_pairs"], retries=net["retries"],
                             backoff_base_seconds=net["backoff_base_seconds"], timeout_ms=net["timeout_ms"])
        reg.log_source_event(ex, "probe", "ok" if res["ok"] else "failed", res)
        _print(res)
        rc |= 0 if res["ok"] else 1
    return rc


def cmd_download(args) -> int:
    from kitb.data.source import download_with_freqtrade

    assert_run_mode_allowed("data")
    project = load_project_config()
    datadir = paths.DATA_DIR / args.exchange
    reg = Registry(paths.REGISTRY_DB)
    proc = download_with_freqtrade(paths.FREQTRADE_BASE_CONFIG, args.exchange, project["allowed_pairs"],
                                   args.timeframes or project["timeframes"], datadir, args.timerange, args.days)
    status = "ok" if proc.returncode == 0 else "failed"
    reg.log_source_event(args.exchange, "download", status, {
        "timeframes": args.timeframes or project["timeframes"], "timerange": args.timerange,
        "days": args.days, "returncode": proc.returncode, "stderr_tail": proc.stderr[-2000:]})
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr[-4000:])
    if proc.returncode != 0:
        return proc.returncode
    _trim_incomplete(datadir, project["allowed_pairs"], args.timeframes or project["timeframes"])
    return _validate_dir(datadir, args.exchange, args.timeframes or project["timeframes"], live_check=True)


def _trim_incomplete(datadir: Path, pairs: list[str], timeframes: list[str]) -> None:
    """Noch nicht geschlossene Kerzen nie als abgeschlossen speichern."""
    from kitb.data.storage import load_ohlcv_raw, store_ohlcv
    from kitb.data.validate import drop_incomplete_candles

    now = pd.Timestamp.now(tz="UTC")
    for pair in pairs:
        for tf in timeframes:
            df = load_ohlcv_raw(datadir, pair, tf)
            trimmed = drop_incomplete_candles(df, tf, now)
            if len(trimmed) < len(df):
                store_ohlcv(datadir, pair, tf, trimmed)
                print(f"{pair} {tf}: {len(df) - len(trimmed)} unvollständige Kerze(n) entfernt.")


def cmd_synthetic(args) -> int:
    from kitb.data.synthetic import write_synthetic_dataset

    project = load_project_config()
    written = write_synthetic_dataset(paths.SYNTHETIC_DATA_DIR, project["allowed_pairs"],
                                      project["timeframes"], args.start, args.end, args.seed)
    Registry(paths.REGISTRY_DB).log_source_event("synthetic", "synthetic", "ok", {
        "seed": args.seed, "start": args.start, "end": args.end,
        "rows": {f"{p} {tf}": n for (p, tf), n in written.items()}})
    print(f"SYNTHETISCHE Daten geschrieben nach {paths.SYNTHETIC_DATA_DIR}:")
    for (pair, tf), n in written.items():
        print(f"  {pair} {tf}: {n} Kerzen")
    return 0


def _validate_dir(datadir: Path, source: str, timeframes: list[str], live_check: bool) -> int:
    from kitb.data.storage import load_ohlcv_raw
    from kitb.data.validate import validate_ohlcv

    project = load_project_config()
    reg = Registry(paths.REGISTRY_DB)
    now = pd.Timestamp.now(tz="UTC") if live_check else None
    all_ok = True
    for pair in project["allowed_pairs"]:
        for tf in timeframes:
            df = load_ohlcv_raw(datadir, pair, tf)
            rep = validate_ohlcv(df, pair, tf, source, now=now,
                                 max_abs_log_return=project["validation"]["max_abs_log_return"].get(tf, 0.3),
                                 max_staleness_candles=project["validation"]["max_staleness_candles"])
            reg.add_validation(rep.to_dict())
            all_ok &= rep.ok
            state = "OK" if rep.ok else "FEHLER"
            print(f"[{state}] {pair} {tf}: {rep.rows} Kerzen {rep.first_candle} .. {rep.last_candle}")
            for i in rep.issues:
                print(f"    {i.severity} {i.check}: {i.message} {i.examples[:3]}")
    return 0 if all_ok else 1


def cmd_validate(args) -> int:
    project = load_project_config()
    datadir = paths.SYNTHETIC_DATA_DIR if args.source == "synthetic" else paths.DATA_DIR / args.source
    return _validate_dir(datadir, args.source, args.timeframes or project["timeframes"], args.check_fresh)


def cmd_backtest(args) -> int:
    from kitb.backtest import run_backtest

    datadir = paths.SYNTHETIC_DATA_DIR if args.source == "synthetic" else paths.DATA_DIR / args.source
    res = run_backtest(strategy=args.strategy, timeframe=args.timeframe, timerange=args.timerange,
                       datadir=datadir, data_source=args.source, offline=args.offline_markets,
                       registry=Registry(paths.REGISTRY_DB))
    _print(res)
    if res["metrics"]["synthetic_data"]:
        print("\nHINWEIS: Backtest auf SYNTHETISCHEN Daten – Ergebnisse sind kein Hinweis auf reale Profitabilität.")
    if res["metrics"]["low_sample_warning"]:
        print("HINWEIS: Weniger als 30 Trades – Kennzahlen statistisch kaum belastbar.")
    return 0


def cmd_runs(args) -> int:
    _print(Registry(paths.REGISTRY_DB).fetch(args.table, args.limit))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kitb", description="KI-Trading-Bot Phase 1 (Daten + Backtest)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("check-safety", help="Konfiguration gegen Phase-1-Regeln prüfen")
    s.add_argument("--config", type=Path, default=paths.FREQTRADE_BASE_CONFIG)
    s.set_defaults(func=cmd_check_safety)

    s = sub.add_parser("probe", help="Öffentliche Börsen-API erreichbar? (Ticker + Kerzen)")
    s.add_argument("exchanges", nargs="+")
    s.set_defaults(func=cmd_probe)

    s = sub.add_parser("download", help="Historische Daten via freqtrade download-data")
    s.add_argument("--exchange", default="binance")
    s.add_argument("--timeframes", nargs="+")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--timerange", help="z. B. 20220101-")
    g.add_argument("--days", type=int)
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("synthetic", help="Klar gekennzeichnete synthetische Offline-Daten erzeugen")
    s.add_argument("--start", default="2023-01-01")
    s.add_argument("--end", default="2025-01-01")
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_synthetic)

    s = sub.add_parser("validate", help="Gespeicherte Daten prüfen")
    s.add_argument("--source", default="binance", help="Börsenname oder 'synthetic'")
    s.add_argument("--timeframes", nargs="+")
    s.add_argument("--check-fresh", action="store_true", help="auch Aktualität/unvollständige Kerzen prüfen")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("backtest", help="Reproduzierbarer Backtest mit Gebühren und Slippage")
    s.add_argument("--source", default="binance", help="Börsenname oder 'synthetic'")
    s.add_argument("--strategy", default="BaselineEmaCross")
    s.add_argument("--timeframe", default="1h")
    s.add_argument("--timerange")
    s.add_argument("--offline-markets", action="store_true",
                   help="statische Marktdefinitionen (nur mit synthetischen Daten)")
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("runs", help="Protokoll anzeigen")
    s.add_argument("table", choices=["source_events", "validation_reports", "backtest_runs"])
    s.add_argument("--limit", type=int, default=10)
    s.set_defaults(func=cmd_runs)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SafetyError as e:
        print(f"SICHERHEITSSTOPP: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
