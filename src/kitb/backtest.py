"""Reproduzierbarer Backtest über die Freqtrade-Backtesting-Engine.

Ablauf: Sicherheitsprüfung -> Datenvalidierung -> Freqtrade-Backtest mit
effektivem Gebührensatz (Gebühr + Slippage + halber Spread) -> Kennzahlen inkl.
Buy-and-Hold-Benchmark -> Eintrag in SQLite.
"""

import hashlib
import json
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd

from kitb import paths
from kitb.config import load_freqtrade_config, load_project_config
from kitb.costs import CostModel
from kitb.data.storage import load_ohlcv_raw
from kitb.data.synthetic import is_synthetic_dir
from kitb.data.validate import validate_ohlcv
from kitb.offline import offline_markets
from kitb.registry import Registry
from kitb.safety import SafetyError, assert_run_mode_allowed, assert_safe_freqtrade_config

log = logging.getLogger(__name__)
MIN_TRADES_FOR_STATS = 30


class DataQualityError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slice(df: pd.DataFrame, timerange: str | None) -> pd.DataFrame:
    if not timerange:
        return df
    start, _, end = timerange.partition("-")
    if start:
        df = df[df["date"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["date"] < pd.Timestamp(end, tz="UTC")]
    return df


def buy_and_hold(df: pd.DataFrame, costs: CostModel) -> dict[str, float]:
    first, last = df["open"].iloc[0], df["close"].iloc[-1]
    gross = last / first - 1
    net = (1 - costs.effective_fee_per_side) ** 2 * (last / first) - 1
    running_max = df["close"].cummax()
    max_dd = float((df["close"] / running_max - 1).min())
    return {"gross_return": float(gross), "net_return": float(net), "max_drawdown": -max_dd}


def exposure(trades: list[dict], start: str, end: str, n_slots: int) -> float | None:
    """Anteil der verfügbaren Positions-Slots (max_open_trades), der im Markt verbracht wurde."""
    total = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() * n_slots
    if total <= 0:
        return None
    in_market = sum((t["close_timestamp"] - t["open_timestamp"]) / 1000 for t in trades)
    return round(in_market / total, 4)


def summarize(strategy_stats: dict[str, Any], cost_model: CostModel) -> dict[str, Any]:
    """Relevante Kennzahlen aus den Freqtrade-Ergebnissen ziehen."""
    trades = strategy_stats.get("trades", [])
    n = int(strategy_stats.get("total_trades", len(trades)))
    wins = [t["profit_abs"] for t in trades if t["profit_abs"] > 0]
    losses = [t["profit_abs"] for t in trades if t["profit_abs"] <= 0]
    fees = sum(t["amount"] * t["open_rate"] * t.get("fee_open", 0)
               + t["amount"] * t["close_rate"] * t.get("fee_close", 0) for t in trades)
    keys = ["profit_total", "profit_total_abs", "max_drawdown_account", "max_drawdown_abs",
            "profit_factor", "sharpe", "sortino", "calmar", "winrate", "market_change",
            "starting_balance", "final_balance", "backtest_start", "backtest_end", "holding_avg", "p_value"]
    out = {k: strategy_stats.get(k) for k in keys}
    out.update({
        "total_trades": n,
        "avg_win_abs": sum(wins) / len(wins) if wins else None,
        "avg_loss_abs": sum(losses) / len(losses) if losses else None,
        "total_cost_abs_incl_slippage": round(fees, 4),
        "effective_fee_per_side": cost_model.effective_fee_per_side,
        "exposure": exposure(trades, strategy_stats["backtest_start"], strategy_stats["backtest_end"],
                             int(strategy_stats.get("max_open_trades", 1))),
        "low_sample_warning": n < MIN_TRADES_FOR_STATS,
    })
    return out


def run_freqtrade_backtest(
    *,
    strategies: list[str],
    timeframe: str,
    timerange: str | None,
    datadir: Path,
    data_source: str,
    offline: bool,
    registry: Registry | None,
    results_dir: Path | None,
    config_path: Path | None,
    extra_metrics: dict[str, Any] | None = None,
    extra_config_files: list[Path] | None = None,
    extra_argv: list[str] | None = None,
    user_data_dir: Path | None = None,
    export: str = "trades",
) -> dict[str, Any]:
    """Ein Freqtrade-Lauf für eine oder mehrere Strategien unter identischen Bedingungen."""
    assert_run_mode_allowed("backtest")
    project = load_project_config()
    config_path = config_path or paths.FREQTRADE_BASE_CONFIG
    ft_config = load_freqtrade_config(config_path)
    assert_safe_freqtrade_config(ft_config, project["allowed_exchanges"], project["allowed_pairs"])
    cost_model = CostModel.from_project_config(project)
    pairs = ft_config["exchange"]["pair_whitelist"]
    synthetic = is_synthetic_dir(datadir)
    if offline and not synthetic:
        raise ValueError("Offline-Markets sind nur für synthetische Daten zulässig.")

    # 1) Daten validieren – bei Fehlern kein Backtest.
    fingerprints, benchmark = {}, {}
    for pair in pairs:
        df = _slice(load_ohlcv_raw(datadir, pair, timeframe), timerange)
        rep = validate_ohlcv(df, pair, timeframe, data_source,
                             max_abs_log_return=project["validation"]["max_abs_log_return"].get(timeframe, 0.3))
        if registry:
            registry.add_validation(rep.to_dict())
        if not rep.ok:
            raise DataQualityError(f"{pair} {timeframe}: Daten fehlerhaft: "
                                   + "; ".join(i.message for i in rep.issues if i.severity == "ERROR"))
        fingerprints[pair] = rep.fingerprint
        benchmark[pair] = buy_and_hold(df, cost_model)
    benchmark["equal_weight_net_return"] = sum(b["net_return"] for b in benchmark.values()) / len(pairs)

    # 2) Freqtrade-Backtest
    from freqtrade import __version__ as ft_version
    from freqtrade.commands import Arguments, start_backtesting
    from freqtrade.data.btanalysis import get_latest_backtest_filename, load_backtest_stats

    results_dir = results_dir or paths.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    for extra in extra_config_files or []:
        overlay = load_freqtrade_config(extra)
        if set(overlay) - {"freqai"}:
            raise SafetyError(f"Zusatzkonfiguration {extra} darf nur 'freqai' enthalten.")
    argv = [
        "backtesting", "-c", str(config_path), *[a for e in extra_config_files or [] for a in ("-c", str(e))],
        "--userdir", str(user_data_dir or paths.USER_DATA),
        "--datadir", str(datadir), "--strategy-list", *strategies,
        "--strategy-path", str(paths.USER_DATA / "strategies"), "--timeframe", timeframe,
        "--fee", str(cost_model.effective_fee_per_side), "--export", export,
        "--backtest-directory", str(results_dir), "--data-format-ohlcv", "parquet",
        "--cache", "none", "--notes", f"kitb data_source={data_source} synthetic={synthetic}",
    ]
    if timerange:
        argv += ["--timerange", timerange]
    argv += extra_argv or []
    args = Arguments(argv).get_parsed_arg()
    with offline_markets() if offline else nullcontext():
        start_backtesting(args)

    result_file = results_dir / get_latest_backtest_filename(results_dir)
    stats = load_backtest_stats(result_file)
    out: dict[str, Any] = {"result_file": str(result_file), "data_fingerprints": fingerprints,
                           "strategies": {}}
    for strategy in strategies:
        strat_stats = stats["strategy"][strategy]
        metrics = summarize(strat_stats, cost_model)
        metrics["buy_and_hold"] = benchmark
        metrics["excess_vs_buy_and_hold"] = (
            metrics["profit_total"] - benchmark["equal_weight_net_return"])
        metrics["synthetic_data"] = synthetic
        metrics.update(extra_metrics or {})
        run_id = None
        if registry:
            run_id = registry.add_backtest(
                strategy=strategy, timeframe=timeframe, timerange=timerange, data_source=data_source,
                synthetic=synthetic, data_fingerprints=fingerprints,
                config_sha256=_sha256_file(config_path), cost_model=cost_model.__dict__,
                freqtrade_version=ft_version, result_file=str(result_file), metrics=metrics)
        out["strategies"][strategy] = {"run_id": run_id, "metrics": metrics,
                                       "trades": strat_stats.get("trades", [])}
    return out


def run_backtest(
    *,
    strategy: str = "BaselineEmaCross",
    timeframe: str = "1h",
    timerange: str | None = None,
    datadir: Path,
    data_source: str,
    offline: bool = False,
    registry: Registry | None = None,
    results_dir: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    res = run_freqtrade_backtest(strategies=[strategy], timeframe=timeframe, timerange=timerange,
                                 datadir=datadir, data_source=data_source, offline=offline,
                                 registry=registry, results_dir=results_dir, config_path=config_path)
    one = res["strategies"][strategy]
    return {"run_id": one["run_id"], "result_file": res["result_file"], "metrics": one["metrics"],
            "data_fingerprints": res["data_fingerprints"]}
