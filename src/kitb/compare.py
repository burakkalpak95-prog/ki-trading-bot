"""Fairer Strategievergleich: gleiche Daten, Kosten, Kapital und Limits für alle Strategien.

Pro Abschnitt (Train, Validation, optional Test) und pro rollierendem Fenster läuft
**ein** Freqtrade-Backtest mit ``--strategy-list``. Ergebnis: JSON + Markdown-Bericht
und je Strategie/Abschnitt ein Eintrag in SQLite.
"""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from kitb import paths
from kitb.backtest import run_freqtrade_backtest
from kitb.config import load_freqtrade_config, load_project_config
from kitb.data.storage import load_ohlcv_raw
from kitb.data.synthetic import resample
from kitb.data.timeframes import to_timedelta
from kitb.evaluation import (Segment, attribute_trades_to_regimes, chronological_split, classify_regimes,
                             regime_summary, rolling_windows)
from kitb.registry import Registry

TABLE_KEYS = ["total_trades", "profit_total", "max_drawdown_account", "profit_factor", "sharpe", "winrate",
              "exposure", "total_cost_abs_incl_slippage", "excess_vs_buy_and_hold", "low_sample_warning"]


def data_span(datadir: Path, pairs: list[str], timeframe: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Gemeinsamer Zeitraum aller Paare (spätester Start, frühestes Ende)."""
    starts, ends = [], []
    for pair in pairs:
        df = load_ohlcv_raw(datadir, pair, timeframe)
        if df.empty:
            raise ValueError(f"Keine Daten für {pair} {timeframe} in {datadir}")
        starts.append(df["date"].iloc[0])
        ends.append(df["date"].iloc[-1] + to_timedelta(timeframe))
    return max(starts), min(ends)


def _daily(datadir: Path, pair: str) -> pd.DataFrame:
    daily = load_ohlcv_raw(datadir, pair, "1d")
    if daily.empty:
        daily = resample(load_ohlcv_raw(datadir, pair, "1h"), "1d")
    return daily


def compare_strategies(
    *,
    datadir: Path,
    data_source: str,
    strategies: list[str] | None = None,
    timeframe: str = "1h",
    include_test: bool = False,
    with_windows: bool = True,
    offline: bool = False,
    registry: Registry | None = None,
    results_dir: Path | None = None,
    report_dir: Path | None = None,
) -> dict[str, Any]:
    project = load_project_config()
    ev = project["evaluation"]
    strategies = strategies or (ev["daily_strategies"] if timeframe == "1d" else ev["strategies"])
    pairs = load_freqtrade_config()["exchange"]["pair_whitelist"]
    compare_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]

    start, end = data_span(datadir, pairs, timeframe)
    segments = chronological_split(start, end, ev["split_fractions"])
    evaluated = [s for s in segments if s.name != "test" or include_test]
    if include_test and registry:
        registry.log_source_event("evaluation", "test_set_used", "ok",
                                  {"compare_id": compare_id, "test": segments[-1].timerange})

    def run(seg: Segment, kind: str) -> dict[str, Any]:
        return run_freqtrade_backtest(
            strategies=strategies, timeframe=timeframe, timerange=seg.timerange, datadir=datadir,
            data_source=data_source, offline=offline, registry=registry, results_dir=results_dir,
            config_path=None, extra_metrics={"compare_id": compare_id, "segment": seg.name, "kind": kind})

    segment_results = {seg.name: run(seg, "split") for seg in evaluated}

    # Marktphasen über alle ausgewerteten Abschnitte
    ev_regime = ev["regime"]
    regimes = {p: classify_regimes(_daily(datadir, p), ev_regime["trend_sma_days"],
                                   ev_regime["slope_lookback_days"], ev_regime["vol_days"]) for p in pairs}
    by_regime = {}
    for strat in strategies:
        trades = [t for r in segment_results.values() for t in r["strategies"][strat]["trades"]]
        by_regime[strat] = regime_summary(attribute_trades_to_regimes(trades, regimes))

    # Rollierende Fenster über den ausgewerteten Zeitraum
    windows: dict[str, list[dict]] = {s: [] for s in strategies}
    if with_windows:
        # Fenster erst dort beginnen, wo genug Vorlauf für alle Indikatoren vorhanden ist (sonst bricht Freqtrade ab
        # oder kürzt das Fenster unbemerkt).
        from kitb.strategies import load_strategy

        warmup = max(load_strategy(s).startup_candle_count for s in strategies) * to_timedelta(timeframe)
        first = max(evaluated[0].start, (start + warmup).ceil("D"))
        for win in rolling_windows(first, evaluated[-1].end, ev["window_days"], ev["step_days"]):
            res = run(win, "window")
            for strat in strategies:
                m = res["strategies"][strat]["metrics"]
                windows[strat].append({"window": win.timerange, "profit_total": m["profit_total"],
                                       "excess_vs_buy_and_hold": m["excess_vs_buy_and_hold"],
                                       "trades": m["total_trades"]})

    report = {
        "compare_id": compare_id,
        "data_source": data_source,
        "synthetic_data": next(iter(segment_results.values()))["strategies"][strategies[0]]["metrics"]["synthetic_data"],
        "timeframe": timeframe,
        "segments": {s.name: s.timerange for s in segments},
        "test_set_used": include_test,
        "results": {
            strat: {
                "segments": {name: {k: r["strategies"][strat]["metrics"].get(k) for k in TABLE_KEYS}
                             for name, r in segment_results.items()},
                "buy_and_hold": {name: r["strategies"][strat]["metrics"]["buy_and_hold"]["equal_weight_net_return"]
                                 for name, r in segment_results.items()},
                "buy_and_hold_max_drawdown": {
                    name: r["strategies"][strat]["metrics"]["buy_and_hold"].get("equal_weight_max_drawdown")
                    for name, r in segment_results.items()},
                "regimes": by_regime[strat],
                "windows": windows[strat],
                "window_stability": _stability(windows[strat]),
            }
            for strat in strategies
        },
    }
    report_dir = (report_dir or paths.USER_DATA / "reports") / f"compare-{compare_id}"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (report_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    report["report_dir"] = str(report_dir)
    return report


def _stability(windows: list[dict]) -> dict[str, Any]:
    if not windows:
        return {"windows": 0}
    n = len(windows)
    return {"windows": n,
            "share_positive": round(sum(w["profit_total"] > 0 for w in windows) / n, 3),
            "share_beating_buy_and_hold": round(sum(w["excess_vs_buy_and_hold"] > 0 for w in windows) / n, 3)}


def _pct(v: Any) -> str:
    return "–" if v is None else f"{v * 100:.2f} %"


def _num(v: Any) -> str:
    return "–" if v is None else f"{v:.2f}"


def return_to_drawdown(profit: Any, drawdown: Any) -> float | None:
    """Rendite je Einheit maximalen Drawdowns – vergleicht Strategien mit unterschiedlichem Kapitaleinsatz fairer."""
    if profit is None or not drawdown:
        return None
    return profit / drawdown


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Strategievergleich {report['compare_id']}", ""]
    if report["synthetic_data"]:
        lines += ["> **SYNTHETISCHE DATEN** – reiner Funktionstest, keine Aussage über reale Märkte.", ""]
    lines += [f"Datenquelle: `{report['data_source']}` · Zeitintervall: `{report['timeframe']}` · "
              f"Test-Abschnitt verwendet: **{'ja' if report['test_set_used'] else 'nein (unberührt)'}**", "",
              "Abschnitte: " + ", ".join(f"{k} `{v}`" for k, v in report["segments"].items()), ""]
    seg_names = list(next(iter(report["results"].values()))["segments"])
    for seg in seg_names:
        lines += [f"## Abschnitt: {seg}", "",
                  "| Strategie | Trades | Netto | Max DD | Rendite/DD | Profit Factor | Sharpe | Trefferquote | Exposure | Kosten (USDT) | vs. Buy&Hold | Stichprobe |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for strat, res in report["results"].items():
            m = res["segments"][seg]
            lines.append(f"| {strat} | {m['total_trades']} | {_pct(m['profit_total'])} | {_pct(m['max_drawdown_account'])} | "
                         f"{_num(return_to_drawdown(m['profit_total'], m['max_drawdown_account']))} | "
                         f"{_num(m['profit_factor'])} | {_num(m['sharpe'])} | {_pct(m['winrate'])} | {_pct(m['exposure'])} | "
                         f"{_num(m['total_cost_abs_incl_slippage'])} | {_pct(m['excess_vs_buy_and_hold'])} | "
                         f"{'zu klein' if m['low_sample_warning'] else 'ok'} |")
        first = next(iter(report["results"].values()))
        bh, bh_dd = first["buy_and_hold"][seg], first.get("buy_and_hold_max_drawdown", {}).get(seg)
        lines += ["", f"Buy&Hold (BTC/ETH gleichgewichtet, nach Kosten): {_pct(bh)} · Max DD {_pct(bh_dd)} · "
                      f"Rendite/DD {_num(return_to_drawdown(bh, bh_dd))}", ""]
    lines += ["## Stabilität über rollierende Fenster", "",
              "| Strategie | Fenster | Anteil positiv | Anteil besser als Buy&Hold |", "|---|---:|---:|---:|"]
    for strat, res in report["results"].items():
        s = res["window_stability"]
        lines.append(f"| {strat} | {s['windows']} | {_pct(s.get('share_positive'))} | "
                     f"{_pct(s.get('share_beating_buy_and_hold'))} |")
    lines += ["", "## Ergebnis nach Marktphase (beim Einstieg bekannte Phase)", ""]
    for strat, res in report["results"].items():
        lines += [f"**{strat}**", "", "| Trend | Volatilität | Trades | Gewinn (USDT) | Ø Gewinn/Trade | Trefferquote | Stichprobe |",
                  "|---|---|---:|---:|---:|---:|---|"]
        for r in res["regimes"]:
            lines.append(f"| {r['trend']} | {r['volatility']} | {r['trades']} | {_num(r['profit_abs'])} | "
                         f"{_pct(r['avg_profit_ratio'])} | {_pct(r['winrate'])} | {'zu klein' if r['low_sample'] else 'ok'} |")
        lines.append("")
    lines += ["Hinweise: Buy&Hold ist voll investiert, die Strategien wegen der Risikolimits höchstens zu einem Teil. "
              "Rendite/DD vergleicht deshalb fairer als die reine Rendite. "
              "Kennzahlen mit weniger als 30 Trades sind statistisch kaum belastbar. "
              "Vergangene Ergebnisse garantieren keine zukünftigen Gewinne."]
    return "\n".join(lines) + "\n"
