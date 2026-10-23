"""Methoden-Suche: welche bekannten Trading-Methoden hatten in der Vergangenheit ≥ 60 % Treffer? (Burak, 23.10.2026)

Jede Methode aus ``user_data/strategies/MethodLibrary.py`` wird je Zeitrahmen auf allen Lern-Coins mit Gebühren
getestet, und zwar in zwei getrennten Zeiträumen (früh und spät, beide vor dem bereits verbrauchten Test-Abschnitt).
Aufgenommen wird nur, was in **beiden** Zeiträumen
- mindestens ``MIN_WINRATE`` Treffer hat,
- nach Gebühren im Plus ist (60 % Treffer allein täuschen: viele kleine Gewinne + wenige große Verluste = Minus),
- genug Trades hat, um nicht Zufall zu sein (``MIN_TRADES``).
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from kitb import paths
from kitb.config import load_freqtrade_config, load_project_config

MIN_WINRATE = 0.60
MIN_TRADES = 30
PERIODS = {"früh": "20220301-20240630", "spät": "20240701-20251104"}  # Test-Abschnitt ab 05.11.2025 bleibt unberührt
TIMEFRAMES = ["15m", "1h", "4h"]


def method_names() -> list[str]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("MethodLibrary", paths.USER_DATA / "strategies" / "MethodLibrary.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [m.__name__ for m in mod.METHODS]


def _config(tmp: Path, pairs: list[str] | None = None) -> Path:
    """Backtest-Konfiguration für alle Lern-Coins (gleiche Kosten/Regeln wie die Basis, nur andere Marktliste)."""
    project = load_project_config()
    cfg = load_freqtrade_config()
    pairs = pairs or project.get("learning_pairs") or project["allowed_pairs"]
    cfg["exchange"] = {**cfg["exchange"], "pair_whitelist": list(pairs)}
    p = tmp / "methods-config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def scan(*, datadir: Path, data_source: str, timeframes: list[str] | None = None, methods: list[str] | None = None,
         periods: dict[str, str] | None = None, offline: bool = False, pairs: list[str] | None = None) -> pd.DataFrame:
    from kitb.backtest import run_freqtrade_backtest

    rows = []
    methods = methods or method_names()
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        cfg = _config(tmp, pairs)
        from freqtrade.configuration.directory_operations import create_userdata_dir

        create_userdata_dir(str(tmp / "ud"), create_dir=True)
        for tf in timeframes or TIMEFRAMES:
            for period, timerange in (periods or PERIODS).items():
                res = run_freqtrade_backtest(strategies=methods, timeframe=tf, timerange=timerange, datadir=datadir,
                                             data_source=data_source, offline=offline, registry=None,
                                             results_dir=tmp / "results", config_path=cfg, user_data_dir=tmp / "ud")
                for name, r in res["strategies"].items():
                    m = r["metrics"]
                    months = max(1.0, (pd.Timestamp(m["backtest_end"]) - pd.Timestamp(m["backtest_start"])).days / 30.4)
                    rows.append({"Methode": name, "Zeitrahmen": tf, "Zeitraum": period, "Trades": m["total_trades"],
                                 "Trades/Monat": round(m["total_trades"] / months, 1),
                                 "Trefferquote": m["winrate"], "Ergebnis": m["profit_total"],
                                 "Profitfaktor": m["profit_factor"], "Max. Drawdown": m["max_drawdown_account"]})
    return pd.DataFrame(rows)


def select(results: pd.DataFrame) -> pd.DataFrame:
    """Je (Methode, Zeitrahmen): bestanden, wenn alle Zeiträume die Regeln erfüllen."""
    if results.empty:
        return results
    ok = ((results["Trefferquote"] >= MIN_WINRATE) & (results["Ergebnis"] > 0) & (results["Trades"] >= MIN_TRADES))
    g = results.assign(ok=ok).groupby(["Methode", "Zeitrahmen"])
    out = g.agg(Trefferquote_min=("Trefferquote", "min"), Ergebnis_min=("Ergebnis", "min"),
                Trades=("Trades", "sum"), Trades_je_Monat=("Trades/Monat", "mean"), bestanden=("ok", "all"))
    return out.reset_index().sort_values(["bestanden", "Trefferquote_min"], ascending=False)


def render(results: pd.DataFrame, selection: pd.DataFrame) -> str:
    lines = ["# Methoden-Suche (≥ 60 % Treffer und Gewinn nach Gebühren, früh UND spät)", "",
             f"Regeln: Trefferquote ≥ {MIN_WINRATE:.0%}, Ergebnis nach Gebühren > 0, mindestens {MIN_TRADES} Trades – "
             f"in jedem Zeitraum ({', '.join(f'{k}: {v}' for k, v in PERIODS.items())}). Test-Abschnitt ab 05.11.2025 "
             "nicht verwendet.", "", "## Auswahl", "", selection.to_markdown(index=False, floatfmt=".3f"), "",
             "## Alle Ergebnisse", "", results.to_markdown(index=False, floatfmt=".3f")]
    return "\n".join(lines)


def run(*, data_source: str = "binance", out_dir: Path | None = None, **kw: Any) -> dict[str, Any]:
    datadir = paths.DATA_DIR / data_source
    results = scan(datadir=datadir, data_source=data_source, **kw)
    selection = select(results)
    out_dir = out_dir or (paths.USER_DATA / "reports" / f"methods-{pd.Timestamp.now(tz='UTC'):%Y%m%d-%H%M}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "methods.md").write_text(render(results, selection), encoding="utf-8")
    results.to_csv(out_dir / "results.csv", index=False)
    selection.to_csv(out_dir / "selection.csv", index=False)
    return {"report": str(out_dir / "methods.md"),
            "bestanden": selection[selection["bestanden"]][["Methode", "Zeitrahmen"]].to_dict("records")}
