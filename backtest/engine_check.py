"""Abgleich des eigenen Backtesters (backtest/run.py) mit ``freqtrade backtesting`` (Burak 26.10.2026, Schritt 3).

Gleiche Paare, gleicher Zeitraum, gleiche Kosten (Gebühr + Slippage als ``--fee`` je Seite), gleicher Stop aus der
Strategie, Startkapital 10.000 USDT, höchstens 2 offene Positionen (eine je Paar). Beide starten ohne Position am
Anfang des Zeitraums. Die Indikatoren laufen auf den Kerzen davor warm, und am Ende wird jede offene Position zum
letzten Schlusskurs geschlossen.

Freqtrade läuft ohne Internet mit den festen Marktdaten aus ``kitb.offline`` (Präzision und Mindestgröße angenommen).

Zwei Freqtrade-Läufe je Bot:
- ``ft_risiko``: die Strategie unverändert, also mit ``RiskManagedMixin``. Das ist die Positionsgröße aus
  config/risk.json wie im Paper-Betrieb, dazu Tagesverlust-Sperre usw.
- ``ft_halb``: je Trade die Hälfte des Kontos (``stake_amount: unlimited`` bei 2 Plätzen), Risikomodul im
  Backtest neutralisiert. Das ist das Kapitalmodell von run.py.

Aufruf:  python backtest/engine_check.py --datadir user_data/data/binance --timerange 20231023-20261023
"""

import argparse
import json
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402

BOTS = {"trend_d1": ("DailyTrendSma", "1d"), "donchian_4h": ("BreakoutDonchian", "4h")}


def own(cls: str, tf: str, pairs: list[str], datadir: Path, start, end, cost: float) -> tuple[pd.DataFrame, pd.Series]:
    candles = {(p, tf): run.load_candles(datadir, p, tf) for p in pairs + ["BTC/USDT"]}
    trades, curves = [], []
    for p in pairs:
        s = run.signals(cls, tf, p, candles)
        s = s[(s["date"] >= start) & (s["date"] < end)].reset_index(drop=True)
        t, e = run.simulate(s, cost)
        trades.append(t.assign(pair=p))
        curves.append(e.resample("1D").last().ffill())
    eq = pd.concat(curves, axis=1).ffill().bfill().mean(axis=1)
    return pd.concat(trades, ignore_index=True), eq


def freqtrade(cls: str, tf: str, pairs: list[str], datadir: Path, timerange: str, cost: float,
              neutral_risk: bool, workdir: Path) -> tuple[pd.DataFrame, dict]:
    from freqtrade.commands import Arguments, start_backtesting
    from freqtrade.data.btanalysis import get_latest_backtest_filename, load_backtest_stats

    from kitb.offline import offline_markets
    from kitb.risk import RiskManagedMixin

    base = json.loads((run.ROOT / "config" / "freqtrade.base.json").read_text(encoding="utf-8"))
    base |= {"stake_amount": "unlimited", "max_open_trades": 2, "dry_run_wallet": 10_000,
             "tradable_balance_ratio": 1.0, "timeframe": tf}
    base["exchange"]["pair_whitelist"] = pairs
    cfg = workdir / f"cfg-{cls}-{tf}.json"
    cfg.write_text(json.dumps(base), encoding="utf-8")
    out = workdir / f"res-{cls}-{tf}-{'halb' if neutral_risk else 'risiko'}"
    out.mkdir(parents=True, exist_ok=True)
    argv = ["backtesting", "-c", str(cfg), "--userdir", str(run.ROOT / "user_data"), "--datadir", str(datadir),
            "--strategy", cls, "--strategy-path", str(run.ROOT / "user_data" / "strategies"), "--timeframe", tf,
            "--fee", str(cost), "--timerange", timerange, "--export", "trades", "--backtest-directory", str(out),
            "--data-format-ohlcv", "parquet", "--cache", "none"]
    with ExitStack() as stack:
        stack.enter_context(offline_markets())
        if neutral_risk:
            stack.enter_context(mock.patch.object(
                RiskManagedMixin, "custom_stake_amount",
                lambda self, pair, current_time, current_rate, proposed_stake, *a, **k: proposed_stake))
            stack.enter_context(mock.patch.object(
                RiskManagedMixin, "confirm_trade_entry", lambda self, *a, side="long", **k: side == "long"))
        start_backtesting(Arguments(argv).get_parsed_arg())
    stats = load_backtest_stats(out / get_latest_backtest_filename(out))["strategy"][cls]
    t = pd.DataFrame(stats["trades"])
    for c in ("open_date", "close_date"):
        t[c] = pd.to_datetime(t[c], utc=True)
    return t, stats


def compare(name: str, mine: pd.DataFrame, eq: pd.Series, ft: pd.DataFrame, stats: dict, label: str) -> dict:
    m = mine.sort_values(["pair", "open_date"]).reset_index(drop=True)
    f = ft.sort_values(["pair", "open_date"]).reset_index(drop=True)
    j = m.merge(f[["pair", "open_date", "close_date", "profit_ratio", "exit_reason"]], on=["pair", "open_date"],
                how="outer", suffixes=("", "_ft"), indicator=True)
    both = j[j["_merge"] == "both"]
    dd_ft = -float(stats.get("max_drawdown_account", np.nan)) * 100
    return {
        "bot": name, "lauf": label, "trades_eigen": len(m), "trades_ft": len(f),
        "nur_eigen": int((j["_merge"] == "left_only").sum()), "nur_ft": int((j["_merge"] == "right_only").sum()),
        "gleicher_ausstieg": int((both["close_date"] == both["close_date_ft"]).sum()),
        "max_abw_je_trade_pp": round(float((both["ret"] - both["profit_ratio"]).abs().max() * 100), 4)
        if len(both) else np.nan,
        "rendite_eigen_pct": round((eq.iloc[-1] - 1) * 100, 2),
        "rendite_ft_pct": round(float(stats["profit_total"]) * 100, 2),
        "dd_eigen_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 2),
        "dd_ft_pct": round(dd_ft, 2),
    }, j


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", nargs="+", default=["BTC/USDT", "ETH/USDT"])
    ap.add_argument("--datadir", type=Path, default=run.ROOT / "user_data" / "data" / "binance")
    ap.add_argument("--timerange", default="20231023-20261023")
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--slippage", type=float, default=0.0005)
    ap.add_argument("--out", type=Path, default=run.ROOT / "backtest" / "output")
    args = ap.parse_args()

    cost = args.fee + args.slippage
    a, _, b = args.timerange.partition("-")
    start, end = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, (cls, tf) in BOTS.items():
            mine, eq = own(cls, tf, args.pairs, args.datadir, start, end, cost)
            mine.to_csv(args.out / f"engine_trades_eigen_{name}.csv", index=False)
            for neutral, label in ((True, "ft_halb"), (False, "ft_risiko")):
                ft, stats = freqtrade(cls, tf, args.pairs, args.datadir, args.timerange, cost, neutral, Path(tmp))
                row, joined = compare(name, mine, eq, ft, stats, label)
                rows.append(row)
                joined.to_csv(args.out / f"engine_vergleich_{name}_{label}.csv", index=False)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(res.to_string(index=False))
    res.to_csv(args.out / "engine_check.csv", index=False)


if __name__ == "__main__":
    main()
