"""Schlanker Backtest der Bot-Signale über feste Zeiträume (Burak 26.10.2026, Aufgaben 3–5 und Schritte 2–4).

Wiederverwendet wird die Signal-Logik der Strategien aus ``user_data/strategies`` (``populate_indicators``,
``populate_entry_trend``, ``populate_exit_trend`` und ``stoploss``), nichts davon ist neu geschrieben. Ausgeführt wird
mit einem eigenen, kleinen Ablauf statt der Freqtrade-Engine:

- **Ohne Lookahead:** Signal aus der abgeschlossenen Kerze i, Ausführung zum Open der Kerze i+1. Ein Einstieg zählt
  nicht, wenn in derselben Kerze auch ein Ausstiegssignal steht (wie Freqtrade).
- **Stoploss:** ab der Einstiegskerze auf dem Tief geprüft; liegt schon das Open darunter, wird zum Open verkauft.
  Der Stop wird vor dem Ausstiegssignal derselben Kerze geprüft (der Stop greift während der Kerze, das Signal erst
  an ihrem Schluss).
- **Kosten:** Gebühr + Slippage je Seite (Standard 0,1 % + 0,05 %), Stresslauf mit doppelten Kosten.
- **Zeiträume:** je ``--period`` startet das Konto ohne Position am ersten Tag. Die Indikatoren laufen auf allen
  Kerzen davor warm. Am Ende wird eine offene Position zum letzten Schlusskurs geschlossen (wie Freqtrade, Abgleich in
  backtest/engine_check.py). Standard sind die letzten 3 Jahre bis Datenende. Getestet werden nur feste Zeitebenen.
  **Verworfen (Burak 26.10.2026):** die Walk-Forward-Wahl der Zeitebene („<stil>_wf“), weil sie schlechter abschnitt
  als fest 4h. Ohne angepasste Parameter hat ein Trainingsfenster keine Aufgabe mehr.
- **Drawdown:** auf der täglichen, zum Schlusskurs bewerteten Kontokurve (mit offenen Positionen). Freqtrade misst
  ihn nur auf abgeschlossenen Trades.
- **Kapitalmodell:** voll investiert, je Paar die Hälfte, eine Position je Paar, keine Umschichtung. Die Paper-Bots setzen über ``RiskManagedMixin`` nur etwa
  20–40 % je Trade ein, ihre Rendite und ihr Drawdown wären entsprechend kleiner.
- **Signifikanz gegen trend_d1:** einseitiger p-Wert aus einem Moving-Block-Bootstrap (Blöcke à 10 Tage) der täglichen
  Renditedifferenz im Zeitraum, je Zeitraum über alle getesteten Bots nach Holm korrigiert.

**Nicht enthalten:** die KI-Entscheidung. Im Backtest sind ``StyleAiMixin`` und das gemeinsame Gedächtnis aus
(kitb.rule_learning: „Ohne kitb_style_ai … verhält sich die Strategie unverändert“). Getestet wird also, was der Stil
vorschlägt, nicht, was die KI daraus macht. bot_a, bot_b, ki_trader (FreqAI) und ki_reward (Bandit auf dem
Live-Gedächtnis) lassen sich so nicht ohne Lookahead nachstellen und fehlen.

Kerzen: lokaler Cache ``user_data/data/binance/<PAIR>-<tf>.parquet`` (Freqtrade-Format). Fehlt etwas, vorher
``kitb download`` ausführen. Die letzte Kerze jeder Datei wird verworfen, weil sie unvollständig sein kann.

Aufruf:  python backtest/run.py [--pairs BTC/USDT ETH/USDT] [--period baer2022=20220101-20230101 ...]
         [--bots trend_d1 donchian_4h ...] [--out backtest/output]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "user_data" / "strategies")]

BASELINE = "trend_d1"
MIN_TRADES = 30  # darunter erlaubt das Bootstrap-Intervall keine Aussage (wie kitb.backtest.MIN_TRADES_FOR_STATS)
TF_MINUTES = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}
# Name im Paper-Betrieb -> Strategieklasse. Stil-Bots laufen dort auf 15m und 1h; 4h ist der Zusatz für Aufgabe 5.
STYLES = {"ema_cross": "BaselineEmaCross", "trend_adx": "TrendEmaAdx", "macd_rsi": "MomentumMacdRsi",
          "donchian": "BreakoutDonchian", "squeeze": "VolatilitySqueeze"}
DAILY = {"trend_d1": "DailyTrendSma", "momentum_d1": "DailyMomentum", "trend_btc_d1": "TrendSmaBtc"}


class _DataProvider:
    """Nur was TrendSmaBtc braucht: Kerzen eines anderen Paars derselben Zeitebene."""

    runmode = "backtest"

    def __init__(self, candles: dict):
        self._candles = candles

    def get_pair_dataframe(self, pair: str, timeframe: str) -> pd.DataFrame:
        return self._candles[(pair, timeframe)].copy()


def load_candles(datadir: Path, pair: str, tf: str) -> pd.DataFrame:
    f = datadir / f"{pair.replace('/', '_')}-{tf}.parquet"
    if not f.exists():
        raise SystemExit(f"Kerzen fehlen: {f} – vorher z. B. `kitb download` ausführen.")
    df = pd.read_parquet(f).sort_values("date").iloc[:-1].reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def signals(cls_name: str, tf: str, pair: str, candles: dict) -> pd.DataFrame:
    mod = __import__(cls_name)
    strat = getattr(mod, cls_name)({"timeframe": tf, "stake_currency": "USDT", "dry_run": True, "runmode": "backtest"})
    strat.timeframe = tf
    strat.dp = _DataProvider(candles)
    meta = {"pair": pair}
    df = strat.populate_indicators(candles[(pair, tf)].copy(), meta)
    df = strat.populate_exit_trend(strat.populate_entry_trend(df, meta), meta)
    for c in ("enter_long", "exit_long"):
        df[c] = df[c].fillna(0).astype(int) if c in df else 0
    return df[["date", "open", "high", "low", "close", "enter_long", "exit_long"]].assign(stoploss=strat.stoploss)


def simulate(df: pd.DataFrame, cost: float) -> tuple[pd.DataFrame, pd.Series]:
    """Trades und Kontokurve (Schlusskurs-bewertet, Start 1,0) eines Paars."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    enter, exit_ = df["enter_long"].to_numpy(), df["exit_long"].to_numpy()
    stop_pct = float(df["stoploss"].iloc[0])
    n = len(df)
    eq = np.empty(n)
    cash, entry, entry_i, stop = 1.0, None, None, None
    trades = []
    pending_entry = pending_exit = False
    for i in range(n):
        if pending_entry:  # Signal der Vorkerze -> Kauf zum Open
            entry, entry_i, stop = o[i], i, o[i] * (1 + stop_pct)
            cash *= 1 - cost
            pending_entry = False
        if entry is not None:
            px = None
            if pending_exit:
                px, why = o[i], "signal"
            elif o[i] <= stop:
                px, why = o[i], "stop"
            elif l[i] <= stop:
                px, why = stop, "stop"
            if px is not None:
                r = px / entry * (1 - cost) ** 2 - 1
                trades.append((df["date"].iat[entry_i], df["date"].iat[i], r, why))
                cash *= px / entry * (1 - cost)
                entry = None
            pending_exit = False
        eq[i] = cash * (c[i] / entry if entry is not None else 1.0)
        if entry is None and enter[i] == 1 and exit_[i] != 1 and i + 1 < n:
            pending_entry = True
        elif entry is not None and exit_[i] == 1:
            pending_exit = True
    if entry is not None:  # wie Freqtrade: offene Position am Ende zum letzten Schlusskurs schließen
        trades.append((df["date"].iat[entry_i], df["date"].iat[n - 1], c[-1] / entry * (1 - cost) ** 2 - 1, "ende"))
        eq[-1] = cash * c[-1] / entry * (1 - cost)
    t = pd.DataFrame(trades, columns=["open_date", "close_date", "ret", "exit"])
    return t, pd.Series(eq, index=df["date"])


def bootstrap_ci(x: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def block_bootstrap_p(diff: np.ndarray, n_boot: int, seed: int, block: int = 10) -> float:
    """Einseitiger p-Wert für „Mittel der täglichen Differenz > 0“ (Moving-Block-Bootstrap, hält Autokorrelation)."""
    n = len(diff)
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    means = diff[idx].mean(axis=1)
    # Verteilung um 0 zentriert: wie oft ist ein Mittel >= dem beobachteten, wenn der wahre Vorteil 0 wäre?
    return float(((means - diff.mean()) >= diff.mean()).mean())


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values()
    m = len(order)
    adj = np.maximum.accumulate([(m - k) * v for k, v in enumerate(order.to_numpy())])
    return pd.Series(np.minimum(adj, 1.0), index=order.index).reindex(p.index)


def parse_period(text: str) -> tuple[str, pd.Timestamp, pd.Timestamp | None]:
    """``name=YYYYMMDD-YYYYMMDD`` (Ende ausschließlich; leeres Ende = bis zum Datenende)."""
    name, _, rng = text.partition("=")
    a, _, b = rng.partition("-")
    return name, pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC") if b else None


def run(pairs: list[str], datadir: Path, periods: list[tuple], cost: float, n_boot: int, seed: int,
        bots: list[str] | None = None) -> pd.DataFrame:
    variants = {f"{s}_{tf}": (cls, tf) for s, cls in STYLES.items() for tf in ("15m", "1h", "4h")}
    variants |= {name: (cls, "1d") for name, cls in DAILY.items()}
    if bots:
        variants = {k: v for k, v in variants.items() if k in set(bots) | {BASELINE}}
    tfs = {tf for _, tf in variants.values()} | {"1d"}
    candles = {(p, tf): load_candles(datadir, p, tf) for p in set(pairs) | {"BTC/USDT"} for tf in tfs}
    data_end = min(candles[(p, "1d")]["date"].max() for p in pairs) + pd.Timedelta("1D")
    sig = {(name, p): signals(cls, tf, p, candles) for name, (cls, tf) in variants.items() for p in pairs}
    btc = candles[("BTC/USDT", "1d")]

    out = []
    for pname, start, end in periods:
        end = min(end or data_end, data_end)
        curves, trades = {}, {}
        for name in variants:
            res = []
            for p in pairs:
                s = sig[(name, p)]
                res.append(simulate(s[(s["date"] >= start) & (s["date"] < end)].reset_index(drop=True), cost))
            daily = [e.resample("1D").last().ffill() for _, e in res]
            curves[name] = pd.concat(daily, axis=1).ffill().bfill().mean(axis=1)
            trades[name] = pd.concat([t for t, _ in res], ignore_index=True)
        b = btc[(btc["date"] >= start) & (btc["date"] < end)]
        btc_eq = b["close"] / b["open"].iloc[0] * (1 - cost)
        btc_hold = float(btc_eq.iloc[-1] * (1 - cost) - 1)
        btc_dd = float((btc_eq / btc_eq.cummax() - 1).min())
        base_eq = curves[BASELINE]
        base = base_eq.pct_change().fillna(base_eq.iloc[0] - 1)
        rows = []
        for name in variants:
            eq = curves[name]
            d = eq.pct_change().fillna(eq.iloc[0] - 1)
            t = trades[name]
            x = t["ret"].to_numpy(float)
            lo, hi = bootstrap_ci(x, n_boot, seed)
            diff = (d - base.reindex(d.index).fillna(0)).to_numpy()
            rows.append({
                "zeitraum": pname, "von": f"{start:%Y-%m-%d}", "bis": f"{end - pd.Timedelta('1D'):%Y-%m-%d}",
                "bot": name, "trades": len(t),
                "erster_trade": f"{t['open_date'].min():%Y-%m-%d}" if len(t) else "",
                "zu_wenige_trades": len(t) < MIN_TRADES,
                "netto_rendite_pct": round((eq.iloc[-1] - 1) * 100, 2),
                "max_drawdown_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 2),
                "je_trade_pct": round(x.mean() * 100, 3) if len(x) else np.nan,
                "ci95_low_pct": round(lo * 100, 3), "ci95_high_pct": round(hi * 100, 3),
                "zufall_nicht_unterscheidbar": bool(lo <= 0 <= hi) if len(x) >= 2 else None,
                "btc_halten_pct": round(btc_hold * 100, 2), "btc_halten_dd_pct": round(btc_dd * 100, 2),
                "vs_btc_halten_pp": round((eq.iloc[-1] - 1 - btc_hold) * 100, 2),
                "vs_trend_d1_pp": round((eq.iloc[-1] - base_eq.iloc[-1]) * 100, 2),
                "p_besser_als_trend_d1": np.nan if name == BASELINE else block_bootstrap_p(diff, n_boot, seed),
            })
        res = pd.DataFrame(rows)
        test = res["bot"] != BASELINE
        res.loc[test, "p_holm"] = holm(res.loc[test, "p_besser_als_trend_d1"]).round(4)
        res.loc[test, "signifikant_besser_als_trend_d1"] = res.loc[test, "p_holm"] < 0.05
        out.append(res)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", nargs="+", default=["BTC/USDT", "ETH/USDT"])
    ap.add_argument("--datadir", type=Path, default=ROOT / "user_data" / "data" / "binance")
    ap.add_argument("--period", action="append", type=parse_period,
                    help="name=YYYYMMDD-YYYYMMDD, mehrfach möglich (Standard: die letzten 3 Jahre bis Datenende)")
    ap.add_argument("--bots", nargs="+", help="nur diese Bots (trend_d1 ist immer dabei)")
    ap.add_argument("--fee", type=float, default=0.001, help="Gebühr je Seite (0.001 = 0,1 %)")
    ap.add_argument("--slippage", type=float, default=0.0005, help="Slippage je Seite")
    ap.add_argument("--bootstrap", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "backtest" / "output")
    args = ap.parse_args()

    if not args.period:
        end = min(load_candles(args.datadir, p, "1d")["date"].max() for p in args.pairs) + pd.Timedelta("1D")
        args.period = [("3jahre", end - pd.DateOffset(years=3), None)]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    args.out.mkdir(parents=True, exist_ok=True)
    for label, factor in (("normal", 1), ("doppelte_kosten", 2)):
        cost = (args.fee + args.slippage) * factor
        res = run(args.pairs, args.datadir, args.period, cost, args.bootstrap, args.seed, args.bots)
        print(f"\n=== Kosten {label}: {cost:.2%} je Seite")
        for _, g in res.groupby("zeitraum", sort=False):
            print(g.to_string(index=False))
        res.to_csv(args.out / f"backtest_{label}.csv", index=False)
    print(f"\nCSV in {args.out}")


if __name__ == "__main__":
    main()
