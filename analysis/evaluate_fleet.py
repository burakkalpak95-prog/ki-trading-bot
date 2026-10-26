"""Auswertung der bisherigen Paper-Trades je Bot (Burak 26.10.2026): Hat irgendein Bot nach Kosten einen echten Vorteil?

Nur lesend. Quelle: die Freqtrade-Dry-Run-DBs ``user_data/paper/<bot>/tradesv3.dryrun.sqlite`` plus archivierte Runden
``user_data/paper/<bot>/rounds/round-*.sqlite`` (Tabelle ``trades``, nur geschlossene Trades).

- Netto      = ``close_profit_abs`` (Freqtrade, nach Gebühr).
- Gebühr     = ``fee_open_cost + fee_close_cost`` (0,1 % je Seite laut Paper-Config).
- Brutto     = Netto + Gebühr (= ``amount * (close_rate - open_rate)``, nur Long/Spot).
- Spread     steckt schon in den Kursen (price_side "other": Kauf zum Ask, Verkauf zum Bid) und ist nicht getrennt
               gespeichert. „Vor Kosten“ heißt hier also nur „vor Gebühren“. Slippage simuliert der Dry-Run nicht.
- Lern-Trade = ``enter_tag`` beginnt mit "lern" oder "ki_uebt" (gleiche Regel wie ``kitb.dashboard.data.trade_kind``).
- Ertrag je Trade = ``close_profit`` (Anteil des Einsatzes nach Gebühr), 95-%-Intervall per Bootstrap des Mittelwerts.
  Annahme: Trades unabhängig. Das stimmt nicht ganz (gleichzeitige Trades im selben Markt), das Intervall ist eher zu eng.

Vergleich je Bot über seinen eigenen Zeitraum (erster Einstieg bis letzter Ausstieg):
- trend_d1: Wertänderung seiner Positionen in diesem Zeitraum (auch offene, zum Kurs bewertet).
- BTC halten: 10.000 USDT am Anfang in BTC, am Ende verkauft, mit Kosten je Seite aus project.json (Gebühr+Slippage+Spread).
Kurse: lokale 15m-Kerzen ``user_data/data/binance/<PAIR>-15m.parquet``; nach deren Ende die Ausführungskurse der
Paper-Trades selbst (letzter bekannter Kurs vor dem Zeitpunkt).

Aufruf:  python analysis/evaluate_fleet.py [--user-data user_data] [--out analysis/output]
"""

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START_CAPITAL = 10_000.0
BASELINE = "trend_d1"
KINDS = ("Alle", "Signal-Trade", "Lern-Trade")


def trade_kind(tag) -> str:
    return "Lern-Trade" if str(tag or "").startswith(("lern", "ki_uebt")) else "Signal-Trade"


def load_trades(user_data: Path) -> pd.DataFrame:
    rows = []
    for bot_dir in sorted((user_data / "paper").iterdir()):
        if not bot_dir.is_dir():
            continue
        dbs = [("aktuell", bot_dir / "tradesv3.dryrun.sqlite")]
        dbs += [(r.stem, r) for r in sorted((bot_dir / "rounds").glob("round-*.sqlite"))]
        for rnd, db in dbs:
            if not db.exists():
                continue
            with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as con:
                t = pd.read_sql_query(
                    "SELECT pair, is_open, is_short, open_date, close_date, open_rate, close_rate, amount, stake_amount, "
                    "fee_open_cost, fee_close_cost, close_profit, close_profit_abs, enter_tag FROM trades", con)
            t["bot"], t["round"] = bot_dir.name, rnd
            rows.append(t)
    df = pd.concat(rows, ignore_index=True)
    for c in ("open_date", "close_date"):
        df[c] = pd.to_datetime(df[c], utc=True, format="mixed")
    df["kind"] = df["enter_tag"].map(trade_kind)
    df["fees"] = df["fee_open_cost"].fillna(0) + df["fee_close_cost"].fillna(0)
    df["net"] = df["close_profit_abs"]
    df["gross"] = df["net"] + df["fees"]
    if (df["is_short"] == 1).any():
        raise SystemExit("Short-Trades gefunden – diese Auswertung rechnet nur Long/Spot.")
    return df


def bootstrap_ci(x: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


class Prices:
    """Letzter bekannter Kurs je Paar zu einem Zeitpunkt (Kerzen-Close, danach Ausführungskurse der Paper-Trades)."""

    def __init__(self, user_data: Path, trades: pd.DataFrame):
        self.series: dict[str, pd.Series] = {}
        self.candle_end: dict[str, pd.Timestamp | None] = {}
        for pair in trades["pair"].unique():
            f = user_data / "data" / "binance" / f"{pair.replace('/', '_')}-15m.parquet"
            s = pd.Series(dtype=float)
            if f.exists():
                c = pd.read_parquet(f, columns=["date", "close"])
                # Close einer 15m-Kerze gilt am Ende der Kerze
                s = pd.Series(c["close"].values, index=pd.to_datetime(c["date"], utc=True) + pd.Timedelta("15min"))
            self.candle_end[pair] = s.index.max() if len(s) else None
            t = trades[trades["pair"] == pair]
            fills = pd.concat([pd.Series(t["open_rate"].values, index=t["open_date"]),
                               pd.Series(t["close_rate"].values, index=t["close_date"]).dropna()])
            if len(s):
                fills = fills[fills.index > s.index.max()]
            self.series[pair] = pd.concat([s, fills]).sort_index()

    def at(self, pair: str, when: pd.Timestamp) -> float:
        s = self.series.get(pair)
        if s is None or s.empty:
            return np.nan
        s = s[s.index <= when]
        return float(s.iloc[-1]) if len(s) else np.nan


def baseline_pnl(trades: pd.DataFrame, prices: Prices, t0, t1) -> float:
    """Wertänderung der trend_d1-Positionen zwischen t0 und t1 (Gebühr nur für Ein-/Ausstiege im Zeitraum)."""
    total = 0.0
    for _, r in trades[trades["bot"] == BASELINE].iterrows():
        a, b = max(r["open_date"], t0), min(r["close_date"] if pd.notna(r["close_date"]) else t1, t1)
        if a >= b:
            continue
        pa = r["open_rate"] if a == r["open_date"] else prices.at(r["pair"], a)
        pb = r["close_rate"] if pd.notna(r["close_date"]) and b == r["close_date"] else prices.at(r["pair"], b)
        total += r["amount"] * (pb - pa)
        total -= (r["fee_open_cost"] or 0) if a == r["open_date"] else 0
        total -= (r["fee_close_cost"] or 0) if pd.notna(r["close_date"]) and b == r["close_date"] else 0
    return total


def evaluate(df: pd.DataFrame, prices: Prices, cost_per_side: float, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    closed = df[df["is_open"] == 0]
    stats, comp = [], []
    for bot in sorted(df["bot"].unique()):
        b = closed[closed["bot"] == bot]
        for kind in KINDS:
            d = b if kind == "Alle" else b[b["kind"] == kind]
            x = d["close_profit"].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(x, n_boot, seed)
            net, fees = float(d["net"].sum()), float(d["fees"].sum())
            stats.append({
                "bot": bot, "typ": kind, "trades": len(d),
                "brutto_usdt": round(float(d["gross"].sum()), 2), "gebuehren_usdt": round(fees, 2),
                "netto_usdt": round(net, 2),
                "netto_je_trade_usdt": round(net / len(d), 3) if len(d) else np.nan,
                "netto_je_trade_pct": round(x.mean() * 100, 4) if len(d) else np.nan,
                "ci95_low_pct": round(lo * 100, 4), "ci95_high_pct": round(hi * 100, 4),
                "zufall_nicht_unterscheidbar": bool(lo <= 0 <= hi) if len(x) >= 2 else None,
                # Anteil der Gebühren am Nettoverlust; > 100 % heißt: vor Gebühren im Plus
                "gebuehrenanteil_verlust_pct": round(fees / -net * 100, 1) if net < 0 else np.nan,
            })
        if b.empty:
            continue
        t0, t1 = b["open_date"].min(), b["close_date"].max()
        p0, p1 = prices.at("BTC/USDT", t0), prices.at("BTC/USDT", t1)
        btc = START_CAPITAL * (p1 / p0 * (1 - cost_per_side) ** 2 - 1)
        base = baseline_pnl(df, prices, t0, t1)
        net = float(b["net"].sum())
        comp.append({
            "bot": bot, "von": t0.strftime("%Y-%m-%d %H:%M"), "bis": t1.strftime("%Y-%m-%d %H:%M"),
            "trades": len(b), "offen": int(((df["bot"] == bot) & (df["is_open"] == 1)).sum()),
            "netto_usdt": round(net, 2), "rendite_pct": round(net / START_CAPITAL * 100, 2),
            "trend_d1_usdt": round(base, 2), "btc_halten_usdt": round(btc, 2),
            "vs_trend_d1_usdt": round(net - base, 2), "vs_btc_usdt": round(net - btc, 2),
        })
    return pd.DataFrame(stats), pd.DataFrame(comp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--user-data", type=Path, default=ROOT / "user_data")
    ap.add_argument("--project-config", type=Path, default=ROOT / "config" / "project.json")
    ap.add_argument("--out", type=Path, default=ROOT / "analysis" / "output")
    ap.add_argument("--bootstrap", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    costs = json.loads(args.project_config.read_text(encoding="utf-8"))["costs"]
    cost_per_side = costs["fee_per_side"] + costs["slippage_per_side"] + costs["half_spread_per_side"]
    df = load_trades(args.user_data)
    prices = Prices(args.user_data, df)
    stats, comp = evaluate(df, prices, cost_per_side, args.bootstrap, args.seed)

    closed = df[df["is_open"] == 0]
    check = (closed["gross"] - closed["amount"] * (closed["close_rate"] - closed["open_rate"])).abs().max()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"Geschlossene Trades: {len(closed)} · offen: {int((df['is_open'] == 1).sum())} · "
          f"Netto gesamt: {closed['net'].sum():.2f} USDT · Gebühren gesamt: {closed['fees'].sum():.2f} USDT")
    print(f"Prüfung Brutto = Menge × Kursdifferenz: größte Abweichung {check:.4f} USDT")
    for pair, end in prices.candle_end.items():
        if pair in ("BTC/USDT", "ETH/USDT"):
            print(f"Kerzen {pair} bis {end}; danach Kurse aus Paper-Ausführungen")
    print("\nJe Bot und Trade-Typ (nur geschlossene Trades):")
    print(stats[stats["trades"] > 0].to_string(index=False))
    print(f"\nVergleich je Bot über seinen eigenen Zeitraum (BTC halten mit {cost_per_side:.2%} Kosten je Seite):")
    print(comp.to_string(index=False))
    no_trades = sorted(set(df["bot"]) - set(closed["bot"]))
    print(f"\nOhne geschlossene Trades: {', '.join(no_trades) or '-'}")

    args.out.mkdir(parents=True, exist_ok=True)
    stats.to_csv(args.out / "fleet_eval.csv", index=False)
    comp.to_csv(args.out / "fleet_vs_baseline.csv", index=False)
    print(f"\nCSV: {args.out / 'fleet_eval.csv'}, {args.out / 'fleet_vs_baseline.csv'}")


if __name__ == "__main__":
    main()
