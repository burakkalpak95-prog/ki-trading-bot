"""Überschneidungs-Check der Paper-Bots (Burak 26.10.2026, Aufgabe 2): Handeln die Bots überhaupt verschieden?

Nur lesend, gleiche Datenquelle wie ``evaluate_fleet.py``.

1. Korrelation der täglichen Netto-PnL-Reihen (UTC-Tage, Summe ``close_profit_abs`` je Ausstiegstag; Tage ohne Trade = 0).
   Zusätzlich stündlich, weil die Flotte erst wenige Tage läuft (4 Tagespunkte geben keine belastbare Korrelation).
2. Gemeinsame Einstiege: Anteil der Einstiege von Bot A, bei denen Bot B im selben Paar und in derselben Richtung
   innerhalb von ±``--window`` Minuten auch eingestiegen ist (Zeile A, Spalte B; nicht symmetrisch). Alle Trades sind
   Long (Spot). Daneben der Anteil, der bei zufälliger Einstiegszeit zu erwarten wäre (gleich viele Einstiege je Paar,
   gleichmäßig über den Zeitraum verteilt): 1 - exp(-n_B,Paar * 2 * Fenster / Zeitraum).
3. Gemeinsame Datenquellen/Indikatoren: aus dem Code abgelesen, siehe ``SOURCES`` (von Hand gepflegt, Stand 26.10.2026).

Aufruf:  python analysis/overlap_check.py [--user-data user_data] [--out analysis/output] [--window 15]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_fleet import ROOT, load_trades  # noqa: E402

GEDAECHTNIS = "gemeinsames Gedächtnis (shared.sqlite: Trades aller Bots + Marktlage-Snapshot)"
# Was jeder Bot zum Entscheiden nutzt (aus user_data/strategies, src/kitb/rule_learning.py, ml_strategy.py,
# reward_learner.py, config/project.json). "Vorschlag" = wer den Einstieg vorschlägt, "Entscheidung" = was ihn freigibt.
SOURCES = {
    "trend_d1": ("SMA200 (1d)", "fest: Regel, kein Lernen"),
    "trend_btc_d1": ("SMA50 (1d) + BTC über SMA200", GEDAECHTNIS),
    "momentum_d1": ("Momentum (1d)", GEDAECHTNIS),
    "ema_cross_1h": ("EMA21/55-Kreuzung (1h)", GEDAECHTNIS),
    "ema_cross_15m": ("EMA21/55-Kreuzung (15m)", GEDAECHTNIS),
    "trend_adx_1h": ("EMA20/50/200 + ADX14 (1h)", GEDAECHTNIS),
    "trend_adx_15m": ("EMA20/50/200 + ADX14 (15m)", GEDAECHTNIS),
    "macd_rsi_1h": ("MACD + RSI14 (1h)", GEDAECHTNIS),
    "macd_rsi_15m": ("MACD + RSI14 (15m)", GEDAECHTNIS),
    "donchian_1h": ("Donchian 20/10 + Volumen-SMA20 (1h)", GEDAECHTNIS),
    "donchian_15m": ("Donchian 20/10 + Volumen-SMA20 (15m)", GEDAECHTNIS),
    "squeeze_1h": ("Bollinger 20/2 Breite + EMA50 (1h)", GEDAECHTNIS),
    "squeeze_15m": ("Bollinger 20/2 Breite + EMA50 (15m)", GEDAECHTNIS),
    "bot_a": ("FreqAI LogReg: RSI/ROC/NATR14, EMA50-Abstand (1h)", "Modell + " + GEDAECHTNIS),
    "bot_b": ("FreqAI LightGBM: RSI/ROC/NATR/ADX/MFI/BB/EMA/MACD (1h+1d, BTC/ETH)", "Modell + " + GEDAECHTNIS),
    "ki_trader": ("FreqAI LightGBM: wie bot_b + Signale aller 6 Regel-Stile + market_flow (15m/1h/4h, BTC/ETH)",
                  "Modell + " + GEDAECHTNIS),
    "ki_reward": ("Bandit auf Marktlage-Snapshot (8 Kursmerkmale + market_flow, 15m)", GEDAECHTNIS + " + Simulationen"),
}


def pnl_corr(closed: pd.DataFrame, freq: str) -> tuple[pd.DataFrame, int]:
    s = closed.assign(t=closed["close_date"].dt.floor(freq)).pivot_table(index="t", columns="bot", values="net",
                                                                          aggfunc="sum")
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq=freq)).fillna(0)
    s = s.loc[:, s.ne(0).sum() >= 2]  # Bots mit nur einem PnL-Punkt haben keine aussagekräftige Korrelation
    return s.corr().round(2), len(s)


def entry_overlap(df: pd.DataFrame, window_min: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    w = pd.Timedelta(minutes=window_min)
    span = df["open_date"].max() - df["open_date"].min()
    bots = sorted(df["bot"].unique())
    share = pd.DataFrame(np.nan, index=bots, columns=bots)
    chance = share.copy()
    times = {(b, p): np.sort(g["open_date"].values) for (b, p), g in df.groupby(["bot", "pair"])}
    for a in bots:
        ea = df[df["bot"] == a]
        for b in bots:
            if a == b:
                continue
            hit, exp_ = 0, 0.0
            for pair, g in ea.groupby("pair"):
                tb = times.get((b, pair))
                if tb is None:
                    continue
                ta = g["open_date"].values
                i = np.searchsorted(tb, ta - w.to_timedelta64())
                j = np.searchsorted(tb, ta + w.to_timedelta64(), side="right")
                hit += int((j > i).sum())
                exp_ += len(ta) * (1 - np.exp(-len(tb) * 2 * w / span))
            share.loc[a, b] = round(hit / len(ea) * 100, 1)
            chance.loc[a, b] = round(exp_ / len(ea) * 100, 1)
    return share, chance


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--user-data", type=Path, default=ROOT / "user_data")
    ap.add_argument("--out", type=Path, default=ROOT / "analysis" / "output")
    ap.add_argument("--window", type=int, default=15, help="Minuten (±) für gemeinsame Einstiege")
    args = ap.parse_args()

    df = load_trades(args.user_data)
    closed = df[df["is_open"] == 0]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    day, n_day = pnl_corr(closed, "1D")
    hour, n_hour = pnl_corr(closed, "1h")
    print(f"Korrelation der täglichen Netto-PnL ({n_day} Tage – zu wenige für eine belastbare Aussage):")
    print(day.to_string())
    print(f"\nKorrelation der stündlichen Netto-PnL ({n_hour} Stunden):")
    print(hour.to_string())

    share, chance = entry_overlap(df, args.window)
    n = df.groupby("bot").size()
    print(f"\nGemeinsame Einstiege in % der Einstiege von Zeile A (gleiches Paar, Long, ±{args.window} Min):")
    print(share.assign(einstiege=n).to_string())
    print("\nZum Vergleich: bei zufälliger Einstiegszeit erwartet (%):")
    print(chance.to_string())

    src = pd.DataFrame([{"bot": b, "vorschlag_indikatoren": v, "entscheidung": e} for b, (v, e) in SOURCES.items()])
    print("\nDatenquellen und Indikatoren (aus dem Code):")
    print(src.to_string(index=False))

    args.out.mkdir(parents=True, exist_ok=True)
    day.to_csv(args.out / "overlap_corr_daily.csv")
    hour.to_csv(args.out / "overlap_corr_hourly.csv")
    share.assign(einstiege=n).to_csv(args.out / "overlap_entries.csv")
    chance.to_csv(args.out / "overlap_entries_chance.csv")
    src.to_csv(args.out / "overlap_sources.csv", index=False)
    print(f"\nCSV in {args.out}")


if __name__ == "__main__":
    main()
