"""Formel-Faktoren („Alphas“) über alle Lern-Coins testen (Idee aus Vibe-Trading, eigene Umsetzung, 23.10.2026).

Vibe-Trading (HKUDS, MIT) bewertet Hunderte veröffentlichte Formel-Faktoren mit dem Informationskoeffizienten (IC):
Rangkorrelation zwischen Faktorwert heute und Rendite danach, je Tag über alle Coins. Der Code dort wurde **nicht**
übernommen; hier stehen eigene Umsetzungen bekannter, öffentlicher Formeln:
- ``a101_*``: Kakushadze, „101 Formulaic Alphas“ (2015, arXiv:1601.00991), Auswahl ohne Branchen-/Fundamentaldaten,
- ``ts_*``: Zeitreihen-Merkmale nach Art von Qlib Alpha158,
- ``x_*``: klassische Querschnitts-Faktoren (Momentum, Kurzfrist-Umkehr, Illiquidität, Schiefe, Beta).

Regel für „lebt“: in **beiden** Zeiträumen (früh/spät, Test-Abschnitt unberührt) gleiches Vorzeichen,
|IC-Mittel| ≥ ``MIN_IC``, |t| ≥ ``MIN_T`` und in ≥ ``MIN_HIT`` der Tage die richtige Richtung.
Erst was hier lebt, kommt als Merkmal in einen A/B-Test der KIs.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kitb import paths
from kitb.config import load_project_config

MIN_IC = 0.02
MIN_T = 2.0
MIN_HIT = 0.53
MIN_COINS = 5  # weniger Coins an einem Tag -> Tag zählt nicht
PERIODS = {"früh": ("2022-03-01", "2024-06-30"), "spät": ("2024-07-01", "2025-11-04")}
HORIZONS = (1, 5)  # Tage


# --- Bausteine (breite Tabellen: Zeilen = Tage, Spalten = Coins) ---

def rank(x: pd.DataFrame) -> pd.DataFrame:
    return x.rank(axis=1, pct=True)


def delta(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x - x.shift(d)


def ts_sum(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d).sum()


def ts_mean(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d).mean()


def ts_std(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d).std()


def ts_min(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d).min()


def ts_max(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d).max()


def ts_rank(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d).rank(pct=True)


def ts_corr(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d).corr(y).replace([np.inf, -np.inf], np.nan)


def ts_cov(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d).cov(y)


def div(a, b):
    return (a / b.where(b.abs() > 1e-12)).replace([np.inf, -np.inf], np.nan)


# --- Faktoren ---

def _ret(p):
    return p["close"].pct_change(fill_method=None)


def _vwap(p):
    return (p["high"] + p["low"] + p["close"]) / 3  # Näherung: Tages-VWAP aus Kerzen


def _a101_009(p):
    d = delta(p["close"], 1)
    return d.where((ts_min(d, 5) > 0) | (ts_max(d, 5) < 0), -d)


def _a101_023(p):
    return (-delta(p["high"], 2)).where(ts_mean(p["high"], 20) < p["high"], 0.0)


def _beta(p, d=60):
    r = _ret(p)
    m = r.mean(axis=1)
    return r.rolling(d, min_periods=d).cov(m).div(m.rolling(d, min_periods=d).var(), axis=0)


FACTORS: dict[str, Callable[[dict[str, pd.DataFrame]], pd.DataFrame]] = {
    "a101_002": lambda p: -ts_corr(rank(delta(np.log(p["volume"]), 2)), rank(div(p["close"] - p["open"], p["open"])), 6),
    "a101_003": lambda p: -ts_corr(rank(p["open"]), rank(p["volume"]), 10),
    "a101_004": lambda p: -ts_rank(rank(p["low"]), 9),
    "a101_006": lambda p: -ts_corr(p["open"], p["volume"], 10),
    "a101_009": _a101_009,
    "a101_012": lambda p: np.sign(delta(p["volume"], 1)) * -delta(p["close"], 1),
    "a101_013": lambda p: -rank(ts_cov(rank(p["close"]), rank(p["volume"]), 5)),
    "a101_014": lambda p: -rank(delta(_ret(p), 3)) * ts_corr(p["open"], p["volume"], 10),
    "a101_018": lambda p: -rank(ts_std((p["close"] - p["open"]).abs(), 5) + (p["close"] - p["open"])
                                + ts_corr(p["close"], p["open"], 10)),
    "a101_023": _a101_023,
    "a101_033": lambda p: rank(-(1 - div(p["open"], p["close"]))),
    "a101_041": lambda p: div(np.sqrt(p["high"] * p["low"]) - _vwap(p), p["close"]),  # durch Kurs geteilt: vergleichbar
    "a101_053": lambda p: -delta(div((p["close"] - p["low"]) - (p["high"] - p["close"]), p["close"] - p["low"]), 9),
    "a101_054": lambda p: div(-(p["low"] - p["close"]) * p["open"] ** 5, (p["low"] - p["high"]) * p["close"] ** 5),
    "a101_101": lambda p: div(p["close"] - p["open"], p["high"] - p["low"] + 1e-3 * p["close"]),
    "ts_kmid": lambda p: div(p["close"] - p["open"], p["open"]),
    "ts_klen": lambda p: div(p["high"] - p["low"], p["open"]),
    "ts_ksft": lambda p: div(2 * p["close"] - p["high"] - p["low"], p["open"]),
    "ts_roc5": lambda p: div(p["close"].shift(5), p["close"]),
    "ts_roc20": lambda p: div(p["close"].shift(20), p["close"]),
    "ts_ma20": lambda p: div(ts_mean(p["close"], 20), p["close"]),
    "ts_std20": lambda p: div(ts_std(p["close"], 20), p["close"]),
    "ts_rsv20": lambda p: div(p["close"] - ts_min(p["low"], 20), ts_max(p["high"], 20) - ts_min(p["low"], 20)),
    "ts_corr20": lambda p: ts_corr(p["close"], np.log1p(p["volume"]), 20),
    "ts_cntp20": lambda p: ts_mean((_ret(p) > 0).astype(float).where(_ret(p).notna()), 20),
    "ts_vma20": lambda p: div(ts_mean(p["volume"], 20), p["volume"]),
    "ts_vstd20": lambda p: div(ts_std(p["volume"], 20), p["volume"]),
    "x_strev5": lambda p: -(p["close"] / p["close"].shift(5) - 1),
    "x_mom30_skip5": lambda p: p["close"].shift(5) / p["close"].shift(30) - 1,
    "x_high60": lambda p: div(p["close"], ts_max(p["high"], 60)),
    "x_illiq20": lambda p: ts_mean(div(_ret(p).abs(), p["close"] * p["volume"]), 20) * 1e9,
    "x_skew20": lambda p: _ret(p).rolling(20, min_periods=20).skew(),
    "x_vol20": lambda p: ts_std(_ret(p), 20),
    "x_beta60": _beta,
}


# Am PC 23.10.2026 (16 Coins, Tageskerzen 2022–11/2025) in beiden Zeiträumen „lebt“; gleichartige zusammengefasst.
# Alle zeigen dieselbe Richtung: ruhige, liquide Coins nahe am 60-Tage-Hoch laufen in den Folgetagen besser als
# schwankende/illiquide. Nur Zusatz-Merkmal für die KIs, kein eigenes Handelssignal.
LIVE = ["ts_klen", "x_illiq20", "x_vol20", "x_beta60", "x_high60", "ts_corr20"]


def daily_ranks(panel: dict[str, pd.DataFrame], names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Je Coin eine Tabelle: Datum (Tagesbeginn) + Rang des Faktors unter allen Coins an diesem Tag (−0,5 … +0,5)."""
    names = names or LIVE
    ranks = {}
    for n in names:
        with np.errstate(all="ignore"):
            ranks[n] = rank(FACTORS[n](panel).replace([np.inf, -np.inf], np.nan)) - 0.5
    out = {}
    for pair in panel["close"].columns:
        df = pd.DataFrame({f"alpha_{n}": ranks[n][pair] for n in names})
        df.index.name = "date"
        out[pair] = df.dropna(how="all").reset_index()
    return out


# --- Daten und Bewertung ---

def load_panel(pairs: list[str], datadir: Path) -> dict[str, pd.DataFrame]:
    """Tageskerzen aller Coins als breite Tabellen; ohne 1d-Datei aus 1h/4h zusammengefasst."""
    from kitb.data.storage import load_ohlcv_raw

    frames = {}
    for pair in pairs:
        df = pd.DataFrame()
        for tf in ("1d", "4h", "1h"):
            df = load_ohlcv_raw(datadir, pair, tf)
            if not df.empty:
                break
        if df.empty:
            continue
        df = (df.set_index("date")[["open", "high", "low", "close", "volume"]]
              .resample("1D", label="left", closed="left")
              .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna())
        frames[pair] = df
    return {col: pd.DataFrame({pair: f[col] for pair, f in frames.items()}).sort_index()
            for col in ("open", "high", "low", "close", "volume")}


def forward_returns(panel: dict[str, pd.DataFrame], h: int) -> pd.DataFrame:
    """Rendite von Tagesschluss t bis t+h; bei h > 1 nur jede h-te Zeile (keine überlappenden Fenster)."""
    close = panel["close"]
    fwd = close.shift(-h) / close - 1
    if h > 1:
        fwd = fwd.iloc[::h]
    return fwd


def ic_series(factor: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Rangkorrelation je Tag zwischen Faktor und späterer Rendite über alle Coins."""
    f, r = factor.reindex(fwd.index), fwd
    mask = f.notna() & r.notna()
    ic = rank(f.where(mask)).corrwith(rank(r.where(mask)), axis=1)
    return ic[mask.sum(axis=1) >= MIN_COINS].dropna()


def _stats(ic: pd.Series) -> dict[str, float]:
    n = len(ic)
    mean = float(ic.mean()) if n else float("nan")
    sd = float(ic.std()) if n > 1 else float("nan")
    t = mean / sd * np.sqrt(n) if n > 1 and sd > 0 else float("nan")
    hit = float((np.sign(ic) == np.sign(mean)).mean()) if n else float("nan")
    return {"Tage": n, "IC": mean, "t": t, "Richtung_stimmt": hit}


def bench(panel: dict[str, pd.DataFrame], periods: dict[str, tuple[str, str]] | None = None,
          horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    periods = periods or PERIODS
    rows = []
    for name, fn in FACTORS.items():
        with np.errstate(all="ignore"):
            fac = fn(panel).replace([np.inf, -np.inf], np.nan)
        for h in horizons:
            ic = ic_series(fac, forward_returns(panel, h))
            for period, (a, b) in periods.items():
                part = ic[(ic.index >= pd.Timestamp(a, tz=ic.index.tz)) & (ic.index <= pd.Timestamp(b, tz=ic.index.tz))]
                rows.append({"Faktor": name, "Horizont_Tage": h, "Zeitraum": period, **_stats(part)})
    return pd.DataFrame(rows)


def select(results: pd.DataFrame) -> pd.DataFrame:
    """Je (Faktor, Horizont): lebt, wenn alle Zeiträume dieselbe Richtung zeigen und jeweils stark genug sind."""
    if results.empty:
        return results
    ok = (results["IC"].abs() >= MIN_IC) & (results["t"].abs() >= MIN_T) & (results["Richtung_stimmt"] >= MIN_HIT)
    g = results.assign(ok=ok, sign=np.sign(results["IC"])).groupby(["Faktor", "Horizont_Tage"])
    out = g.agg(IC_frueh=("IC", "first"), IC_spaet=("IC", "last"), t_min=("t", lambda s: s.abs().min()),
                gleiche_Richtung=("sign", lambda s: s.nunique() == 1), stark=("ok", "all")).reset_index()
    out["lebt"] = out["gleiche_Richtung"] & out["stark"]
    out["Richtung"] = np.where(out["IC_frueh"] > 0, "+", "−")
    return out.sort_values(["lebt", "t_min"], ascending=False)


def render(results: pd.DataFrame, selection: pd.DataFrame, coins: int) -> str:
    return "\n".join([
        "# Formel-Faktoren über alle Lern-Coins (IC-Test)", "",
        f"{coins} Coins, Tageskerzen. Regel: in jedem Zeitraum ({', '.join(f'{k}: {a}–{b}' for k, (a, b) in PERIODS.items())}) "
        f"gleiche Richtung, |IC| ≥ {MIN_IC}, |t| ≥ {MIN_T}, Richtung an ≥ {MIN_HIT:.0%} der Tage. "
        "Test-Abschnitt ab 05.11.2025 nicht verwendet.", "",
        "Unsicher: 16 Coins sind ein kleiner Querschnitt; ein IC um 0,02–0,05 ist für sich allein nicht handelbar, "
        "sondern nur ein mögliches Zusatz-Merkmal für die KIs.", "",
        "## Auswahl", "", selection.to_markdown(index=False, floatfmt=".3f"), "",
        "## Alle Ergebnisse", "", results.to_markdown(index=False, floatfmt=".3f")])


def run(*, data_source: str = "binance", out_dir: Path | None = None) -> dict[str, Any]:
    project = load_project_config()
    pairs = project.get("learning_pairs") or project["allowed_pairs"]
    panel = load_panel(pairs, paths.DATA_DIR / data_source)
    results = bench(panel)
    selection = select(results)
    out_dir = out_dir or (paths.USER_DATA / "reports" / f"alphas-{pd.Timestamp.now(tz='UTC'):%Y%m%d-%H%M}")
    out_dir.mkdir(parents=True, exist_ok=True)
    coins = panel["close"].shape[1] if panel["close"].size else 0
    (out_dir / "alphas.md").write_text(render(results, selection, coins), encoding="utf-8")
    results.to_csv(out_dir / "results.csv", index=False)
    selection.to_csv(out_dir / "selection.csv", index=False)
    return {"report": str(out_dir / "alphas.md"), "coins": coins,
            "lebt": selection[selection["lebt"]][["Faktor", "Horizont_Tage", "Richtung"]].to_dict("records")}
