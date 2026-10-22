"""Drift-Erkennung: Haben sich Markt, Modellvorhersagen oder Trefferquote gegenüber einer
Referenzperiode verändert?

- Merkmals-Drift: Population Stability Index (PSI) und Kolmogorow-Smirnow-Test je Marktmerkmal.
  Faustregel (verbreitete Konvention, keine Norm): PSI < 0,1 stabil, 0,1–0,25 moderat, > 0,25 deutlich.
- Leistungs-Drift: Trefferquote / Fehlerrate der Entscheidungen jüngst vs. Referenz.
Drift löst nur eine **Meldung** aus (Empfehlung: neuer Trainingszyklus + Prüfung), nie eine Aktion.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PSI_MODERATE, PSI_SIGNIFICANT = 0.1, 0.25


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    expected, actual = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    if len(expected) < bins or len(actual) < bins:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Modellunabhängige Marktmerkmale aus OHLCV (nur Vergangenheit)."""
    out = pd.DataFrame({"date": df["date"]})
    logret = np.log(df["close"]).diff()
    out["return_1"] = logret
    out["return_24"] = np.log(df["close"]).diff(24)
    out["volatility_24"] = logret.rolling(24).std()
    out["range_pct"] = (df["high"] - df["low"]) / df["close"]
    out["log_volume"] = np.log1p(df["volume"])
    return out


def feature_drift(df: pd.DataFrame, reference_days: int, recent_days: int) -> dict[str, Any]:
    feats = market_features(df).dropna()
    end = feats["date"].max()
    recent = feats[feats["date"] > end - pd.Timedelta(days=recent_days)]
    ref = feats[(feats["date"] <= end - pd.Timedelta(days=recent_days))
                & (feats["date"] > end - pd.Timedelta(days=recent_days + reference_days))]
    result = {"reference": [str(ref["date"].min()), str(ref["date"].max())],
              "recent": [str(recent["date"].min()), str(recent["date"].max())], "features": {}}
    for col in [c for c in feats.columns if c != "date"]:
        p = psi(ref[col].to_numpy(), recent[col].to_numpy())
        ks = stats.ks_2samp(ref[col], recent[col]) if len(ref) and len(recent) else None
        level = "unknown" if np.isnan(p) else ("significant" if p > PSI_SIGNIFICANT else
                                               "moderate" if p > PSI_MODERATE else "stable")
        result["features"][col] = {"psi": None if np.isnan(p) else round(p, 4),
                                   "ks_pvalue": None if ks is None else float(ks.pvalue), "level": level}
    levels = [f["level"] for f in result["features"].values()]
    result["drift"] = "significant" if "significant" in levels else "moderate" if "moderate" in levels else "stable"
    return result


def performance_drift(decisions: pd.DataFrame, recent_n: int = 30, tolerance: float = 0.15) -> dict[str, Any]:
    """decisions: Spalten decided_at, win (bool), model_error (str|None). Vergleich letzte N vs. davor."""
    d = decisions.sort_values("decided_at")
    if len(d) < 2 * recent_n:
        return {"status": "insufficient_data", "decisions": len(d), "needed": 2 * recent_n}
    ref, rec = d.iloc[:-recent_n], d.iloc[-recent_n:]
    ref_win, rec_win = ref["win"].mean(), rec["win"].mean()
    ref_err, rec_err = ref["model_error"].notna().mean(), rec["model_error"].notna().mean()
    drift = (ref_win - rec_win) > tolerance or (rec_err - ref_err) > tolerance
    return {"status": "drift" if drift else "stable", "reference_winrate": round(ref_win, 3),
            "recent_winrate": round(rec_win, 3), "reference_error_rate": round(ref_err, 3),
            "recent_error_rate": round(rec_err, 3), "recent_n": recent_n}
