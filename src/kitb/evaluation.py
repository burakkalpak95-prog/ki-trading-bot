"""Zeitliche Aufteilung, rollierende Fenster und Marktphasen – alles ohne Zukunftsdaten.

- Chronologischer Split Train/Validation/Test: Der Test-Abschnitt bleibt unberührt,
  bis er ausdrücklich angefordert wird (``include_test``) – das wird protokolliert.
- Rollierende Fenster prüfen die zeitliche Stabilität einer Strategie.
- Marktphasen werden aus Tageskerzen bestimmt und gelten erst **nach** Schluss der
  jeweiligen Tageskerze (``valid_from``), damit keine Information aus der Zukunft einfließt.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

DATE_FMT = "%Y%m%d"


@dataclass(frozen=True)
class Segment:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp  # exklusiv

    @property
    def timerange(self) -> str:
        return f"{self.start.strftime(DATE_FMT)}-{self.end.strftime(DATE_FMT)}"


def chronological_split(start: pd.Timestamp, end: pd.Timestamp,
                        fractions: dict[str, float]) -> list[Segment]:
    """Teilt [start, end) in aufeinanderfolgende, nicht überlappende Abschnitte (auf Tage gerundet)."""
    if abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError("Anteile müssen sich zu 1 summieren.")
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    total_days = (end - start).days
    segments, cursor, used = [], start, 0.0
    names = list(fractions)
    for i, name in enumerate(names):
        used += fractions[name]
        seg_end = end if i == len(names) - 1 else start + pd.Timedelta(days=round(total_days * used))
        if seg_end <= cursor:
            raise ValueError(f"Abschnitt '{name}' wäre leer.")
        segments.append(Segment(name, cursor, seg_end))
        cursor = seg_end
    return segments


def rolling_windows(start: pd.Timestamp, end: pd.Timestamp, window_days: int, step_days: int) -> list[Segment]:
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    out, i, cursor = [], 0, start
    while cursor + pd.Timedelta(days=window_days) <= end:
        out.append(Segment(f"window_{i:02d}", cursor, cursor + pd.Timedelta(days=window_days)))
        cursor += pd.Timedelta(days=step_days)
        i += 1
    return out


def classify_regimes(daily: pd.DataFrame, trend_sma_days: int = 50, slope_lookback_days: int = 10,
                     vol_days: int = 30) -> pd.DataFrame:
    """Marktphasen je Tageskerze.

    trend: ``up`` (Close > SMA und SMA steigend), ``down`` (Close < SMA und SMA fallend), sonst ``sideways``.
    volatility: ``high`` wenn annualisierte 30-Tage-Volatilität über dem Median aller **bisherigen** Werte, sonst ``low``.
    ``valid_from`` = Schlusszeit der Tageskerze; ab dann ist die Phase bekannt.
    """
    df = daily[["date", "close"]].copy().reset_index(drop=True)
    sma = df["close"].rolling(trend_sma_days).mean()
    slope = sma - sma.shift(slope_lookback_days)
    trend = np.where((df["close"] > sma) & (slope > 0), "up",
                     np.where((df["close"] < sma) & (slope < 0), "down", "sideways"))
    vol = np.log(df["close"]).diff().rolling(vol_days).std() * np.sqrt(365)
    vol_median = vol.expanding().median()
    volatility = np.where(vol > vol_median, "high", "low")
    df["trend"] = pd.Series(trend).where(sma.notna() & slope.notna())
    df["volatility"] = pd.Series(volatility).where(vol.notna())
    df["valid_from"] = df["date"] + pd.Timedelta(days=1)
    return df.dropna(subset=["trend", "volatility"]).reset_index(drop=True)


def attribute_trades_to_regimes(trades: list[dict], regimes_by_pair: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Ordnet jedem Trade die beim Einstieg **bekannte** Marktphase seines Paares zu."""
    if not trades:
        return pd.DataFrame(columns=["pair", "trend", "volatility", "profit_abs", "profit_ratio"])
    t = pd.DataFrame(trades)[["pair", "open_date", "profit_abs", "profit_ratio"]].copy()
    t["open_date"] = pd.to_datetime(t["open_date"], utc=True)
    parts = []
    for pair, grp in t.groupby("pair"):
        reg = regimes_by_pair.get(pair)
        grp = grp.sort_values("open_date")
        if reg is None or reg.empty:
            grp = grp.assign(trend="unknown", volatility="unknown")
        else:
            grp = pd.merge_asof(grp, reg[["valid_from", "trend", "volatility"]].sort_values("valid_from"),
                                left_on="open_date", right_on="valid_from", direction="backward")
            grp[["trend", "volatility"]] = grp[["trend", "volatility"]].fillna("unknown")
            grp = grp.drop(columns="valid_from")
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)


def regime_summary(attributed: pd.DataFrame, min_trades: int = 30) -> list[dict]:
    rows = []
    for (trend, vol), grp in attributed.groupby(["trend", "volatility"]):
        n = len(grp)
        rows.append({"trend": trend, "volatility": vol, "trades": n,
                     "profit_abs": round(float(grp["profit_abs"].sum()), 2),
                     "avg_profit_ratio": round(float(grp["profit_ratio"].mean()), 5),
                     "winrate": round(float((grp["profit_abs"] > 0).mean()), 3),
                     "low_sample": n < min_trades})
    return rows
