"""KI 1 lernt aus ihren eigenen Trades (Burak, 24.10.2026, Daueregel 1: jedes Ergebnis fließt sofort zurück).

Zwei Rückkopplungen aus dem gemeinsamen Gedächtnis (``shared.sqlite``, Ergebnis NACH Gebühren). Gelernt wird aus ALLEN
Trades – den eigenen, denen der anderen KIs und denen der Regel-Bots (Burak 24.10.: „Daten aus allen Richtungen“, die
Bots liefern, was gut geht und was nicht); eigene Trades zählen dreifach.

1. **Gewinnchance je Einstieg:** ein kleines Modell (logistische Regression) lernt aus den eigenen abgeschlossenen
   Trades, bei welcher Marktlage ein Einstieg Gewinn brachte. Es wird neu angepasst, sobald ein neuer Trade im
   Gedächtnis steht. Es filtert Signal-Trades erst, wenn es auf den jüngsten, ungesehenen Trades nachweislich besser
   war als „alle nehmen“ – sonst lässt es alles durch. Lern-Trades bleiben immer ungefiltert (Negativdaten, Regel 3).
2. **Haltedauer:** jede Haltedauer-Option wird nach dem Ø-Ergebnis der Trades mit dieser (oder ähnlicher) Haltedauer bewertet
   (Thompson-Sampling: gute Optionen öfter, unbekannte weiter ausprobieren). Die gewählte Option steht im
   Einstiegsgrund („halten:16“) und damit im Gedächtnis.
"""

import re
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MIN_TRADES = 40  # vorher kein Filter: zu wenig eigene Ergebnisse
HOLDOUT = 0.3  # jüngste 30 % der Trades prüfen das Modell, das nur die älteren gesehen hat
HALF_LIFE = 200  # jüngere Trades zählen mehr (Halbwertszeit in Trades)
HOLD_RE = re.compile(r"halten:(\d+)")
STOP_RE = re.compile(r"stop:(\d+)")
STOP_OPTIONS = [2, 4, 6]  # Ausstieg bei −2/−4/−6 %; −6 % ist zugleich die harte Grenze der Strategie (nie lockerer)

_cache: dict[tuple, Any] = {}


OWN_WEIGHT = 3.0


def own_trades(ki: str, user_data: Path | None = None) -> pd.DataFrame:
    """Alle Trades im Gedächtnis (Spalte ``own`` = Trade dieser KI), nach Schlusszeit sortiert."""
    from kitb import shared_memory

    df = shared_memory.load(user_data)
    if df.empty or "ki" not in df:
        return pd.DataFrame()
    df = df.assign(own=df["ki"] == ki)
    return df.sort_values("close_date").reset_index(drop=True)


def _fit(x: pd.DataFrame, y: np.ndarray, own: np.ndarray | None = None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    w = 0.5 ** ((len(y) - 1 - np.arange(len(y))) / HALF_LIFE)
    if own is not None:
        w = w * np.where(own, OWN_WEIGHT, 1.0)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=500))
    m.fit(x, y, logisticregression__sample_weight=w)
    return m


def _design(df: pd.DataFrame, columns: list[str], fill: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    x = df.reindex(columns=columns).astype(float)
    fill = x.median() if fill is None else fill
    return x.fillna(fill).fillna(0.0), fill


def train(trades: pd.DataFrame) -> dict[str, Any]:
    """Modell + Nachweis: filtert nur, wenn es auf ungesehenen jüngsten Trades das Ø-Ergebnis verbessert hätte."""
    from kitb.shared_memory import SNAPSHOT_FEATURES

    t = trades.dropna(subset=["profit_pct"])
    cols = [c for c in SNAPSHOT_FEATURES if c in t and t[c].notna().mean() > 0.5]
    t = t[t[cols].notna().any(axis=1)] if cols else t
    own = (t["own"] if "own" in t else pd.Series(True, index=t.index)).to_numpy(bool)
    out: dict[str, Any] = {"trades": len(t), "active": False, "model": None}
    if len(t) < MIN_TRADES or not cols:
        out["why"] = f"erst {len(t)} Trades mit Marktlage im Gedächtnis (ab {MIN_TRADES} lernt der Filter)"
        return out
    y = (t["profit_pct"] > 0).to_numpy(int)
    if y.min() == y.max():
        out["why"] = "nur Gewinner oder nur Verlierer – nichts zu unterscheiden"
        return out
    cut = int(len(t) * (1 - HOLDOUT))
    old, new = t.iloc[:cut], t.iloc[cut:]
    if np.unique(y[:cut]).size == 2:
        x_old, fill = _design(old, cols)
        m = _fit(x_old, y[:cut], own[:cut])
        p = m.predict_proba(_design(new, cols, fill)[0])[:, 1]
        taken = new["profit_pct"].to_numpy()[p >= 0.5]
        base, filt = float(new["profit_pct"].mean()), (float(taken.mean()) if len(taken) >= 5 else None)
        out.update(check_all=base, check_filtered=filt, check_taken=int(len(taken)), check_n=len(new))
        out["active"] = filt is not None and filt > base
    x, fill = _design(t, cols)
    out.update(model=_fit(x, y, own), columns=cols, fill=fill, own=int(own.sum()))
    out["why"] = ("Filter aktiv: auf den jüngsten Trades besser als alle nehmen" if out["active"]
                  else "Filter aus: noch kein Vorteil auf ungesehenen Trades nachgewiesen")
    return out


def state(ki: str, user_data: Path | None = None) -> dict[str, Any]:
    """Aktuelles Modell; neu angepasst, sobald ein neuer eigener Trade im Gedächtnis steht."""
    t = own_trades(ki, user_data)
    key = (ki, str(user_data), len(t), None if t.empty else str(t["close_date"].iloc[-1]))
    if key not in _cache:
        _cache.clear()
        _cache[key] = {**train(t), "_trades": t}
    return _cache[key]


def win_chance(ki: str, features: dict[str, float], user_data: Path | None = None) -> tuple[bool, float | None, str]:
    """(erlaubt?, Gewinnchance, Begründung). Ohne nachgewiesenen Vorteil: immer erlaubt."""
    s = state(ki, user_data)
    if s.get("model") is None:
        return True, None, s["why"]
    x, _ = _design(pd.DataFrame([features]), s["columns"], s["fill"])
    p = float(s["model"].predict_proba(x)[0, 1])
    return (p >= 0.5 or not s["active"]), p, s["why"]


def _hold_option(row, options: list[int], minutes: int) -> int | None:
    """Gewählte Option aus dem Einstiegsgrund, sonst die tatsächliche Haltedauer auf die nächste Option gerundet."""
    tagged = hold_from_tag(row.get("reason"))
    if tagged in options:
        return tagged
    h = row.get("hold_hours")
    if h is None or not np.isfinite(h) or h <= 0:
        return None
    candles = h * 60 / minutes
    return min(options, key=lambda o: abs(np.log(o) - np.log(max(candles, 0.25))))


def hold_stats(trades: pd.DataFrame, options: list[int], minutes: int = 15) -> pd.DataFrame:
    t = trades.dropna(subset=["profit_pct"])
    opt = pd.Series([_hold_option(r, options, minutes) for r in t.to_dict("records")], index=t.index, dtype="float")
    w = np.where(t["own"], OWN_WEIGHT, 1.0) if "own" in t else np.ones(len(t))
    rows = []
    for o in options:
        m = (opt == o).to_numpy()
        r, ww = t["profit_pct"].to_numpy()[m], w[m]
        n = float(ww.sum())
        mean = float((r * ww).sum() / n) if n else 0.0
        std = float(np.sqrt((ww * (r - mean) ** 2).sum() / n)) if len(r) > 1 else 0.02
        rows.append({"halten": o, "n": n, "mean": mean, "std": std})
    return pd.DataFrame(rows)


def choose_hold(trades: pd.DataFrame, options: list[int], seed: int, minutes: int = 15) -> int:
    """Thompson-Sampling über das Ø-Ergebnis je Haltedauer (Vorwissen: Ø 0, Streuung 2 %)."""
    rng = np.random.default_rng(seed)
    st = hold_stats(trades, options, minutes) if not trades.empty and "profit_pct" in trades else \
        pd.DataFrame({"halten": options, "n": 0.0, "mean": 0.0, "std": 0.02})
    sd = st["std"].clip(lower=0.005) / np.sqrt(st["n"] + 1)
    draw = rng.normal(st["mean"] * st["n"] / (st["n"] + 1), sd)
    return int(st["halten"].iloc[int(np.argmax(draw))])


def hold_for(ki: str, options: list[int], pair: str, when: Any, minutes: int = 15,
             user_data: Path | None = None) -> int:
    """Haltedauer (in Kerzen) für einen neuen Einstieg; gleiches Ergebnis für dieselbe Kerze und denselben Coin."""
    t = state(ki, user_data)["_trades"]
    seed = zlib.crc32(f"{pair}|{pd.Timestamp(when).isoformat()}|{len(t)}".encode())
    return choose_hold(t, options, seed, minutes)


def hold_from_tag(tag: str | None) -> int | None:
    m = HOLD_RE.search(tag or "")
    return int(m.group(1)) if m else None


def stop_from_tag(tag: str | None) -> int | None:
    m = STOP_RE.search(tag or "")
    return int(m.group(1)) if m else None


def choose_stop(trades: pd.DataFrame, seed: int, options: list[int] | None = None) -> int:
    """Thompson-Sampling über das Ø-Ergebnis je Stoploss-Stufe. Nur Trades mit gewählter Stufe zählen (bei anderen ist
    unbekannt, wie sie mit einer anderen Stufe ausgegangen wären)."""
    options = options or STOP_OPTIONS
    rng = np.random.default_rng(seed)
    rows = []
    t = trades.dropna(subset=["profit_pct"]) if not trades.empty and "profit_pct" in trades else pd.DataFrame()
    for o in options:
        part = t[t["reason"].map(stop_from_tag) == o] if not t.empty and "reason" in t else t
        w = np.where(part["own"], OWN_WEIGHT, 1.0) if "own" in part else np.ones(len(part))
        r = part["profit_pct"].to_numpy() if len(part) else np.array([])
        n = float(w.sum())
        mean = float((r * w).sum() / n) if n else 0.0
        std = float(np.sqrt((w * (r - mean) ** 2).sum() / n)) if len(r) > 1 else 0.02
        rows.append((o, n, mean, std))
    st = pd.DataFrame(rows, columns=["stop", "n", "mean", "std"])
    sd = st["std"].clip(lower=0.005) / np.sqrt(st["n"] + 1)
    draw = rng.normal(st["mean"] * st["n"] / (st["n"] + 1), sd)
    return int(st["stop"].iloc[int(np.argmax(draw))])


def stop_for(ki: str, pair: str, when: Any, user_data: Path | None = None) -> int:
    t = state(ki, user_data)["_trades"]
    return choose_stop(t, zlib.crc32(f"stop|{pair}|{pd.Timestamp(when).isoformat()}|{len(t)}".encode()))
