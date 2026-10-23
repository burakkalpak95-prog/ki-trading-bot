"""KI 2 – Belohnungslerner (Contextual Bandit mit Thompson Sampling, Burak 23.10.2026).

Statt Kursmuster vorherzusagen, lernt KI 2 direkt aus Belohnung und Strafe: Für jede Marktlage wählt sie eine
Aktion (nicht handeln oder kaufen und 1 h / 4 h / 12 h halten) und bekommt als Belohnung den tatsächlichen Gewinn
nach Gebühren in Prozent. Je Aktion schätzt eine Bayes'sche lineare Regression den erwarteten Gewinn samt
Unsicherheit. Thompson Sampling zieht aus dieser Unsicherheit – unsichere Aktionen werden so von selbst ausprobiert
(Lern-Trades), sichere gute Aktionen genutzt (Signal-Trades). Gelernt wird aus dem gesamten gemeinsamen Gedächtnis,
also auch aus den Trades und Verlusten von KI 1. Klein und schnell (CPU, Millisekunden), keine Grafikkarte nötig.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from kitb.shared_memory import SNAPSHOT_FEATURES

ARMS = {"1h": 1.0, "4h": 4.0, "12h": 12.0}
PRIOR_PRECISION = 1.0  # Regularisierung: ohne Daten erwartet die KI 0 % (weder Gewinn noch Verlust)
MIN_NOISE = 0.25       # Untergrenze der Streuung (in %-Punkten²), verhindert Übermut bei wenigen Trades


@dataclass
class RewardModel:
    center: np.ndarray = field(default_factory=lambda: np.zeros(len(SNAPSHOT_FEATURES)))
    scale: np.ndarray = field(default_factory=lambda: np.ones(len(SNAPSHOT_FEATURES)))
    arms: dict[str, dict[str, Any]] = field(default_factory=dict)
    n_trades: int = 0

    def design(self, feats: dict[str, float] | pd.DataFrame) -> np.ndarray:
        x = (pd.DataFrame([feats]) if isinstance(feats, dict) else feats).reindex(columns=SNAPSHOT_FEATURES)
        z = ((x.astype(float).to_numpy() - self.center) / self.scale)
        z = np.clip(np.nan_to_num(z, nan=0.0), -5, 5)
        return np.column_stack([np.ones(len(z)), z])

    def posterior(self, arm: str) -> tuple[np.ndarray, np.ndarray, float]:
        d = len(SNAPSHOT_FEATURES) + 1
        a = self.arms.get(arm)
        if not a:
            return np.zeros(d), np.eye(d) / PRIOR_PRECISION, 1.0
        return a["mean"], a["cov"], a["noise"]

    def expected(self, feats: dict[str, float]) -> dict[str, float]:
        x = self.design(feats)[0]
        return {arm: float(x @ self.posterior(arm)[0]) for arm in ARMS}


def arm_for_hold(hours: float) -> str:
    return min(ARMS, key=lambda a: abs(ARMS[a] - hours))


def fit(memory: pd.DataFrame) -> RewardModel:
    """Neu lernen aus allen abgeschlossenen Trades beider KIs (Belohnung = Gewinn nach Gebühren in %)."""
    m = RewardModel()
    if memory is None or memory.empty:
        return m
    mem = memory.dropna(subset=["profit_pct"]).copy()
    mem = mem[mem[SNAPSHOT_FEATURES].notna().any(axis=1)]
    m.n_trades = len(mem)
    if mem.empty:
        return m
    f = mem[SNAPSHOT_FEATURES].astype(float)
    m.center = f.median().fillna(0.0).to_numpy()
    iqr = (f.quantile(0.75) - f.quantile(0.25)).replace(0, np.nan).fillna(1.0).to_numpy()
    m.scale = np.where(np.isfinite(iqr) & (iqr > 0), iqr, 1.0)
    mem["arm"] = [a if a in ARMS else arm_for_hold(h) for a, h in zip(mem.get("action", ""), mem["hold_hours"])]
    for arm in ARMS:
        part = mem[mem["arm"] == arm]
        if part.empty:
            continue
        x = m.design(part[SNAPSHOT_FEATURES])
        y = part["profit_pct"].astype(float).to_numpy() * 100
        prec = PRIOR_PRECISION * np.eye(x.shape[1]) + x.T @ x
        cov_unscaled = np.linalg.inv(prec)
        mean = cov_unscaled @ x.T @ y
        resid = y - x @ mean
        noise = max(MIN_NOISE, float(resid @ resid) / max(1, len(y) - 1))
        m.arms[arm] = {"mean": mean, "cov": cov_unscaled * noise, "noise": noise, "n": len(y)}
    return m


def choose(model: RewardModel, feats: dict[str, float], rng: np.random.Generator,
           epsilon: float = 0.05) -> dict[str, Any]:
    """Aktion wählen. Ergebnis: arm (None = nicht handeln), kind signal/learn, erwarteter Gewinn in %, Begründung."""
    x = model.design(feats)[0]
    means = {arm: float(x @ model.posterior(arm)[0]) for arm in ARMS}
    greedy = max(means, key=means.get)
    greedy_arm = greedy if means[greedy] > 0 else None
    if rng.random() < epsilon:
        arm = str(rng.choice(list(ARMS)))
        why = "Zufallsversuch"
    else:
        samples = {}
        for a in ARMS:
            mean, cov, _ = model.posterior(a)
            samples[a] = float(x @ rng.multivariate_normal(mean, cov))
        best = max(samples, key=samples.get)
        arm = best if samples[best] > 0 else None
        why = "Thompson-Stichprobe"
    if arm is None:
        return {"arm": None, "kind": None, "expected_pct": means[greedy], "reason": ""}
    kind = "signal" if arm == greedy_arm else "learn"
    label = "signal: Belohnung" if kind == "signal" else f"lern: {why}"
    reason = f"{label}, halten {arm}, erwartet {means[arm]:+.2f} %"
    return {"arm": arm, "kind": kind, "expected_pct": means[arm], "reason": reason}
