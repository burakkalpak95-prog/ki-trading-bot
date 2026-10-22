"""Agenten-Rahmen: feste Rollen, azyklische Abhängigkeiten, jeder Agent läuft genau einmal je Analyse."""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import talib.abstract as ta

AGENT_COLUMNS = ["agent_market_ok", "agent_trend_score", "agent_vol_pct", "agent_risk_veto", "agent_allow_entry"]


class CycleError(ValueError):
    pass


@dataclass(frozen=True)
class Agent:
    """Ein Agent liest den DataFrame (inkl. Ausgaben seiner Abhängigkeiten) und liefert neue Spalten."""
    name: str
    role: str
    depends_on: tuple[str, ...]
    outputs: tuple[str, ...]
    analyze: Callable[[pd.DataFrame], dict[str, pd.Series]] = field(compare=False)


class Coordinator:
    def __init__(self, agents: list[Agent]):
        self.agents = {a.name: a for a in agents}
        if len(self.agents) != len(agents):
            raise ValueError("Agentennamen müssen eindeutig sein.")
        self.order = self._topological_order()
        self.calls: dict[str, int] = {}

    def _topological_order(self) -> list[str]:
        order, state = [], {}

        def visit(name: str, path: tuple[str, ...]) -> None:
            if state.get(name) == "done":
                return
            if state.get(name) == "visiting":
                raise CycleError("Zyklische Agenten-Abhängigkeit: " + " -> ".join(path + (name,)))
            if name not in self.agents:
                raise ValueError(f"Unbekannte Abhängigkeit: {name}")
            state[name] = "visiting"
            for dep in self.agents[name].depends_on:
                visit(dep, path + (name,))
            state[name] = "done"
            order.append(name)

        for name in self.agents:
            visit(name, ())
        return order

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Führt alle Agenten in Abhängigkeitsreihenfolge genau einmal aus (keine Schleifen, keine Rückfragen)."""
        out = df.copy()
        self.calls = {}
        for name in self.order:
            agent = self.agents[name]
            result = agent.analyze(out)
            if set(result) != set(agent.outputs):
                raise ValueError(f"Agent {name} lieferte {sorted(result)} statt {sorted(agent.outputs)}.")
            for col, series in result.items():
                out[col] = series
            self.calls[name] = self.calls.get(name, 0) + 1
        return out


def _market(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Market Analyst: Datenlage plausibel? (Volumen vorhanden, keine extreme Kerzenspanne)."""
    rng = (df["high"] - df["low"]) / df["close"]
    typical = rng.rolling(500, min_periods=50).median()
    return {"agent_market_ok": (df["volume"] > 0) & ~(rng > 8 * typical)}


def _trend(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Trend Analyst: +1 Aufwärts, -1 Abwärts, 0 unklar (EMA50/EMA200 und deren Steigung)."""
    ema50, ema200 = ta.EMA(df, timeperiod=50), ta.EMA(df, timeperiod=200)
    slope = ema200 - ema200.shift(24)
    score = np.where((ema50 > ema200) & (slope > 0), 1, np.where((ema50 < ema200) & (slope < 0), -1, 0))
    return {"agent_trend_score": pd.Series(score, index=df.index)}


def _volatility(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Volatility Analyst: Perzentil der aktuellen NATR(14) innerhalb der letzten 1000 Kerzen."""
    natr = ta.NATR(df, timeperiod=14)
    return {"agent_vol_pct": natr.rolling(1000, min_periods=200).rank(pct=True)}


def _risk(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Risk Analyst: Veto bei Abwärtstrend mit sehr hoher Volatilität."""
    return {"agent_risk_veto": (df["agent_trend_score"] < 0) & (df["agent_vol_pct"] > 0.9)}


def _coordinate(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Coordinator: Einstieg nur erlaubt, wenn Markt ok, kein Abwärtstrend und kein Risiko-Veto."""
    allow = df["agent_market_ok"].astype(bool) & (df["agent_trend_score"] >= 0) & ~df["agent_risk_veto"].astype(bool)
    return {"agent_allow_entry": allow}


def default_agents() -> list[Agent]:
    return [
        Agent("market", "Market Analyst", (), ("agent_market_ok",), _market),
        Agent("trend", "Trend Analyst", (), ("agent_trend_score",), _trend),
        Agent("volatility", "Volatility Analyst", (), ("agent_vol_pct",), _volatility),
        Agent("risk", "Risk Analyst", ("trend", "volatility"), ("agent_risk_veto",), _risk),
        Agent("coordinator", "Coordinator", ("market", "trend", "volatility", "risk"),
              ("agent_allow_entry",), _coordinate),
    ]
