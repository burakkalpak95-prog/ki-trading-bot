"""Kostenmodell für Backtests.

Freqtrade-Backtests füllen Orders ohne Slippage zum angefragten Preis
(siehe Freqtrade-Doku "Assumptions made by backtesting"). Deshalb werden
Gebühr, Slippage und halber Spread je Order-Seite zu einem effektiven
Gebührensatz addiert und per ``--fee`` übergeben. Freqtrade wendet ihn bei
Einstieg und Ausstieg an. Das ist eine bewusst konservative Näherung, kein
Orderbuch-Modell.
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CostModel:
    fee_per_side: float
    slippage_per_side: float
    half_spread_per_side: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not 0 <= value < 0.05:
                raise ValueError(f"{name}={value} liegt außerhalb [0, 0.05).")

    @property
    def effective_fee_per_side(self) -> float:
        return round(self.fee_per_side + self.slippage_per_side + self.half_spread_per_side, 10)

    @property
    def round_trip_cost(self) -> float:
        return 2 * self.effective_fee_per_side

    @classmethod
    def from_project_config(cls, project_config: dict[str, Any]) -> "CostModel":
        return cls(**project_config["costs"])
