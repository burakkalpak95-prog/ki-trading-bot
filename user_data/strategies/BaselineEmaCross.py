"""Regelbasierte Baseline für Phase 1 – bewusst einfach und NICHT optimiert.

Regeln (nur Long, Spot, kein Hebel):
- Einstieg: EMA(21) kreuzt EMA(55) von unten nach oben (auf abgeschlossenen Kerzen).
- Ausstieg: EMA(21) kreuzt EMA(55) von oben nach unten.
- Notbremse: Stoploss -10 %. Kein ROI-Ausstieg (minimal_roi praktisch deaktiviert).

Zweck: Referenzpunkt für spätere Strategien und KI-Bots, kein Profitabilitätsversprechen.
"""

from pandas import DataFrame

import talib.abstract as ta
from freqtrade.strategy import IStrategy, IntParameter
from kitb.risk import RiskManagedMixin
from technical import qtpylib
from kitb.indicators import STARTUP_CANDLES_1H


class BaselineEmaCross(RiskManagedMixin, IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False
    minimal_roi = {"0": 100.0}
    stoploss = -0.10
    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    startup_candle_count = STARTUP_CANDLES_1H

    # Feste Werte; als Parameter deklariert nur zur Dokumentation (keine Hyperopt in Phase 1).
    ema_fast = IntParameter(5, 50, default=21, space="buy", optimize=False)
    ema_slow = IntParameter(20, 200, default=55, space="buy", optimize=False)

    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            qtpylib.crossed_above(dataframe["ema_fast"], dataframe["ema_slow"])
            & (dataframe["volume"] > 0),
            ["enter_long", "enter_tag"],
        ] = (1, "ema_cross_up")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            qtpylib.crossed_below(dataframe["ema_fast"], dataframe["ema_slow"])
            & (dataframe["volume"] > 0),
            ["exit_long", "exit_tag"],
        ] = (1, "ema_cross_down")
        return dataframe
