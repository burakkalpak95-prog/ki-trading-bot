"""Trendfolge mit Marktphasen-Filter, Tageskerzen – Stil der Trendfolger (Dennis, Seykota, Jones).

Vorschlag zum Einstieg: Tagesschluss über dem 50-Tage-Durchschnitt UND Bitcoin über seinem 200-Tage-Durchschnitt (der
Gesamtmarkt ist im Aufwärtstrend). Ausstieg: Tagesschluss unter dem 50-Tage-Durchschnitt oder Bitcoin unter seinem
200-Tage-Durchschnitt. Im Paper-Betrieb als Stil-KI (kitb.rule_learning): die KI entscheidet selbst, ob sie einem
Vorschlag folgt, und wählt den Stoploss.

Test 25.10.2026 (Tageskerzen, 29 Coins, 0,2 % Kosten, ohne Kapital-Modell): 2022–23 Ø +9,7 % je Trade (n 279),
2024–heute Ø +2,6 % (n 759), beide > 2 Standardfehler, Treffer nur 19–28 %. Vorbehalt: nur heute gelistete Coins.
"""

from pandas import DataFrame

import talib.abstract as ta
from freqtrade.strategy import IStrategy, merge_informative_pair
from kitb.indicators import STARTUP_CANDLES_1D
from kitb.risk import RiskManagedMixin
from kitb.rule_learning import StyleAiMixin

MARKET = "BTC/USDT"


class TrendSmaBtc(StyleAiMixin, RiskManagedMixin, IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1d"
    can_short = False
    minimal_roi = {"0": 100.0}
    stoploss = -0.06  # harte Grenze der KIs; die gelernte Stufe (2/4/6 %) ist nie lockerer
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = STARTUP_CANDLES_1D
    order_types = {"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False}

    def informative_pairs(self):
        return [(MARKET, self.timeframe)]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["sma50"] = ta.SMA(dataframe, timeperiod=50)
        if metadata.get("pair") == MARKET:
            btc = dataframe[["date", "close"]].copy()
        else:
            btc = self.dp.get_pair_dataframe(MARKET, self.timeframe)[["date", "close"]].copy()
        btc["sma200"] = ta.SMA(btc["close"], timeperiod=200)
        btc["up"] = (btc["close"] > btc["sma200"]).astype(int)
        dataframe = merge_informative_pair(dataframe, btc[["date", "up"]], self.timeframe, self.timeframe,
                                           ffill=True)
        dataframe["trend"] = ((dataframe["close"] > dataframe["sma50"]) & (dataframe[f"up_{self.timeframe}"] == 1))
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["trend"] & ~dataframe["trend"].shift(1, fill_value=False) & (dataframe["volume"] > 0),
                      ["enter_long", "enter_tag"]] = (1, "trend_sma50_btc")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[~dataframe["trend"] & (dataframe["volume"] > 0), ["exit_long", "exit_tag"]] = (1, "trend_ende")
        return dataframe
