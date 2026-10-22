"""Gemeinsame Basis der KI-Bots (FreqAI).

Beide Bots teilen Ziel (Label), Ein-/Ausstiegsregeln, Stoploss und Kapital-/Kostenbedingungen.
Sie unterscheiden sich nur in Features (``feature_engineering_*``) und Modell (``freqaimodel``).

Label: Steigt der Schlusskurs in den nächsten ``label_period_candles`` Kerzen um mehr als
``label_threshold`` (Standard 0,6 % > Round-Trip-Kosten 0,34 %)? -> Klasse ``up``, sonst ``down``.
Das Label nutzt bewusst Zukunftsdaten (Lernziel); FreqAI trainiert nur auf Daten **vor** dem
jeweiligen Testfenster. Mit ``shuffle=false`` und ``test_size`` > Label-Horizont liegen die letzten
Trainingslabels nicht im Backtest-Zeitraum (siehe tests/test_ml.py).
"""

import json
import logging

import numpy as np
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from kitb.risk import RiskManagedMixin

log = logging.getLogger(__name__)


class KitbFreqaiBase(RiskManagedMixin, IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False
    minimal_roi = {"0": 100.0}
    stoploss = -0.06
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 200
    order_types = {"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False}

    LABEL = "&s-up_or_down"

    def _kitb(self) -> dict:
        return self.freqai_info.get("kitb", {"label_threshold": 0.006, "entry_proba": 0.6, "exit_proba": 0.45})

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour / 23
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        horizon = self.freqai_info["feature_parameters"]["label_period_candles"]
        future_return = dataframe["close"].shift(-horizon) / dataframe["close"] - 1
        self.freqai.class_names = ["down", "up"]
        dataframe[self.LABEL] = np.where(future_return > self._kitb()["label_threshold"], "up", "down")
        dataframe.loc[future_return.isna(), self.LABEL] = np.nan
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.freqai.start(dataframe, metadata, self)

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[(df["do_predict"] == 1) & (df["up"] > self._kitb()["entry_proba"]) & (df["volume"] > 0),
               ["enter_long", "enter_tag"]] = (1, "ml_up")
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[(df["up"] < self._kitb()["exit_proba"]) & (df["volume"] > 0),
               ["exit_long", "exit_tag"]] = (1, "ml_weak")
        return df

    # --- Entscheidungsprotokoll im Paper-Trading (im Backtest übernimmt das kitb.decisions) ---
    def _registry(self):
        from kitb import paths
        from kitb.registry import Registry

        return Registry(paths.REGISTRY_DB)

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force, current_time, entry_tag, side,
                            **kwargs) -> bool:
        allowed = super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force, current_time,
                                              entry_tag, side, **kwargs)
        if allowed and self._is_live_like():
            try:
                df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                last = df.iloc[-1]
                self._registry().insert("decisions", {
                    "source": "paper", "run_ref": self.config.get("db_url"), "bot": self.config.get("bot_name"),
                    "strategy": self.__class__.__name__, "model_identifier": self.freqai_info.get("identifier"),
                    "model_version_id": self.config.get("kitb_model_version_id"), "pair": pair,
                    "decided_at": current_time.isoformat(), "action": "enter_long",
                    "proba_up": float(last.get("up", np.nan)), "do_predict": int(last.get("do_predict", 0)),
                    "explanation": last.get("kitb_explain"),
                    "market": json.dumps({"close": float(last["close"]), "volume": float(last["volume"])}),
                    "features": json.dumps({}), "outcome": None,
                    "risk": json.dumps({"stake": amount * rate, "stoploss": self.stoploss}), "model_error": None})
            except Exception:  # Protokollfehler dürfen den Handel nicht beeinflussen
                log.exception("Entscheidung konnte nicht protokolliert werden")
        return allowed

    def confirm_trade_exit(self, pair, trade, order_type, amount, rate, time_in_force, exit_reason, current_time,
                           **kwargs) -> bool:
        if self._is_live_like():
            try:
                profit = trade.calc_profit_ratio(rate)
                reg = self._registry()
                rows = reg.query("SELECT id, proba_up FROM decisions WHERE source = 'paper' AND pair = ? AND bot = ? "
                                 "AND outcome IS NULL ORDER BY id DESC LIMIT 1", (pair, self.config.get("bot_name")))
                if rows:
                    error = ("false_positive" if profit <= 0 and (rows[0]["proba_up"] or 0) >= self._kitb()["entry_proba"]
                             else None)
                    reg.execute("UPDATE decisions SET outcome = ?, model_error = ? WHERE id = ?",
                                (json.dumps({"profit_ratio": profit, "exit_reason": exit_reason,
                                             "close_date": current_time.isoformat()}), error, rows[0]["id"]))
            except Exception:
                log.exception("Ergebnis konnte nicht protokolliert werden")
        return True
