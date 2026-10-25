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
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from kitb.risk import RiskManagedMixin

log = logging.getLogger(__name__)


def explore_draw(dates, pair: str, minutes: int = 60):
    """Pseudozufall 0..999 je Kerze und Markt, gut durchmischt (kein Block über Tage) und reproduzierbar."""
    candle = ((dates - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(minutes=minutes)).astype("int64")
    salt = sum((i + 1) * ord(c) for i, c in enumerate(pair))
    return ((candle * 2_654_435_761 + salt * 40_503) % 4_294_967_296) % 1000


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
        k = self._kitb()
        ok = (df["do_predict"] == 1) & (df["volume"] > 0)
        explore = float(k.get("explore_share", 0.0))
        if explore > 0:
            # Nur Lernmodus im Paper-Trading (Burak, 23.10.2026): die KI geht gezielt auch Trades ohne Signal ein
            # (Lern-Trades), damit sie Negativdaten sammelt. Auswahl deterministisch je Kerze und Markt (nachvollziehbar).
            from freqtrade.exchange import timeframe_to_minutes

            h = explore_draw(df["date"], metadata.get("pair", ""), timeframe_to_minutes(self.timeframe))
            mask = ok & (df["up"] > k.get("explore_floor", 0.45)) & (h < explore * 1000)
            df.loc[mask, "enter_long"] = 1
            df.loc[mask, "enter_tag"] = "lern: ohne Signal, Modell " + (df.loc[mask, "up"] * 100).round().astype(int).astype(str) + " %"
        sig = ok & (df["up"] > k["entry_proba"])
        df.loc[sig, "enter_long"] = 1
        df.loc[sig, "enter_tag"] = "signal: Modell " + (df.loc[sig, "up"] * 100).round().astype(int).astype(str) + " %"
        if k.get("self_learning") and self._is_live_like() and len(df) and df["enter_long"].iloc[-1] == 1:
            # Selbst lernen (Burak 24.10.2026): Haltedauer nach den eigenen Ergebnissen wählen, steht im Einstiegsgrund
            try:
                from kitb import self_learning

                from freqtrade.exchange import timeframe_to_minutes

                n = self_learning.hold_for(self.config.get("bot_name", ""), self._hold_options(),
                                           metadata.get("pair", ""), df["date"].iloc[-1],
                                           timeframe_to_minutes(self.timeframe))
                st = self_learning.stop_for(self.config.get("bot_name", ""), metadata.get("pair", ""),
                                            df["date"].iloc[-1])
                df.loc[df.index[-1], "enter_tag"] = f"{df['enter_tag'].iloc[-1]} · halten:{n} · stop:{st}"
            except Exception:  # noqa: BLE001 – ohne Wahl gilt die feste Haltedauer
                log.exception("Haltedauer konnte nicht gewählt werden")
        return df

    def _hold_options(self) -> list[int]:
        return [int(x) for x in self._kitb().get("hold_options", [4, 8, 16, 32])]

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        if self._kitb().get("self_learning") and self._is_live_like():
            # Selbst lernen: kein Sofort-Ausstieg bei leichter Unsicherheit (Paper 24.10.: 111 von 125 Trades endeten so
            # nach Ø 1,5 h mit kleinem Minus); Ausstieg über die gelernte Haltedauer oder den Stoploss.
            return df
        df.loc[(df["up"] < self._kitb()["exit_proba"]) & (df["volume"] > 0),
               ["exit_long", "exit_tag"]] = (1, "ml_weak")
        return df

    use_custom_stoploss = True

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, after_fill=False, **kwargs):
        """Gelernte Stoploss-Stufe (Burak 24.10.2026). Freqtrade zieht den Stop nur nach oben: die feste Grenze
        ``stoploss`` (−6 %) bleibt die lockerste mögliche Stufe."""
        from freqtrade.strategy import stoploss_from_open

        from kitb.self_learning import stop_from_tag

        chosen = stop_from_tag(trade.enter_tag)
        if not chosen:
            return None
        return stoploss_from_open(-min(chosen / 100, abs(self.stoploss)), current_profit, is_short=False,
                                  leverage=trade.leverage or 1.0) or None

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        """Lernmodus: Übungstrades spätestens nach ``max_hold_candles`` schließen, damit Ergebnisse entstehen."""
        from freqtrade.exchange import timeframe_to_seconds

        from kitb.self_learning import hold_from_tag

        chosen = hold_from_tag(trade.enter_tag)
        limit = chosen or self._kitb().get("max_hold_candles")
        if not limit and self._kitb().get("self_learning"):
            # Ohne gewählte Haltedauer (Trade vor dem Selbstlernen eröffnet) und ohne Lernhorizont (Bot A/B) gab es seit
            # dem Abschalten von ml_weak keinen Ausstieg mehr außer dem Stoploss (25.10.: 13 Trades seit 16 Uhr offen)
            limit = max(self._hold_options())
        if limit and (current_time - trade.open_date_utc).total_seconds() >= limit * timeframe_to_seconds(self.timeframe):
            return f"halten {chosen}" if chosen else "lernhorizont"
        return None

    # --- Entscheidungsprotokoll im Paper-Trading (im Backtest übernimmt das kitb.decisions) ---
    def _registry(self):
        from kitb import paths
        from kitb.registry import Registry

        return Registry(paths.REGISTRY_DB)

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force, current_time, entry_tag, side,
                            **kwargs) -> bool:
        allowed = super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force, current_time,
                                              entry_tag, side, **kwargs)
        if (allowed and self._is_live_like() and self._kitb().get("memory_gate_signals")
                and str(entry_tag or "").startswith("signal")):
            # Rückkopplung: Signal-Trades prüft zusätzlich das KI-Gedächtnis, das aus den eigenen Trade-Ergebnissen lernt.
            # Lern-Trades bleiben ungefiltert, damit weiter Daten entstehen. Fehler -> erlauben.
            try:
                from kitb import ai_memory

                allowed, p = ai_memory.allow_entry(strategy=type(self).__name__, pair=pair, when=current_time)
                log.info("KI-Gedächtnis %s: Gewinnchance %s -> %s", pair, "?" if p is None else f"{p:.0%}",
                         "erlaubt" if allowed else "ausgelassen")
            except Exception as e:  # noqa: BLE001
                log.warning("KI-Gedächtnis nicht verfügbar (%s)", e)
                allowed = True
        if (allowed and self._is_live_like() and self._kitb().get("self_learning")
                and str(entry_tag or "").startswith("signal")):
            # Selbst lernen (Burak 24.10.2026): Signal-Trades prüft das Modell aus den EIGENEN Trade-Ergebnissen (nach
            # Gebühren). Es filtert erst, wenn es auf ungesehenen Trades besser war. Lern-Trades bleiben ungefiltert.
            try:
                from freqtrade.exchange import timeframe_to_minutes

                from kitb import self_learning, shared_memory

                df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                snap = shared_memory.market_snapshot(df, timeframe_to_minutes(self.timeframe), pair).iloc[-1]
                allowed, p, why = self_learning.win_chance(self.config.get("bot_name", ""), snap.to_dict())
                log.info("Eigene Erfahrung %s: Gewinnchance %s -> %s (%s)", pair, "?" if p is None else f"{p:.0%}",
                         "erlaubt" if allowed else "ausgelassen", why)
            except Exception as e:  # noqa: BLE001
                log.warning("Eigene Erfahrung nicht verfügbar (%s)", e)
                allowed = True
        from kitb.shared_memory import KIS

        if allowed and self._is_live_like() and self.config.get("bot_name") in KIS:
            self._shared_snapshot(pair, current_time, entry_tag)
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

    def _shared_snapshot(self, pair, current_time, entry_tag) -> None:
        """Einstieg ins gemeinsame Gedächtnis beider KIs notieren (Fehler beeinflussen den Handel nicht)."""
        try:
            from kitb import shared_memory
            from freqtrade.exchange import timeframe_to_minutes

            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            snap = shared_memory.market_snapshot(df, timeframe_to_minutes(self.timeframe), pair).iloc[-1]
            tag = str(entry_tag or "")
            shared_memory.snapshot(self.config["bot_name"], pair=pair, when=current_time,
                                   kind="learn" if tag.startswith("lern") else "signal", reason=tag,
                                   features=snap.to_dict(), timeframe=self.timeframe)
        except Exception:  # noqa: BLE001
            log.exception("Gemeinsames Gedächtnis: Einstieg konnte nicht notiert werden")

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
