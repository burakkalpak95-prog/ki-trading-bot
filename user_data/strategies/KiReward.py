"""KI 2 – Belohnungslerner (Paper-Lernmodus). Lernt aus Belohnung/Strafe je Trade, 15-Minuten-Kerzen.

Entscheidet je neue Kerze und Markt: nicht handeln oder kaufen und 1 h / 4 h / 12 h halten (siehe kitb.reward_learner).
Lernt neu, sobald das gemeinsame Gedächtnis neue abgeschlossene Trades enthält (beider KIs). Nur Long (Spot ohne
Hebel). Risikolimits gelten unverändert (RiskManagedMixin). Nur für Paper-Trading gedacht: im Backtest fehlt die
Rückkopplung Trade -> Lernen, daher entscheidet die KI dort nicht.
"""

import logging
import re

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from kitb import reward_learner, shared_memory
from kitb.risk import RiskManagedMixin

log = logging.getLogger(__name__)


class KiReward(RiskManagedMixin, IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = False
    minimal_roi = {"0": 100.0}
    stoploss = -0.05
    process_only_new_candles = True
    use_exit_signal = True  # nötig, damit Freqtrade custom_exit (Haltedauer) überhaupt aufruft; exit_long bleibt 0
    startup_candle_count = 900  # 200-Perioden-EMA auf 15m + Puffer
    order_types = {"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False}

    _model = None
    _model_stamp = None

    def _epsilon(self) -> float:
        return float(self.config.get("kitb_practice", {}).get("explore_share", 0.05))

    def _refresh_model(self) -> None:
        """Neu lernen, sobald sich das gemeinsame Gedächtnis geändert hat (Bayes-Update dauert Millisekunden)."""
        p = shared_memory.db_path()
        stamp = p.stat().st_mtime if p.exists() else None
        # + Simulationen (höchstens alle 15 Min. neu übernommen, die Endlos-Simulation schreibt laufend)
        stamp = (stamp, pd.Timestamp.now(tz="UTC").floor("15min"))
        if self._model is None or stamp != self._model_stamp:
            try:
                self._model = reward_learner.fit(reward_learner.with_simulations(shared_memory.load()))
                log.info("KI 2 hat neu gelernt: %s Trades im gemeinsamen Gedächtnis", self._model.n_trades)
            except Exception:  # noqa: BLE001 – Lernfehler darf Handel/Ausstiege nicht blockieren: altes Modell behalten
                log.exception("KI 2: Neu-Lernen fehlgeschlagen, nutze bisheriges Modell")
                if self._model is None:
                    self._model = reward_learner.RewardModel()
            self._model_stamp = stamp

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        snap = shared_memory.market_snapshot(dataframe, 15, metadata.get("pair"))
        for c in snap.columns:
            dataframe[f"snap_{c}"] = snap[c]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        if not self._is_live_like() or dataframe.empty or dataframe["volume"].iloc[-1] <= 0:
            return dataframe
        self._refresh_model()
        feats = {c: float(dataframe[f"snap_{c}"].iloc[-1]) for c in shared_memory.SNAPSHOT_FEATURES
                 if f"snap_{c}" in dataframe}
        d = reward_learner.choose(self._model, feats, np.random.default_rng(), self._epsilon())
        if d["arm"]:
            dataframe.loc[dataframe.index[-1], ["enter_long", "enter_tag"]] = (1, d["reason"])
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        return dataframe

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        m = re.search(r"halten (\d+)h", str(trade.enter_tag or ""))
        hours = float(m.group(1)) if m else 4.0
        if (current_time - trade.open_date_utc).total_seconds() >= hours * 3600:
            return f"haltedauer {int(hours)}h"
        return None

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force, current_time, entry_tag, side,
                            **kwargs) -> bool:
        allowed = super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force, current_time,
                                              entry_tag, side, **kwargs)
        if allowed and self._is_live_like():
            try:
                df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                feats = {c: float(df[f"snap_{c}"].iloc[-1]) for c in shared_memory.SNAPSHOT_FEATURES
                         if f"snap_{c}" in df}
                tag = str(entry_tag or "")
                arm = re.search(r"halten (\d+h)", tag)
                shared_memory.snapshot("ki_reward", pair=pair, when=current_time,
                                       kind="learn" if tag.startswith("lern") else "signal", reason=tag,
                                       features=feats, action=arm.group(1) if arm else "", timeframe=self.timeframe)
            except Exception:  # noqa: BLE001 – Notizfehler beeinflussen den Handel nicht
                log.exception("Gemeinsames Gedächtnis: Einstieg konnte nicht notiert werden")
        return allowed
