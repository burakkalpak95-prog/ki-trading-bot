"""Stil-KIs (Burak 25.10.2026: „lieber mehr KIs, die sich das Gedächtnis teilen, alle auf andere Art vorgehen, aber
wenn sie was dazulernen, es umsetzen“).

Ein Regel-Bot wird damit zur KI: sein Stil (EMA-Kreuzung, Ausbruch, Squeeze …) schlägt Einstiege nur noch vor. Ob
gekauft wird und welcher Stoploss gilt, lernt jede Stil-KI selbst aus dem gemeinsamen Gedächtnis aller Bots (eigene
Trades zählen dreifach, kitb.self_learning):

- **Einstieg:** Gewinnchance bzw. erwarteter Gewinn nach Gebühren. Gefiltert wird erst, wenn das auf den jüngsten,
  ungesehenen Trades nachweislich besser war. Abgelehnte Vorschläge werden mit ``explore_share`` trotzdem als
  Lern-Trade genommen (Negativdaten, Daueregel 3) und so markiert.
- **Stoploss:** 2/4/6 % per Thompson-Sampling; −6 % bleibt die härteste mögliche Grenze (nie lockerer als ``stoploss``).
- **Ausstieg:** der eigene Stil-Ausstieg bleibt (Test 25.10.: feste Ziel/Stop-Ausstiege waren schlechter als Halten,
  die beste Haltedauer unterscheidet sich je Stil).

Ohne ``kitb_style_ai`` in der Konfiguration (Backtests, alte Bots) verhält sich die Strategie unverändert.
Gelernt wird ohne Simulationen – das spart ~1 GB Arbeitsspeicher je Bot (KI 1/2 und Bot A/B nutzen sie).
"""

import logging
import zlib

import pandas as pd

log = logging.getLogger(__name__)


class StyleAiMixin:
    """Vor ``RiskManagedMixin`` in die Basisklassen setzen: ``class X(StyleAiMixin, RiskManagedMixin, IStrategy)``."""

    use_custom_stoploss = True

    def _style_ai(self) -> bool:
        return bool(self.config.get("kitb_style_ai")) and self._is_live_like()

    def advise_entry(self, dataframe, metadata: dict):
        df = super().advise_entry(dataframe, metadata)
        if not self._style_ai() or df.empty or not df["enter_long"].fillna(0).iloc[-1]:
            return df
        pair, i = metadata.get("pair", ""), df.index[-1]
        try:
            from freqtrade.exchange import timeframe_to_minutes

            from kitb import self_learning, shared_memory

            bot = self.config.get("bot_name", "")
            snap = shared_memory.market_snapshot(df, timeframe_to_minutes(self.timeframe), pair).iloc[-1]
            allowed, p, why = self_learning.win_chance(bot, snap.to_dict(), sims=False)
            style = str(df.at[i, "enter_tag"] or "stil")
            if allowed:
                tag = f"signal: {style}"
            elif zlib.crc32(f"{pair}|{df.at[i, 'date']}".encode()) % 1000 < self._explore() * 1000:
                tag = f"lern: {style} (KI hätte ausgelassen)"
            else:
                log.info("Stil-KI %s: Vorschlag %s ausgelassen (%s)", pair, style, why)
                df.loc[i, ["enter_long", "enter_tag"]] = (0, "")
                return df
            st = self_learning.stop_for(bot, pair, df.at[i, "date"], sims=False)
            df.loc[i, "enter_tag"] = f"{tag} · stop:{st}"
            log.info("Stil-KI %s: %s, Chance %s (%s)", pair, tag, "?" if p is None else f"{p:.0%}", why)
        except Exception:  # noqa: BLE001 – Lernfehler: der Stil-Vorschlag gilt wie bisher
            log.exception("Stil-KI: eigene Erfahrung nicht verfügbar")
        return df

    def _explore(self) -> float:
        return float(self.config.get("kitb_practice", {}).get("explore_share", 0.2))

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, after_fill=False, **kwargs):
        from freqtrade.strategy import stoploss_from_open

        from kitb.self_learning import stop_from_tag

        chosen = stop_from_tag(trade.enter_tag)
        if not chosen:
            return None
        return stoploss_from_open(-min(chosen / 100, abs(self.stoploss)), current_profit, is_short=False,
                                  leverage=trade.leverage or 1.0) or None

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force, current_time, entry_tag, side,
                            **kwargs) -> bool:
        allowed = super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force, current_time,
                                              entry_tag, side, **kwargs)
        if allowed and self._style_ai():
            try:  # Marktlage beim Einstieg ins gemeinsame Gedächtnis (davon lernen alle KIs)
                from freqtrade.exchange import timeframe_to_minutes

                from kitb import shared_memory

                df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                snap = shared_memory.market_snapshot(df, timeframe_to_minutes(self.timeframe), pair).iloc[-1]
                tag = str(entry_tag or "")
                shared_memory.snapshot(self.config["bot_name"], pair=pair, when=pd.Timestamp(current_time),
                                       kind="learn" if tag.startswith("lern") else "signal", reason=tag,
                                       features=snap.to_dict(), timeframe=self.timeframe)
            except Exception:  # noqa: BLE001
                log.exception("Gemeinsames Gedächtnis: Einstieg konnte nicht notiert werden")
        return allowed
