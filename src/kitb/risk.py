"""Unabhängiges Risikomodul – hat Vorrang vor allen Handelssignalen.

Die Prüfungen sind reine Funktionen (``RiskManager``) und werden über ``RiskManagedMixin`` in
jede Strategie eingebunden (Freqtrade-Callbacks ``custom_stake_amount`` und ``confirm_trade_entry``).
Damit gelten dieselben Limits im Backtest und im Paper-Trading.

Bei Überschreitung des maximalen Drawdowns löst das Modul im Paper-/Live-Modus den Not-Stopp aus
(dauerhaft, zurücksetzen nur durch einen Menschen). Ausnahme (Freigabe Burak, 22.10.2026): Paper-Bots im
Modus ``learning``/``explore`` beenden nur ihre eigene Runde und starten mit frischem virtuellem Kapital neu. Im Backtest würde ein dauerhafter Stopp alle
späteren Zeitfenster entwerten; dort wird stattdessen der Zeitpunkt ausgewiesen, an dem die
Abschaltung ausgelöst hätte (``risk_halt_would_trigger_at``).
"""

import json
import logging
import os
import signal
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kitb import killswitch, paths
from kitb.config import load_json

log = logging.getLogger(__name__)
RISK_CONFIG = paths.CONFIG_DIR / "risk.json"


@dataclass(frozen=True)
class RiskLimits:
    max_position_fraction: float
    max_risk_per_trade_fraction: float
    max_total_exposure_fraction: float
    max_open_positions: int
    max_daily_loss_fraction: float
    max_drawdown_fraction: float
    max_data_age_candles: float
    max_stake_share_of_24h_volume: float
    leverage_allowed: bool = False
    futures_allowed: bool = False
    margin_allowed: bool = False
    withdrawals_allowed: bool = False

    def __post_init__(self) -> None:
        if self.leverage_allowed or self.futures_allowed or self.margin_allowed or self.withdrawals_allowed:
            raise ValueError("Hebel, Futures, Margin und Auszahlungen sind nicht erlaubt.")
        for name in ("max_position_fraction", "max_risk_per_trade_fraction", "max_total_exposure_fraction",
                     "max_daily_loss_fraction", "max_drawdown_fraction", "max_stake_share_of_24h_volume"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} muss in (0, 1] liegen.")
        if self.max_total_exposure_fraction > 1:
            raise ValueError("Gesamtposition > 100 % wäre Hebel.")

    @classmethod
    def load(cls, path: Path | None = None) -> "RiskLimits":
        return cls(**load_json(path or RISK_CONFIG))


@dataclass
class AccountState:
    starting_balance: float
    realized_equity_curve: list[float]  # Kapital nach jedem geschlossenen Trade (chronologisch)
    today_realized_pnl: float
    open_positions: int
    open_exposure: float
    open_pairs: set[str] = field(default_factory=set)

    @property
    def equity(self) -> float:
        return self.realized_equity_curve[-1] if self.realized_equity_curve else self.starting_balance

    @property
    def drawdown(self) -> float:
        peak = max([self.starting_balance, *self.realized_equity_curve])
        return max(0.0, 1 - self.equity / peak)


@dataclass
class Decision:
    allowed: bool
    reasons: list[str]
    stake: float = 0.0


class RiskManager:
    def __init__(self, limits: RiskLimits):
        self.limits = limits

    def size_stake(self, proposed: float, state: AccountState, stoploss: float) -> float:
        L = self.limits
        caps = [proposed,
                L.max_position_fraction * state.equity,
                L.max_risk_per_trade_fraction * state.equity / max(abs(stoploss), 1e-6),
                max(0.0, L.max_total_exposure_fraction * state.equity - state.open_exposure)]
        return max(0.0, min(caps))

    def check_entry(self, *, pair: str, stake: float, state: AccountState, kill_switch: bool,
                    data_age_candles: float | None = None, quote_volume_24h: float | None = None,
                    min_stake: float | None = None, enforce_drawdown_halt: bool = True) -> Decision:
        L, reasons = self.limits, []
        if kill_switch:
            reasons.append("kill_switch")
        if pair in state.open_pairs:
            reasons.append("duplicate_position")
        if state.open_positions >= L.max_open_positions:
            reasons.append("max_open_positions")
        if -state.today_realized_pnl >= L.max_daily_loss_fraction * state.starting_balance:
            reasons.append("max_daily_loss")
        if enforce_drawdown_halt and state.drawdown >= L.max_drawdown_fraction:
            reasons.append("max_drawdown")
        if data_age_candles is not None and data_age_candles > L.max_data_age_candles:
            reasons.append("stale_data")
        if quote_volume_24h is not None and stake > L.max_stake_share_of_24h_volume * quote_volume_24h:
            reasons.append("insufficient_liquidity")
        if stake <= 0 or (min_stake is not None and stake < min_stake):
            reasons.append("stake_below_minimum")
        return Decision(not reasons, reasons, stake if not reasons else 0.0)


def write_state(bot: str, payload: dict[str, Any], user_data_dir: Path | None = None) -> None:
    p = (user_data_dir or paths.USER_DATA) / "paper" / "risk_state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        current = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except ValueError:
        current = {}
    current[bot] = {**payload, "updated_at": datetime.now(UTC).isoformat()}
    p.write_text(json.dumps(current, indent=2, default=str), encoding="utf-8")


def _refill_enabled() -> bool:
    from kitb.config import load_project_config

    return bool(load_project_config().get("paper_learning", {}).get("refill_on_drawdown"))


class RiskManagedMixin:
    """In Freqtrade-Strategien einbinden: ``class S(RiskManagedMixin, IStrategy)``."""

    _risk: RiskManager | None = None

    def _rm(self) -> RiskManager:
        if self._risk is None:
            self._risk = RiskManager(RiskLimits.load())
        return self._risk

    def _request_new_round(self, state: "AccountState") -> None:
        """Nur Lern-/Sammler-Bots: Runde beenden; ``kitb paper run`` archiviert sie und startet mit frischem Kapital."""
        from kitb import paper

        bot = self.config.get("bot_name", "bot")
        paper.request_new_round(bot, f"Max. Drawdown {state.drawdown:.1%} erreicht")
        log.warning("%s: Verlustlimit erreicht – Runde endet, Neustart mit frischem virtuellem Kapital.", bot)
        os.kill(os.getpid(), signal.SIGTERM)

    def _is_live_like(self) -> bool:
        runmode = getattr(getattr(self, "dp", None), "runmode", None)
        return getattr(runmode, "value", runmode) in ("dry_run", "live")

    def _account_state(self, current_time: datetime) -> AccountState:
        from freqtrade.persistence import Trade

        start = float(self.wallets.get_starting_balance()) if self.wallets else float(self.config["dry_run_wallet"])
        closed = sorted(Trade.get_trades_proxy(is_open=False), key=lambda t: t.close_date_utc)
        curve, eq, today = [], start, 0.0
        for t in closed:
            eq += t.close_profit_abs or 0.0
            curve.append(eq)
            if t.close_date_utc.date() == current_time.date():
                today += t.close_profit_abs or 0.0
        open_trades = Trade.get_trades_proxy(is_open=True)
        return AccountState(start, curve, today, len(open_trades),
                            sum(t.stake_amount for t in open_trades), {t.pair for t in open_trades})

    def _market_checks(self, pair: str, current_time: datetime) -> tuple[float | None, float | None]:
        if not self._is_live_like():
            return None, None  # im Backtest liefert Freqtrade nur abgeschlossene Kerzen
        from freqtrade.exchange import timeframe_to_seconds

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty:
            return float("inf"), None
        tf_s = timeframe_to_seconds(self.timeframe)
        last_close = df["date"].iloc[-1].to_pydatetime().timestamp() + tf_s
        age = (current_time.timestamp() - last_close) / tf_s
        n = max(1, int(86400 / tf_s))
        qv = float((df["volume"].tail(n) * df["close"].tail(n)).sum())
        return age, qv

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake, min_stake, max_stake,
                            leverage, entry_tag, side, **kwargs) -> float:
        if leverage and leverage > 1:
            return 0.0
        stake = self._rm().size_stake(min(proposed_stake, max_stake), self._account_state(current_time),
                                      self.stoploss)
        return stake if (min_stake is None or stake >= min_stake) else 0.0

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force, current_time, entry_tag, side,
                            **kwargs) -> bool:
        if side != "long":
            return False
        state = self._account_state(current_time)
        age, qv = self._market_checks(pair, current_time)
        rm = self._rm()
        slots = self.config.get("kitb_max_open_positions")
        if slots and self.config.get("dry_run") and self.config.get("kitb_mode") == "learning":
            # Freigabe Burak (23.10.2026): KIs im Paper-Lernmodus bis zu `slots` gleichzeitige Positionen.
            # Alle übrigen Limits (Tagesverlust, Positionsgröße, Gesamteinsatz, Drawdown) gelten unverändert.
            rm = RiskManager(replace(rm.limits, max_open_positions=int(slots)))
        decision = rm.check_entry(pair=pair, stake=amount * rate, state=state,
                                          kill_switch=killswitch.is_active(), data_age_candles=age,
                                          quote_volume_24h=qv, enforce_drawdown_halt=self._is_live_like())
        if self._is_live_like():
            write_state(self.config.get("bot_name", "bot"), {
                "equity": round(state.equity, 2), "drawdown": round(state.drawdown, 4),
                "today_realized_pnl": round(state.today_realized_pnl, 2), "open_positions": state.open_positions,
                "last_decision": {"pair": pair, "allowed": decision.allowed, "reasons": decision.reasons}})
            if "max_drawdown" in decision.reasons:
                if self.config.get("kitb_mode") in ("learning", "explore") and _refill_enabled():
                    # Freigabe Burak (22.10.2026): Lern-/Sammler-Bots stoppen nur sich selbst; die Runde wird
                    # archiviert und mit frischem virtuellem Kapital neu begonnen (kitb.paper.archive_round).
                    self._request_new_round(state)
                elif not killswitch.is_active():
                    killswitch.activate(f"Max. Drawdown {state.drawdown:.1%} erreicht", "risk-module")
        if not decision.allowed:
            log.info("Risikomodul blockiert %s: %s", pair, decision.reasons)
            return False
        if self._is_live_like() and self.config.get("kitb_ai_filter"):
            # KI-Filter darf nur zusätzlich ablehnen, nie ein Risikolimit aufheben.
            from kitb import ai_memory

            try:
                ok, p = ai_memory.allow_entry(strategy=type(self).__name__, pair=pair, when=current_time)
            except Exception as e:  # noqa: BLE001 – Filterfehler darf den Bot nicht stoppen
                log.warning("KI-Filter nicht verfügbar (%s) – Trade ohne Filter.", e)
                ok, p = True, None
            log.info("KI-Filter %s: Gewinnchance %s -> %s", pair, "?" if p is None else f"{p:.0%}",
                     "erlaubt" if ok else "ausgelassen")
            return ok
        return decision.allowed

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """Sicherheitsabschaltung: bei aktivem Not-Stopp beendet sich der Paper-Bot geordnet."""
        if self._is_live_like() and killswitch.is_active():
            log.warning("Not-Stopp aktiv – Bot wird beendet.")
            os.kill(os.getpid(), signal.SIGTERM)
