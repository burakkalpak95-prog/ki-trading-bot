"""Entscheidungsprotokoll: Was wusste der Bot beim Einstieg, was hat er getan, was kam heraus?

Quelle im Backtest: Freqtrade-Export ``signals`` (Signalkerzen mit allen Spalten inkl.
FreqAI-Vorhersage und ``kitb_explain``) verknüpft mit der Trade-Liste.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from kitb.data.timeframes import to_timedelta
from kitb.explain import EXPLAIN_COLUMN, parse
from kitb.registry import Registry

CONTEXT_COLUMNS = ["open", "high", "low", "close", "volume"]


def build_decisions(signals: dict[str, pd.DataFrame], trades: list[dict], *, strategy: str, timeframe: str,
                    entry_proba: float | None, regimes_by_pair: dict[str, pd.DataFrame] | None = None,
                    max_features: int = 12) -> list[dict[str, Any]]:
    """Eine Zeile pro ausgeführtem Einstieg. Die Signalkerze ist die Kerze **vor** dem Einstieg."""
    tf = to_timedelta(timeframe)
    out = []
    for t in trades:
        pair = t["pair"]
        sig = signals.get(pair)
        open_date = pd.Timestamp(t["open_date"]).tz_convert("UTC") if pd.Timestamp(t["open_date"]).tzinfo \
            else pd.Timestamp(t["open_date"], tz="UTC")
        signal_date = open_date - tf
        row = None
        if sig is not None and not sig.empty:
            match = sig[pd.to_datetime(sig["date"], utc=True) == signal_date]
            row = match.iloc[-1] if not match.empty else None
        proba = float(row["up"]) if row is not None and "up" in row and pd.notna(row["up"]) else None
        explanation = row.get(EXPLAIN_COLUMN) if row is not None else None
        features = {}
        if row is not None:
            feat_cols = [c for c in row.index if str(c).startswith("%")]
            chosen = list(parse(explanation)) if explanation else []
            chosen += [c for c in feat_cols if c not in chosen][: max(0, max_features - len(chosen))]
            features = {c: _num(row.get(c)) for c in chosen if c in row.index}
            if not features and explanation:
                # FreqAI entfernt Rohfeatures aus dem Strategie-DataFrame: dann die normierten Beiträge speichern.
                features = {"top_contributions": parse(explanation)}
        market = {c: _num(row.get(c)) for c in CONTEXT_COLUMNS} if row is not None else {}
        if row is not None and row.get("high") and row.get("close"):
            market["range_pct"] = _num((row["high"] - row["low"]) / row["close"])
        regime = _regime_at(regimes_by_pair, pair, open_date)
        market.update(regime)
        profit = t.get("profit_abs", 0.0)
        fees = t["amount"] * t["open_rate"] * t.get("fee_open", 0) + t["amount"] * t["close_rate"] * t.get("fee_close", 0)
        out.append({
            "strategy": strategy, "pair": pair, "decided_at": (signal_date + tf).isoformat(),
            "action": "enter_long", "proba_up": proba,
            "do_predict": int(row["do_predict"]) if row is not None and "do_predict" in row and pd.notna(row["do_predict"]) else None,
            "explanation": explanation, "market": market, "features": features,
            "outcome": {"profit_abs": profit, "profit_ratio": t.get("profit_ratio"), "exit_reason": t.get("exit_reason"),
                        "close_date": str(t.get("close_date")), "duration_min": t.get("trade_duration"),
                        "fees_abs": round(fees, 6)},
            "risk": {"stake_amount": t.get("stake_amount"), "stop_loss_ratio": t.get("stop_loss_ratio"),
                     "initial_stop_loss_ratio": t.get("initial_stop_loss_ratio")},
            "model_error": _model_error(proba, entry_proba, profit),
        })
    return out


def _num(v: Any) -> float | None:
    try:
        return None if v is None or pd.isna(v) else round(float(v), 8)
    except (TypeError, ValueError):
        return None


def _regime_at(regimes_by_pair, pair, ts) -> dict[str, str]:
    if not regimes_by_pair or pair not in regimes_by_pair:
        return {}
    reg = regimes_by_pair[pair]
    known = reg[reg["valid_from"] <= ts]
    if known.empty:
        return {"trend": "unknown", "volatility": "unknown"}
    last = known.iloc[-1]
    return {"trend": last["trend"], "volatility": last["volatility"]}


def _model_error(proba: float | None, entry_proba: float | None, profit: float) -> str | None:
    if proba is None:
        return None
    if entry_proba is not None and proba >= entry_proba and profit <= 0:
        return "false_positive"
    return None


def store_decisions(registry: Registry, decisions: list[dict], *, source: str, run_ref: str, bot: str | None,
                    model_identifier: str | None, model_version_id: int | None = None) -> int:
    for d in decisions:
        registry.insert("decisions", {
            "source": source, "run_ref": run_ref, "bot": bot, "strategy": d["strategy"],
            "model_identifier": model_identifier, "model_version_id": model_version_id, "pair": d["pair"],
            "decided_at": d["decided_at"], "action": d["action"], "proba_up": d["proba_up"],
            "do_predict": d["do_predict"], "explanation": d["explanation"],
            "market": json.dumps(d["market"], default=str), "features": json.dumps(d["features"], default=str),
            "outcome": json.dumps(d["outcome"], default=str), "risk": json.dumps(d["risk"], default=str),
            "model_error": d["model_error"]})
    return len(decisions)


def load_signals(result_file: Path, strategy: str) -> dict[str, pd.DataFrame]:
    from freqtrade.data.btanalysis import load_backtest_analysis_data

    data = load_backtest_analysis_data(Path(result_file), "signals") or {}
    return data.get(strategy, {})
