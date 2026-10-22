"""Strategien aus ``user_data/strategies`` direkt laden (ohne Freqtrade-Bot), z. B. für Tests und Dashboard."""

import importlib.util

import pandas as pd

from kitb import paths


def load_strategy(name: str, timeframe: str = "1h"):
    spec = importlib.util.spec_from_file_location(name, paths.USER_DATA / "strategies" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)({"timeframe": timeframe, "stake_currency": "USDT", "runmode": None})


def compute_signals(strategy, df: pd.DataFrame, pair: str = "BTC/USDT") -> pd.DataFrame:
    meta = {"pair": pair}
    out = strategy.populate_indicators(df.copy(), meta)
    out = strategy.populate_entry_trend(out, meta)
    out = strategy.populate_exit_trend(out, meta)
    for col in ("enter_long", "exit_long"):
        if col not in out:
            out[col] = 0
    out[["enter_long", "exit_long"]] = out[["enter_long", "exit_long"]].fillna(0).astype(int)
    return out
