"""Formel-Faktoren (kitb.alphas): alle Faktoren rechnen, IC erkennt echte Vorhersagekraft, Auswahl verlangt beide Zeiträume."""

import numpy as np
import pandas as pd

from kitb import alphas


def _panel(n_coins=12, days=900, seed=1, signal=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=days, freq="D", tz="UTC")
    cols = [f"C{i}/USDT" for i in range(n_coins)]
    r = rng.normal(0, 0.03, (days, n_coins))
    if signal:  # Kurzfrist-Umkehr einbauen: Rendite morgen = −signal × Rendite der letzten 5 Tage
        past5 = pd.DataFrame(r).rolling(5).sum().shift(1).fillna(0).to_numpy()
        r = r - signal * past5
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), idx, cols)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.02, close.shape))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.02, close.shape))
    vol = pd.DataFrame(rng.lognormal(10, 0.5, close.shape), idx, cols)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": vol}


PERIODS = {"früh": ("2022-02-15", "2023-03-31"), "spät": ("2023-04-01", "2024-06-01")}


def test_all_factors_compute_and_random_data_has_no_live_factor():
    res = alphas.bench(_panel(), PERIODS)
    assert set(res["Faktor"]) == set(alphas.FACTORS) and len(alphas.FACTORS) >= 30
    assert (res.loc[res["Horizont_Tage"] == 1, "Tage"] > 150).all() and (res["Tage"] > 40).all()
    sel = alphas.select(res)
    assert sel["lebt"].sum() <= 2  # reiner Zufall: höchstens vereinzelte Zufallstreffer


def test_planted_reversal_is_found():
    sel = alphas.select(alphas.bench(_panel(signal=0.15), PERIODS)).set_index(["Faktor", "Horizont_Tage"])
    assert sel.loc[("x_strev5", 1), "lebt"] and sel.loc[("x_strev5", 1), "Richtung"] == "+"


def test_forward_returns_no_overlap_and_ic_needs_enough_coins():
    p = _panel(n_coins=4, days=60)
    assert len(alphas.forward_returns(p, 5)) == 12
    assert alphas.ic_series(alphas.FACTORS["ts_kmid"](p), alphas.forward_returns(p, 1)).empty  # < 5 Coins
