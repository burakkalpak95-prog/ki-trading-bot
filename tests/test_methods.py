"""Methoden-Suche: alle Methoden laufen im Backtest, Auswahlregel verlangt Treffer UND Gewinn in jedem Zeitraum."""

import pandas as pd

from kitb import methods, paths
from kitb.data.synthetic import write_synthetic_dataset


def test_all_methods_backtest_on_synthetic(tmp_path):
    data = tmp_path / "synthetic"
    write_synthetic_dataset(data, ["BTC/USDT", "ETH/USDT"], ["1h"], "2022-01-01", "2022-08-01", seed=3)
    res = methods.scan(datadir=data, data_source="synthetic", timeframes=["1h"],
                       periods={"früh": "20220201-20220430", "spät": "20220501-20220731"}, offline=True,
                       pairs=["BTC/USDT", "ETH/USDT"])
    assert set(res["Methode"]) == set(methods.method_names()) and len(methods.method_names()) >= 15
    assert (res["Trades"] >= 0).all() and set(res["Zeitraum"]) == {"früh", "spät"}
    assert "Alle Ergebnisse" in methods.render(res, methods.select(res))


def test_selection_needs_winrate_profit_and_both_periods():
    r = pd.DataFrame([
        {"Methode": "A", "Zeitrahmen": "1h", "Zeitraum": "früh", "Trades": 80, "Trades/Monat": 3, "Trefferquote": 0.7, "Ergebnis": 0.1},
        {"Methode": "A", "Zeitrahmen": "1h", "Zeitraum": "spät", "Trades": 50, "Trades/Monat": 3, "Trefferquote": 0.65, "Ergebnis": 0.05},
        {"Methode": "B", "Zeitrahmen": "1h", "Zeitraum": "früh", "Trades": 80, "Trades/Monat": 3, "Trefferquote": 0.8, "Ergebnis": -0.2},
        {"Methode": "B", "Zeitrahmen": "1h", "Zeitraum": "spät", "Trades": 80, "Trades/Monat": 3, "Trefferquote": 0.8, "Ergebnis": 0.1},
        {"Methode": "C", "Zeitrahmen": "1h", "Zeitraum": "früh", "Trades": 80, "Trades/Monat": 3, "Trefferquote": 0.7, "Ergebnis": 0.1},
        {"Methode": "C", "Zeitrahmen": "1h", "Zeitraum": "spät", "Trades": 10, "Trades/Monat": 1, "Trefferquote": 0.9, "Ergebnis": 0.1},
    ])
    sel = methods.select(r).set_index("Methode")["bestanden"]
    assert sel["A"] and not sel["B"] and not sel["C"]  # B: Verlust trotz 80 % Treffer, C: zu wenig Trades
