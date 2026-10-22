# KI-Bot-Vergleich 20261022-200847-cf69c0

> **SYNTHETISCHE DATEN** (Zufallsprozess ohne echtes Muster). Ein ML-Vorteil wäre hier ein Warnsignal für Datenlecks; erwartet ist kein Vorteil.

Walk-Forward: Training 90 Tage, Neutraining alle 30 Tage, Label-Horizont 12 Kerzen, Schwelle 0.60 %.
Test-Abschnitt verwendet: **nein (unberührt)**

- **bot_a** (BotAEfficient, KitbLogisticRegression): Effizient: logistische Regression, wenige Features, nur 1h
- **bot_b** (BotBExtended, KitbLightGBMClassifier): Erweitert: LightGBM, mehr Features, 1h+1d, BTC/ETH-Korrelation, verschobene Kerzen

## Abschnitt: validation `20240315-20240808`

| Bot/Strategie | Trades | Netto | Max DD | PF | Sharpe | Trefferquote | Exposure | Kosten | vs. B&H | Stichprobe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BaselineEmaCross | 70 | -6.36 % | 9.43 % | 0.75 | -0.77 | 27.14 % | 50.03 % | 462.41 | -4.78 % | ok |
| TrendEmaAdx | 43 | -4.98 % | 7.58 % | 0.59 | -0.99 | 23.26 % | 12.14 % | 362.14 | -3.41 % | ok |
| MomentumMacdRsi | 88 | -11.82 % | 13.39 % | 0.51 | -2.98 | 21.59 % | 10.10 % | 930.85 | -10.24 % | ok |
| MeanReversionBollinger | 3 | -0.18 % | 0.20 % | 0.49 | -0.14 | 33.33 % | 0.80 % | 28.85 | 1.39 % | zu klein |
| BreakoutDonchian | 44 | 4.97 % | 5.55 % | 1.34 | 0.48 | 27.27 % | 17.04 % | 381.94 | 6.54 % | ok |
| VolatilitySqueeze | 60 | -3.76 % | 7.57 % | 0.77 | -0.78 | 33.33 % | 12.69 % | 666.35 | -2.18 % | ok |
| bot_a | 55 | 0.48 % | 15.40 % | 1.01 | 0.03 | 43.64 % | 27.31 % | 625.53 | 2.06 % | ok |
| bot_b | 234 | -16.69 % | 20.08 % | 0.71 | -3.12 | 54.70 % | 18.15 % | 2347.71 | -15.11 % | ok |

Buy&Hold (gleichgewichtet, nach Kosten): -1.57 %

| Bot | Laufzeit (s) | CPU (s) | Spitzen-RAM (MB) | Modelle auf Disk (KB) | Trainings |
|---|---:|---:|---:|---:|---:|
| bot_a | 2.72 | 4.9 | 417.3 | 393.6 | 10 |
| bot_b | 34.79 | 119.37 | 555.8 | 557.4 | 10 |
- bot_a: Modellversion #3 – Freigabekriterien nicht erfüllt: min_profit_factor, max_mean_profit_pvalue
- bot_b: Modellversion #4 – Freigabekriterien nicht erfüllt: min_profit_factor, min_profit_total, max_mean_profit_pvalue, beats_buy_and_hold, beats_reference


**Einordnung:** Bot B bringt in diesem Abschnitt keinen Mehrwert gegenüber Bot A. Bot B − Bot A: -17.17 %; Laufzeitfaktor Bot B/Bot A: 12.79×; beste Regel-Strategie: BreakoutDonchian.

Hinweis: Kein Signifikanztest. Vergangene Ergebnisse garantieren keine zukünftigen Gewinne.
