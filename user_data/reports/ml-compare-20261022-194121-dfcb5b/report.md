# KI-Bot-Vergleich 20261022-194121-dfcb5b

> **SYNTHETISCHE DATEN** (Zufallsprozess ohne echtes Muster). Ein ML-Vorteil wäre hier ein Warnsignal für Datenlecks; erwartet ist kein Vorteil.

Walk-Forward: Training 90 Tage, Neutraining alle 30 Tage, Label-Horizont 12 Kerzen, Schwelle 0.60 %.
Test-Abschnitt verwendet: **nein (unberührt)**

- **bot_a** (BotAEfficient, KitbLogisticRegression): Effizient: logistische Regression, wenige Features, nur 1h
- **bot_b** (BotBExtended, KitbLightGBMClassifier): Erweitert: LightGBM, mehr Features, 1h+1d, BTC/ETH-Korrelation, verschobene Kerzen

## Abschnitt: validation `20240315-20240808`

| Bot/Strategie | Trades | Netto | Max DD | PF | Sharpe | Trefferquote | Exposure | Kosten | vs. B&H | Stichprobe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BaselineEmaCross | 70 | -12.48 % | 18.61 % | 0.76 | -0.72 | 27.14 % | 50.03 % | 951.47 | -10.90 % | ok |
| TrendEmaAdx | 43 | -8.09 % | 12.09 % | 0.59 | -0.99 | 23.26 % | 12.14 % | 584.40 | -6.52 % | ok |
| MomentumMacdRsi | 88 | -15.08 % | 17.21 % | 0.52 | -2.88 | 21.59 % | 10.10 % | 1196.24 | -13.51 % | ok |
| MeanReversionBollinger | 3 | -0.26 % | 0.28 % | 0.49 | -0.14 | 33.33 % | 0.80 % | 40.82 | 1.31 % | zu klein |
| BreakoutDonchian | 44 | 8.43 % | 8.50 % | 1.37 | 0.50 | 27.27 % | 17.04 % | 600.84 | 10.00 % | ok |
| VolatilitySqueeze | 60 | -4.46 % | 9.27 % | 0.78 | -0.74 | 33.33 % | 12.69 % | 816.61 | -2.88 % | ok |
| bot_a | 55 | 2.38 % | 18.63 % | 1.05 | 0.10 | 43.64 % | 27.41 % | 749.65 | 3.95 % | ok |
| bot_b | 222 | -23.29 % | 27.02 % | 0.69 | -3.22 | 53.60 % | 17.99 % | 3020.29 | -21.71 % | ok |

Buy&Hold (gleichgewichtet, nach Kosten): -1.57 %

| Bot | Laufzeit (s) | CPU (s) | Spitzen-RAM (MB) | Modelle auf Disk (KB) | Trainings |
|---|---:|---:|---:|---:|---:|
| bot_a | 2.94 | 5.23 | 426.9 | 393.6 | 10 |
| bot_b | 34.28 | 117.64 | 552.4 | 557.4 | 10 |
- bot_a: Modellversion #1 – Freigabekriterien erfüllt
- bot_b: Modellversion #2 – Freigabekriterien nicht erfüllt: min_profit_factor, max_drawdown, min_profit_total, beats_buy_and_hold, beats_reference


**Einordnung:** Bot B bringt in diesem Abschnitt keinen Mehrwert gegenüber Bot A. Bot B − Bot A: -25.66 %; Laufzeitfaktor Bot B/Bot A: 11.66×; beste Regel-Strategie: BreakoutDonchian.

Hinweis: Kein Signifikanztest. Vergangene Ergebnisse garantieren keine zukünftigen Gewinne.
