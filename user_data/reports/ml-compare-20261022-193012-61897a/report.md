# KI-Bot-Vergleich 20261022-193012-61897a

> **SYNTHETISCHE DATEN** (Zufallsprozess ohne echtes Muster). Ein ML-Vorteil wäre hier ein Warnsignal für Datenlecks; erwartet ist kein Vorteil.

Walk-Forward: Training 90 Tage, Neutraining alle 30 Tage, Label-Horizont 12 Kerzen, Schwelle 0.60 %.
Test-Abschnitt verwendet: **nein (unberührt)**

- **bot_a** (BotAEfficient, KitbLogisticRegression): Effizient: logistische Regression, wenige Features, nur 1h
- **bot_b** (BotBExtended, LightGBMClassifier): Erweitert: LightGBM, mehr Features, 1h+1d, BTC/ETH-Korrelation, verschobene Kerzen

## Abschnitt: validation `20240315-20240808`

| Bot/Strategie | Trades | Netto | Max DD | PF | Sharpe | Trefferquote | Exposure | Kosten | vs. B&H | Stichprobe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bot_a | 27 | -17.08 % | 18.01 % | 0.35 | -1.51 | 37.04 % | 6.62 % | 364.91 | -15.51 % | zu klein |
| bot_b | 226 | -22.47 % | 26.93 % | 0.71 | -3.14 | 51.33 % | 17.01 % | 3074.92 | -20.89 % | ok |
| BaselineEmaCross | 70 | -12.48 % | 18.61 % | 0.76 | -0.72 | 27.14 % | 50.03 % | 951.47 | -10.90 % | ok |
| TrendEmaAdx | 43 | -8.09 % | 12.09 % | 0.59 | -0.99 | 23.26 % | 12.14 % | 584.40 | -6.52 % | ok |
| MomentumMacdRsi | 88 | -15.08 % | 17.21 % | 0.52 | -2.88 | 21.59 % | 10.10 % | 1196.24 | -13.51 % | ok |
| MeanReversionBollinger | 3 | -0.26 % | 0.28 % | 0.49 | -0.14 | 33.33 % | 0.80 % | 40.82 | 1.31 % | zu klein |
| BreakoutDonchian | 44 | 8.43 % | 8.50 % | 1.37 | 0.50 | 27.27 % | 17.04 % | 600.84 | 10.00 % | ok |
| VolatilitySqueeze | 60 | -4.46 % | 9.27 % | 0.78 | -0.74 | 33.33 % | 12.69 % | 816.61 | -2.88 % | ok |

Buy&Hold (gleichgewichtet, nach Kosten): -1.57 %

| Bot | Laufzeit (s) | CPU (s) | Spitzen-RAM (MB) | Modelle auf Disk (KB) | Trainings |
|---|---:|---:|---:|---:|---:|
| bot_a | 2.85 | 4.97 | 401.9 | 236.2 | 10 |
| bot_b | 25.17 | 83.84 | 532.3 | 368.9 | 10 |

**Einordnung:** Stichprobe zu klein für eine Aussage. Bot B − Bot A: -5.38 %; Laufzeitfaktor Bot B/Bot A: 8.83×; beste Regel-Strategie: BreakoutDonchian.

Hinweis: Kein Signifikanztest. Vergangene Ergebnisse garantieren keine zukünftigen Gewinne.
