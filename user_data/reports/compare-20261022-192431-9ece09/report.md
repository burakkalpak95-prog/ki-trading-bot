# Strategievergleich 20261022-192431-9ece09

> **SYNTHETISCHE DATEN** – reiner Funktionstest, keine Aussage über reale Märkte.

Datenquelle: `synthetic` · Zeitintervall: `1h` · Test-Abschnitt verwendet: **nein (unberührt)**

Abschnitte: train `20230101-20240315`, validation `20240315-20240808`, test `20240808-20250101`

## Abschnitt: train

| Strategie | Trades | Netto | Max DD | Profit Factor | Sharpe | Trefferquote | Exposure | Kosten (USDT) | vs. Buy&Hold | Stichprobe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BaselineEmaCross | 161 | 22.63 % | 40.71 % | 1.16 | 0.35 | 27.33 % | 44.99 % | 2197.10 | 26.76 % | ok |
| TrendEmaAdx | 114 | 14.44 % | 9.67 % | 1.27 | 0.33 | 39.47 % | 12.30 % | 1555.43 | 18.56 % | ok |
| MomentumMacdRsi | 271 | -23.06 % | 30.99 % | 0.80 | -0.96 | 31.37 % | 11.37 % | 3687.80 | -18.94 % | ok |
| MeanReversionBollinger | 10 | -2.65 % | 5.32 % | 0.63 | -0.08 | 60.00 % | 0.63 % | 135.78 | 1.47 % | zu klein |
| BreakoutDonchian | 112 | 9.09 % | 12.04 % | 1.12 | 0.18 | 34.82 % | 13.49 % | 1527.29 | 13.22 % | ok |
| VolatilitySqueeze | 170 | -31.29 % | 31.93 % | 0.61 | -1.38 | 27.65 % | 11.13 % | 2310.52 | -27.17 % | ok |

Buy&Hold (BTC/ETH gleichgewichtet, nach Kosten): -4.12 %

## Abschnitt: validation

| Strategie | Trades | Netto | Max DD | Profit Factor | Sharpe | Trefferquote | Exposure | Kosten (USDT) | vs. Buy&Hold | Stichprobe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BaselineEmaCross | 70 | -12.48 % | 18.61 % | 0.76 | -0.72 | 27.14 % | 50.03 % | 951.47 | -10.90 % | ok |
| TrendEmaAdx | 43 | -8.09 % | 12.09 % | 0.59 | -0.99 | 23.26 % | 12.14 % | 584.40 | -6.52 % | ok |
| MomentumMacdRsi | 88 | -15.08 % | 17.21 % | 0.52 | -2.88 | 21.59 % | 10.10 % | 1196.24 | -13.51 % | ok |
| MeanReversionBollinger | 3 | -0.26 % | 0.28 % | 0.49 | -0.14 | 33.33 % | 0.80 % | 40.82 | 1.31 % | zu klein |
| BreakoutDonchian | 44 | 8.43 % | 8.50 % | 1.37 | 0.50 | 27.27 % | 17.04 % | 600.84 | 10.00 % | ok |
| VolatilitySqueeze | 60 | -4.46 % | 9.27 % | 0.78 | -0.74 | 33.33 % | 12.69 % | 816.61 | -2.88 % | ok |

Buy&Hold (BTC/ETH gleichgewichtet, nach Kosten): -1.57 %

## Stabilität über rollierende Fenster

| Strategie | Fenster | Anteil positiv | Anteil besser als Buy&Hold |
|---|---:|---:|---:|
| BaselineEmaCross | 6 | 50.00 % | 50.00 % |
| TrendEmaAdx | 6 | 66.70 % | 50.00 % |
| MomentumMacdRsi | 6 | 16.70 % | 50.00 % |
| MeanReversionBollinger | 6 | 33.30 % | 50.00 % |
| BreakoutDonchian | 6 | 66.70 % | 50.00 % |
| VolatilitySqueeze | 6 | 0.00 % | 33.30 % |

## Ergebnis nach Marktphase (beim Einstieg bekannte Phase)

**BaselineEmaCross**

| Trend | Volatilität | Trades | Gewinn (USDT) | Ø Gewinn/Trade | Trefferquote | Stichprobe |
|---|---|---:|---:|---:|---:|---|
| down | high | 44 | -1479.96 | -0.84 % | 15.90 % | ok |
| down | low | 54 | 3492.47 | 1.61 % | 33.30 % | ok |
| sideways | high | 23 | 426.02 | 0.46 % | 21.70 % | zu klein |
| sideways | low | 30 | -790.28 | -0.66 % | 33.30 % | ok |
| unknown | unknown | 15 | 474.72 | 0.79 % | 33.30 % | zu klein |
| up | high | 34 | -223.40 | -0.16 % | 29.40 % | ok |
| up | low | 31 | -883.80 | -0.71 % | 25.80 % | ok |

**TrendEmaAdx**

| Trend | Volatilität | Trades | Gewinn (USDT) | Ø Gewinn/Trade | Trefferquote | Stichprobe |
|---|---|---:|---:|---:|---:|---|
| down | high | 21 | -243.13 | -0.29 % | 28.60 % | zu klein |
| down | low | 19 | -312.86 | -0.41 % | 26.30 % | zu klein |
| sideways | high | 17 | 1689.31 | 2.48 % | 41.20 % | zu klein |
| sideways | low | 22 | -335.75 | -0.38 % | 36.40 % | zu klein |
| unknown | unknown | 13 | 954.45 | 1.83 % | 46.20 % | zu klein |
| up | high | 29 | -495.28 | -0.43 % | 34.50 % | zu klein |
| up | low | 36 | -621.88 | -0.43 % | 36.10 % | ok |

**MomentumMacdRsi**

| Trend | Volatilität | Trades | Gewinn (USDT) | Ø Gewinn/Trade | Trefferquote | Stichprobe |
|---|---|---:|---:|---:|---:|---|
| down | high | 39 | 308.44 | 0.20 % | 30.80 % | ok |
| down | low | 76 | -610.50 | -0.20 % | 28.90 % | ok |
| sideways | high | 35 | -1437.59 | -1.03 % | 20.00 % | ok |
| sideways | low | 71 | -1834.22 | -0.65 % | 21.10 % | ok |
| unknown | unknown | 14 | 120.37 | 0.21 % | 42.90 % | zu klein |
| up | high | 55 | 240.99 | 0.11 % | 32.70 % | ok |
| up | low | 69 | -602.12 | -0.22 % | 34.80 % | ok |

**MeanReversionBollinger**

| Trend | Volatilität | Trades | Gewinn (USDT) | Ø Gewinn/Trade | Trefferquote | Stichprobe |
|---|---|---:|---:|---:|---:|---|
| down | low | 1 | 24.61 | 0.61 % | 100.00 % | zu klein |
| sideways | high | 2 | 72.04 | 0.90 % | 100.00 % | zu klein |
| sideways | low | 3 | -53.36 | -0.44 % | 0.00 % | zu klein |
| unknown | unknown | 3 | -41.36 | -0.34 % | 66.70 % | zu klein |
| up | high | 3 | -305.73 | -2.54 % | 33.30 % | zu klein |
| up | low | 1 | 13.02 | 0.33 % | 100.00 % | zu klein |

**BreakoutDonchian**

| Trend | Volatilität | Trades | Gewinn (USDT) | Ø Gewinn/Trade | Trefferquote | Stichprobe |
|---|---|---:|---:|---:|---:|---|
| down | high | 24 | 374.14 | 0.39 % | 33.30 % | zu klein |
| down | low | 29 | 886.04 | 0.76 % | 24.10 % | zu klein |
| sideways | high | 21 | 896.78 | 1.07 % | 28.60 % | zu klein |
| sideways | low | 23 | -374.81 | -0.41 % | 30.40 % | zu klein |
| unknown | unknown | 9 | 1341.77 | 3.72 % | 55.60 % | zu klein |
| up | high | 24 | -305.19 | -0.32 % | 41.70 % | zu klein |
| up | low | 26 | -1066.62 | -1.02 % | 30.80 % | zu klein |

**VolatilitySqueeze**

| Trend | Volatilität | Trades | Gewinn (USDT) | Ø Gewinn/Trade | Trefferquote | Stichprobe |
|---|---|---:|---:|---:|---:|---|
| down | high | 44 | -691.69 | -0.39 % | 25.00 % | ok |
| down | low | 39 | 156.95 | 0.10 % | 38.50 % | ok |
| sideways | high | 24 | -626.91 | -0.65 % | 20.80 % | zu klein |
| sideways | low | 40 | -427.42 | -0.27 % | 35.00 % | ok |
| unknown | unknown | 8 | 93.73 | 0.29 % | 37.50 % | zu klein |
| up | high | 40 | -1129.55 | -0.70 % | 22.50 % | ok |
| up | low | 35 | -950.25 | -0.68 % | 28.60 % | ok |

Hinweise: Kennzahlen mit weniger als 30 Trades sind statistisch kaum belastbar. Vergangene Ergebnisse garantieren keine zukünftigen Gewinne.
