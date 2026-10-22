# Strategievergleich 20261022-194634-1f2a61

> **SYNTHETISCHE DATEN** – reiner Funktionstest, keine Aussage über reale Märkte.

Datenquelle: `synthetic` · Zeitintervall: `1h` · Test-Abschnitt verwendet: **nein (unberührt)**

Abschnitte: train `20230101-20240315`, validation `20240315-20240808`, test `20240808-20250101`

## Abschnitt: train

| Strategie | Trades | Netto | Max DD | Profit Factor | Sharpe | Trefferquote | Exposure | Kosten (USDT) | vs. Buy&Hold | Stichprobe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AgentGatedBaseline | 71 | 37.06 % | 13.25 % | 1.63 | 0.49 | 33.80 % | 20.42 % | 973.52 | 41.18 % | ok |
| BaselineEmaCross | 161 | 15.02 % | 41.57 % | 1.10 | 0.23 | 26.09 % | 43.35 % | 2195.80 | 19.14 % | ok |
| AgentGatedBreakout | 71 | -3.39 % | 12.44 % | 0.93 | -0.08 | 32.39 % | 8.11 % | 966.63 | 0.74 % | ok |
| BreakoutDonchian | 113 | 8.04 % | 12.15 % | 1.10 | 0.15 | 34.51 % | 13.45 % | 1540.73 | 12.16 % | ok |

Buy&Hold (BTC/ETH gleichgewichtet, nach Kosten): -4.12 %

## Abschnitt: validation

| Strategie | Trades | Netto | Max DD | Profit Factor | Sharpe | Trefferquote | Exposure | Kosten (USDT) | vs. Buy&Hold | Stichprobe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AgentGatedBaseline | 28 | -19.52 % | 19.52 % | 0.06 | -3.38 | 17.86 % | 16.25 % | 378.11 | -17.94 % | zu klein |
| BaselineEmaCross | 70 | -12.48 % | 18.61 % | 0.76 | -0.72 | 27.14 % | 50.03 % | 951.47 | -10.90 % | ok |
| AgentGatedBreakout | 28 | -7.99 % | 9.89 % | 0.43 | -1.35 | 28.57 % | 9.35 % | 380.08 | -6.42 % | zu klein |
| BreakoutDonchian | 44 | 8.43 % | 8.50 % | 1.37 | 0.50 | 27.27 % | 17.04 % | 600.84 | 10.00 % | ok |

Buy&Hold (BTC/ETH gleichgewichtet, nach Kosten): -1.57 %

## Stabilität über rollierende Fenster

| Strategie | Fenster | Anteil positiv | Anteil besser als Buy&Hold |
|---|---:|---:|---:|
| AgentGatedBaseline | 6 | 50.00 % | 33.30 % |
| BaselineEmaCross | 6 | 50.00 % | 50.00 % |
| AgentGatedBreakout | 6 | 33.30 % | 50.00 % |
| BreakoutDonchian | 6 | 66.70 % | 50.00 % |

## Ergebnis nach Marktphase (beim Einstieg bekannte Phase)

**AgentGatedBaseline**

| Trend | Volatilität | Trades | Gewinn (USDT) | Ø Gewinn/Trade | Trefferquote | Stichprobe |
|---|---|---:|---:|---:|---:|---|
| down | high | 10 | -164.17 | -0.41 % | 40.00 % | zu klein |
| down | low | 16 | 898.21 | 1.40 % | 25.00 % | zu klein |
| sideways | high | 15 | 2610.49 | 4.34 % | 33.30 % | zu klein |
| sideways | low | 11 | -585.04 | -1.33 % | 27.30 % | zu klein |
| unknown | unknown | 9 | -924.72 | -2.56 % | 22.20 % | zu klein |
| up | high | 21 | -379.92 | -0.45 % | 23.80 % | zu klein |
| up | low | 17 | 299.24 | 0.44 % | 35.30 % | zu klein |

**BaselineEmaCross**

| Trend | Volatilität | Trades | Gewinn (USDT) | Ø Gewinn/Trade | Trefferquote | Stichprobe |
|---|---|---:|---:|---:|---:|---|
| down | high | 44 | -1479.96 | -0.84 % | 15.90 % | ok |
| down | low | 54 | 3492.47 | 1.61 % | 33.30 % | ok |
| sideways | high | 23 | 426.02 | 0.46 % | 21.70 % | zu klein |
| sideways | low | 30 | -790.28 | -0.66 % | 33.30 % | ok |
| unknown | unknown | 17 | 231.98 | 0.34 % | 29.40 % | zu klein |
| up | high | 32 | -741.95 | -0.58 % | 25.00 % | ok |
| up | low | 31 | -883.80 | -0.71 % | 25.80 % | ok |

**AgentGatedBreakout**

| Trend | Volatilität | Trades | Gewinn (USDT) | Ø Gewinn/Trade | Trefferquote | Stichprobe |
|---|---|---:|---:|---:|---:|---|
| down | high | 6 | -183.81 | -0.77 % | 33.30 % | zu klein |
| down | low | 14 | -586.18 | -1.04 % | 14.30 % | zu klein |
| sideways | high | 15 | 1180.30 | 1.96 % | 33.30 % | zu klein |
| sideways | low | 15 | -195.37 | -0.33 % | 33.30 % | zu klein |
| unknown | unknown | 7 | 516.31 | 1.84 % | 42.90 % | zu klein |
| up | high | 19 | -917.09 | -1.21 % | 36.80 % | zu klein |
| up | low | 23 | -952.17 | -1.03 % | 30.40 % | zu klein |

**BreakoutDonchian**

| Trend | Volatilität | Trades | Gewinn (USDT) | Ø Gewinn/Trade | Trefferquote | Stichprobe |
|---|---|---:|---:|---:|---:|---|
| down | high | 24 | 374.14 | 0.39 % | 33.30 % | zu klein |
| down | low | 29 | 886.04 | 0.76 % | 24.10 % | zu klein |
| sideways | high | 21 | 896.78 | 1.07 % | 28.60 % | zu klein |
| sideways | low | 23 | -374.81 | -0.41 % | 30.40 % | zu klein |
| unknown | unknown | 10 | 1236.43 | 3.09 % | 50.00 % | zu klein |
| up | high | 24 | -305.19 | -0.32 % | 41.70 % | zu klein |
| up | low | 26 | -1066.62 | -1.02 % | 30.80 % | zu klein |

Hinweise: Kennzahlen mit weniger als 30 Trades sind statistisch kaum belastbar. Vergangene Ergebnisse garantieren keine zukünftigen Gewinne.
