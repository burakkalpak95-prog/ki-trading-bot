# Strategievergleich 20261022-200943-91cb88

> **SYNTHETISCHE DATEN** – reiner Funktionstest, keine Aussage über reale Märkte.

Datenquelle: `synthetic` · Zeitintervall: `1h` · Test-Abschnitt verwendet: **nein (unberührt)**

Abschnitte: train `20230101-20240315`, validation `20240315-20240808`, test `20240808-20250101`

## Abschnitt: train

| Strategie | Trades | Netto | Max DD | Profit Factor | Sharpe | Trefferquote | Exposure | Kosten (USDT) | vs. Buy&Hold | Stichprobe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AgentGatedBaseline | 71 | 18.45 % | 6.56 % | 1.60 | 0.47 | 33.80 % | 20.42 % | 517.13 | 22.57 % | ok |
| BaselineEmaCross | 176 | 17.65 % | 21.23 % | 1.22 | 0.51 | 28.41 % | 48.96 % | 1259.25 | 21.78 % | ok |
| AgentGatedBreakout | 71 | -2.65 % | 7.99 % | 0.91 | -0.10 | 32.39 % | 8.11 % | 596.33 | 1.47 % | ok |
| BreakoutDonchian | 113 | 3.74 % | 8.90 % | 1.07 | 0.11 | 34.51 % | 13.45 % | 1004.52 | 7.86 % | ok |

Buy&Hold (BTC/ETH gleichgewichtet, nach Kosten): -4.12 %

## Abschnitt: validation

| Strategie | Trades | Netto | Max DD | Profit Factor | Sharpe | Trefferquote | Exposure | Kosten (USDT) | vs. Buy&Hold | Stichprobe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AgentGatedBaseline | 28 | -9.26 % | 9.26 % | 0.06 | -3.36 | 17.86 % | 16.25 % | 179.38 | -7.69 % | zu klein |
| BaselineEmaCross | 70 | -6.36 % | 9.43 % | 0.75 | -0.77 | 27.14 % | 50.03 % | 462.41 | -4.78 % | ok |
| AgentGatedBreakout | 28 | -4.86 % | 6.00 % | 0.43 | -1.35 | 28.57 % | 9.35 % | 230.49 | -3.28 % | zu klein |
| BreakoutDonchian | 44 | 4.97 % | 5.55 % | 1.34 | 0.48 | 27.27 % | 17.04 % | 381.94 | 6.54 % | ok |

Buy&Hold (BTC/ETH gleichgewichtet, nach Kosten): -1.57 %

## Stabilität über rollierende Fenster

| Strategie | Fenster | Anteil positiv | Anteil besser als Buy&Hold |
|---|---:|---:|---:|
| AgentGatedBaseline | 6 | 50.00 % | 50.00 % |
| BaselineEmaCross | 6 | 50.00 % | 50.00 % |
| AgentGatedBreakout | 6 | 33.30 % | 50.00 % |
| BreakoutDonchian | 6 | 66.70 % | 50.00 % |

## Ergebnis nach Marktphase (beim Einstieg bekannte Phase)

**AgentGatedBaseline**

| Trend | Volatilität | Trades | Gewinn (USDT) | Ø Gewinn/Trade | Trefferquote | Stichprobe |
|---|---|---:|---:|---:|---:|---|
| down | high | 10 | -93.88 | -0.41 % | 40.00 % | zu klein |
| down | low | 16 | 441.83 | 1.40 % | 25.00 % | zu klein |
| sideways | high | 15 | 1335.64 | 4.34 % | 33.30 % | zu klein |
| sideways | low | 11 | -297.04 | -1.33 % | 27.30 % | zu klein |
| unknown | unknown | 9 | -451.20 | -2.56 % | 22.20 % | zu klein |
| up | high | 21 | -177.96 | -0.45 % | 23.80 % | zu klein |
| up | low | 17 | 161.39 | 0.44 % | 35.30 % | zu klein |

**BaselineEmaCross**

| Trend | Volatilität | Trades | Gewinn (USDT) | Ø Gewinn/Trade | Trefferquote | Stichprobe |
|---|---|---:|---:|---:|---:|---|
| down | high | 50 | -656.34 | -0.56 % | 20.00 % | ok |
| down | low | 57 | 1811.25 | 1.60 % | 35.10 % | ok |
| sideways | high | 25 | 872.53 | 1.82 % | 24.00 % | zu klein |
| sideways | low | 30 | -369.84 | -0.66 % | 33.30 % | ok |
| unknown | unknown | 17 | 72.60 | 0.34 % | 29.40 % | zu klein |
| up | high | 35 | -97.59 | -0.22 % | 28.60 % | ok |
| up | low | 32 | -502.84 | -0.72 % | 25.00 % | ok |

**AgentGatedBreakout**

| Trend | Volatilität | Trades | Gewinn (USDT) | Ø Gewinn/Trade | Trefferquote | Stichprobe |
|---|---|---:|---:|---:|---:|---|
| down | high | 6 | -111.16 | -0.77 % | 33.30 % | zu klein |
| down | low | 14 | -359.55 | -1.04 % | 14.30 % | zu klein |
| sideways | high | 15 | 703.31 | 1.96 % | 33.30 % | zu klein |
| sideways | low | 15 | -124.66 | -0.33 % | 33.30 % | zu klein |
| unknown | unknown | 7 | 312.33 | 1.84 % | 42.90 % | zu klein |
| up | high | 19 | -573.02 | -1.21 % | 36.80 % | zu klein |
| up | low | 23 | -598.17 | -1.03 % | 30.40 % | zu klein |

**BreakoutDonchian**

| Trend | Volatilität | Trades | Gewinn (USDT) | Ø Gewinn/Trade | Trefferquote | Stichprobe |
|---|---|---:|---:|---:|---:|---|
| down | high | 24 | 202.75 | 0.39 % | 33.30 % | zu klein |
| down | low | 29 | 550.65 | 0.76 % | 24.10 % | zu klein |
| sideways | high | 21 | 545.33 | 1.07 % | 28.60 % | zu klein |
| sideways | low | 23 | -250.81 | -0.41 % | 30.40 % | zu klein |
| unknown | unknown | 10 | 744.32 | 3.09 % | 50.00 % | zu klein |
| up | high | 24 | -213.01 | -0.32 % | 41.70 % | zu klein |
| up | low | 26 | -708.50 | -1.02 % | 30.80 % | zu klein |

Hinweise: Kennzahlen mit weniger als 30 Trades sind statistisch kaum belastbar. Vergangene Ergebnisse garantieren keine zukünftigen Gewinne.
