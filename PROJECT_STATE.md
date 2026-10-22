# Projektstand

**Stand:** 22.10.2026 · **Phase:** 1 (Fundament) umgesetzt, Abschluss mit echten Daten auf dem Zielrechner ausstehend
**Handelsmodus:** keiner – Paper- und Live-Trading gesperrt

## Umgesetzt

- Framework-Entscheidung: Freqtrade 2026.9 als Engine, Docker als Laufzeit (siehe docs/ARCHITECTURE.md, ADR-001…005).
- Paket `kitb` mit CLI: `check-safety`, `probe`, `download`, `synthetic`, `validate`, `backtest`, `runs`.
- Datenabruf BTC/USDT, ETH/USDT (1h, 1d) über `freqtrade download-data`; Entfernen unvollständiger letzter Kerzen nach Download.
- Datenvalidierung (15 Prüfungen) mit Fehler/Warnung-Stufen, Datensatz-Fingerprint.
- Speicherung: Parquet (Freqtrade-Format) + SQLite-Protokoll (`source_events`, `validation_reports`, `backtest_runs`).
- Baseline `BaselineEmaCross` (EMA 21/55, Stoploss −10 %, nur Long/Spot).
- Backtest mit effektiver Gebühr 0,17 % je Seite (0,10 % Gebühr + 0,05 % Slippage + 0,02 % halber Spread),
  Buy-and-Hold-Benchmark, Exposure, Durchschnittsgewinn/-verlust, Kosten, Warnung bei < 30 Trades.
- Synthetische, gekennzeichnete Offline-Daten + statische Marktdefinitionen für Offline-Backtests.
- Sicherheitssperren und Doku (docs/SECURITY.md).

## Testergebnisse (nur in der Linux-Cloud-Umgebung ausgeführt)

| Umgebung | Ergebnis |
|---|---|
| venv, Python 3.13.16, Pakete gemäß `constraints.txt` | 56 bestanden, 0 fehlgeschlagen |
| Docker-Image `kitb:phase1` (Basis `freqtradeorg/freqtrade:2026.9`, Python 3.14.7) | 56 bestanden, 0 fehlgeschlagen |
| Netzwerktest `pytest -m network` (Binance) | **fehlgeschlagen** – Cloud-Proxy sperrt Börsen-Hosts (HTTP 403) |
| Windows 11 / WSL2 / Docker Desktop | **nicht getestet** |

Reproduzierbarkeit: Der Baseline-Backtest auf synthetischen Daten (Seed 42, 2023–2024) ergab in venv und Docker
identische Werte (286 Trades, −48,89 %). Diese Zahlen sind ein **Funktionstest**, keine Aussage über echte Märkte.

## Erreichte Datenquellen (aus der Cloud, 22.10.2026)

| Quelle | Ergebnis |
|---|---|
| api.binance.com, data-api.binance.vision, data.binance.vision | gesperrt (Proxy 403) |
| api.kraken.com, okx.com, api.bybit.com, bitstamp.net, api.exchange.coinbase.com, api.coingecko.com | gesperrt (Proxy 403) |
| pypi.org, Docker Hub, raw.githubusercontent.com | erreichbar |

→ Es wurden **keine echten Kursdaten** abgerufen. Alle Backtests liefen auf synthetischen Daten.

## Offene Punkte / Abhängigkeiten

1. Auf Windows 11: Docker-Build, Tests, `kitb probe`, echter Download + Validierung + Backtest ausführen.
2. Börsenwahl nach `probe` bestätigen (Binance-Verfügbarkeit am Standort: Weiß ich nicht).
3. Gebührenannahme 0,10 % beim gewählten Anbieter prüfen; Slippage-/Spread-Werte sind Schätzungen.
4. Docker-Image-Python (3.14) und Cloud-venv (3.13) unterscheiden sich; maßgeblich ist das Docker-Image.
5. Freqtrade meldet Pandas-Deprecation-Warnungen (Pandas4Warning) – harmlos, aber bei künftigen Updates beobachten.
6. Kein Git-Remote eingerichtet: Ziel-Repository muss festgelegt werden.

## Nächster konkreter Schritt

Phase 1 auf dem Zielrechner mit echten BTC/ETH-Daten abschließen (ROADMAP.md → „Vorgeschlagener nächster Schritt“),
danach Freigabe für Phase 2 einholen.
