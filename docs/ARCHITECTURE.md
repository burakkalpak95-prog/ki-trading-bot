# Architektur und Entscheidungen

## Komponenten und Schnittstellen (Zielbild, Phase 1 fett)

| # | Komponente | Phase | Umsetzung / Schnittstelle |
|---|---|---|---|
| 1 | **Market Data Layer** | 1 | `freqtrade download-data` (CCXT, Rate-Limit + Retries von Freqtrade); `kitb probe` prüft öffentliche Ticker/Kerzen mit eigenem Backoff |
| 2 | **Data Validation Layer** | 1 | `kitb.data.validate.validate_ohlcv(df, pair, timeframe, source, now=…) -> ValidationReport` |
| 3 | **Historical Data Storage** | 1 | Parquet über Freqtrade-Datahandler (`kitb.data.storage`), Protokoll in SQLite (`kitb.registry.Registry`) |
| 4 | Feature Engineering | 2/3 | Freqtrade-Strategien (`populate_indicators`) bzw. FreqAI-Features |
| 5 | Strategy Engine | 1 (Baseline) / 2 | Freqtrade `IStrategy`; Baseline `BaselineEmaCross` |
| 6 | Machine Learning Layer | 3 | geplant: FreqAI (Bot A: einfache Modelle, Bot B: erweitert) |
| 7 | Decision Logging | 4 | geplant |
| 8 | Learning & Evaluation | 4 | geplant |
| 9 | **Backtesting Engine** | 1 | Freqtrade-Backtesting, gekapselt in `kitb.backtest.run_backtest()` |
| 10 | Paper-Trading Engine | 7 | geplant: Freqtrade dry-run – **in Phase 1 gesperrt** |
| 11 | Risk Management | 2+ | geplant; Phase 1 nur Konfigurationssperren (`kitb.safety`) |
| 12 | Multi-Agent Coordination | 5 | nur bei nachgewiesenem Nutzen |
| 13 | Monitoring & Dashboard | 6 | geplant (OpenTerminal prüfen, sonst schlanke Alternative) |
| 14 | Live-Trading-Adapter | 8 | nicht implementiert; nur nach separater Freigabe |

Datenfluss Phase 1:

```
Börse (öffentlich) --freqtrade download-data--> Parquet --validate--> SQLite (validation_reports)
                                                   |
synthetic (offline, gekennzeichnet) ---------------+
                                                   v
                       kitb backtest: Safety-Check -> Validierung (Abbruch bei ERROR)
                       -> Freqtrade-Backtest (--fee = Gebühr+Slippage+½Spread)
                       -> Kennzahlen + Buy&Hold -> user_data/backtest_results/*.zip + SQLite (backtest_runs)
```

## ADR-001: Freqtrade als Kern statt Eigenentwicklung

- **Geprüft:** Freqtrade 2026.9 (PyPI und Docker Hub, am 22.10.2026 erreichbar; Image-Tag `2026.9` vom 29.09.2026),
  Lizenz GPLv3, Python ≥ 3.11 (Paket-Metadaten). Deckt Datendownload, Backtesting, Dry-Run (später) und mit FreqAI
  ML-Integration ab. Installation und Backtest wurden in dieser Umgebung tatsächlich ausgeführt.
- **Alternativen:** Eigene Engine (verworfen – Prinzip „keine eigene Handels-Engine“), vectorbt/backtrader
  (nicht im Detail geprüft; kein Bedarf, da Freqtrade Backtest *und* späteres Paper-Trading aus einer Hand liefert).
- **Konsequenz:** Eigener Code nur für Lücken: Validierung, Sicherheitssperren, Kostenmodell, Protokoll, Offline-Tests.
- **Lizenzhinweis:** GPLv3 ist für private Nutzung unproblematisch. Bei Weitergabe des Systems gelten die GPL-Pflichten (Unsicher, keine Rechtsberatung).

## ADR-002: Docker (WSL2-Backend) als Laufzeit unter Windows 11

- Freqtrade-Doku (Installationsseite, Stand `stable`): Windows-Nutzern wird **„strongly recommended“ Docker**, alternativ WSL2;
  native Installation nur mit `setup.ps1` und ggf. Visual-C++-Build-Tools.
- Entscheidung: `docker compose` mit gepinntem Image `freqtradeorg/freqtrade:2026.9` (Python 3.14.7, CCXT 4.5.84).
  Lokale venvs nutzen `constraints.txt` (aus dem Image erzeugt) für identische Paketversionen.
- GPU: in Phase 1 nicht benötigt. Für spätere Sequenzmodelle muss die CUDA-Unterstützung der RTX 5070 Ti in Docker/WSL2
  separat geprüft werden (Unsicher; erst in Phase 3 relevant).

## ADR-003: Datenquelle Binance (öffentliche API), Fallbacks OKX/Kraken

- BTC/USDT und ETH/USDT haben bei Binance hohe Liquidität; Freqtrade unterstützt Binance offiziell (Log-Ausgabe von Freqtrade).
- **Nicht verifiziert:** Erreichbarkeit vom Standort des Nutzers. Aus der Cloud-Umgebung waren *alle* Börsen-APIs gesperrt
  (Proxy 403). Deshalb `kitb probe binance okx kraken` auf dem Zielrechner ausführen und die erreichbare Börse wählen.
- Es werden nur öffentliche Marktdaten genutzt – kein Konto, keine API-Keys.

## ADR-004: Kostenmodell über effektive Gebühr

Freqtrade füllt Backtest-Orders ohne Slippage zum angefragten Preis (Freqtrade-Doku „Assumptions made by backtesting“).
Deshalb: `effektive Gebühr je Seite = Gebühr (0,10 %) + Slippage (0,05 %) + halber Spread (0,02 %) = 0,17 %`,
übergeben per `--fee` (Freqtrade wendet sie bei Ein- und Ausstieg an). Werte in `config/project.json`.
Die 0,10 % Gebühr ist eine Annahme für eine Standard-Spotgebühr und muss beim gewählten Anbieter geprüft werden (Unsicher).

## ADR-005: Offline-Modus nur für synthetische Daten

Freqtrade lädt auch im Backtest die Marktliste der Börse. `kitb.offline.offline_markets()` ersetzt das Laden durch feste
Spot-Marktdefinitionen (Präzision/Mindestgrößen angenommen). Erlaubt **nur**, wenn der Datenordner als synthetisch
gekennzeichnet ist (`SYNTHETIC_DATA_README.txt`) – sonst Abbruch.

## Reproduzierbarkeit

- Exakte Versionen: Docker-Image-Tag + `constraints.txt`.
- Jeder Backtest speichert SHA-256 der Daten je Paar, SHA-256 der Konfiguration, Kostenmodell und Freqtrade-Version.
- Synthetische Daten sind seed-basiert deterministisch.
- Geprüft: gleicher Backtest im venv (Python 3.13) und im Docker-Image (Python 3.14) lieferte identische Kennzahlen.
