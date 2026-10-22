# Projektstand

**Stand:** 22.10.2026 · **Phasen:** 1–8 umgesetzt (Code + Tests), **nur mit synthetischen Daten geprüft**
**Handelsmodus:** keiner aktiv · Paper-Trading vorbereitet (Dry-Run) · Live-Trading gesperrt

## Umgesetzt je Phase

| Phase | Ergebnis | Commit |
|---|---|---|
| 1 | Freqtrade 2026.9 in Docker, Datenabruf, Validierung (15 Prüfungen), Parquet + SQLite, Baseline-Backtest mit Kostenmodell | 309ee15 |
| 2 | 6 Regel-Strategien (EMA-Kreuzung, Trend EMA+ADX, Momentum MACD+RSI, Mean Reversion Bollinger, Breakout Donchian, Volatility Squeeze), kausales Indikator-Modul, Split 60/20/20 mit gesperrtem Test-Abschnitt, 90-Tage-Fenster, Marktphasen, Vergleichsbericht | 0f2128b |
| 3 | Bot A (logistische Regression) und Bot B (LightGBM) über FreqAI Walk-Forward; Laufzeit/CPU/RAM-Messung | 0018069 |
| 4 | Modellregister mit Kriterien, menschlicher Freigabe, Aktivierung, Rollback; Entscheidungslog mit Top-Features je Vorhersage; Drift (PSI, KS-Test, Leistung) | 29fb691 |
| 5 | Regelbasierte Veto-Agenten mit Koordinator und Nutzenauswertung; standardmäßig aus | 58a04ed |
| 6 | Streamlit-Dashboard (10 Reiter, nur 127.0.0.1), Not-Stopp-Schalter; OpenTerminal geprüft und verworfen | 6132cc9 |
| 7 | Zentrales Risikomodul in allen Strategien; Paper-Trading beider Bots als Freqtrade-Dry-Run; Not-Stopp | df581a1 |
| 8 | Gesperrter Live-Adapter, Bereitschafts-Checkliste, Bestätigung der Risikolimits | 8fa0982 |
| Doku | README, ROADMAP, Architektur (ADR-006…013), Sicherheit, Windows-Anleitung auf Stand Phase 8; Test-Korrektur | siehe Git-Log |

## Testergebnisse (nur Linux-Cloud-Umgebung)

| Umgebung | Ergebnis |
|---|---|
| venv, Python 3.13, Pakete gemäß `constraints.txt` | 154 bestanden, 0 fehlgeschlagen (nach Korrektur eines veralteten Tests, s. u.) |
| Docker-Image `kitb:local` (Basis `freqtradeorg/freqtrade:2026.9_freqai`, Python 3.14.7) | 154 bestanden, 0 fehlgeschlagen |
| Netzwerktest `pytest -m network` (Binance) | **fehlgeschlagen** – Cloud-Proxy sperrt Börsen-Hosts (HTTP 403) |
| Paper-Dry-Run (Rauchtest) | Konfiguration, Strategie, Modell, DB-Anlage und geordnetes Beenden funktionieren; **keine Marktdaten** (Börse gesperrt) |
| Dashboard | Startet; Reiter „Kurse & Signale“ und „Bot A vs. Bot B“ per Browser-Screenshot geprüft, Datenfunktionen per Tests |
| Windows 11 / WSL2 / Docker Desktop | **nicht getestet** |

Korrigierter Test: `test_cli_has_no_trading_command` verbot noch den Unterbefehl `live`, der seit Phase 8 existiert
(nur `readiness`/`confirm-risk`). Der Test prüft jetzt genau diese beiden erlaubten Aktionen.

## Ergebnisse auf synthetischen Daten (Funktionstest, keine Marktaussage)

Validation-Abschnitt 15.03.–08.08.2024, Seed 42, inkl. Risikomodul und 0,17 % Kosten je Seite; Buy&Hold −1,57 %.

| | Trades | Netto | Max DD | Freigabe |
|---|---:|---:|---:|---|
| BreakoutDonchian (beste Regel-Strategie) | 44 | +4,97 % | 5,55 % | – |
| BaselineEmaCross | 70 | −6,36 % | 9,43 % | – |
| Bot A (LogReg) | 55 | +0,48 % | 15,40 % | nicht erfüllt (Profit Factor, p-Wert) |
| Bot B (LightGBM) | 234 | −16,69 % | 20,08 % | nicht erfüllt (5 Kriterien) |

- Bot B brauchte ≈ 12,8× so lange wie Bot A (34,8 s vs. 2,7 s für 10 Trainings) und brachte keinen Mehrwert.
- Agenten: kein belegter Nutzen → bleiben deaktiviert.
- Auf Zufallsdaten ist „kein Modell besteht“ das erwartete Ergebnis; ein Bestehen wäre ein Hinweis auf ein Datenleck.

## Erreichte Datenquellen (aus der Cloud, 22.10.2026)

| Quelle | Ergebnis |
|---|---|
| Binance, OKX, Kraken, Bybit, Bitstamp, Coinbase, CoinGecko | gesperrt (Proxy 403) |
| pypi.org, Docker Hub, raw.githubusercontent.com | erreichbar |

→ Es wurden **keine echten Kursdaten** abgerufen.

## Kosten

0 €. Nur Open-Source-Software und öffentliche Endpunkte; keine bezahlten Dienste genutzt.

## Offene Punkte / bekannte Probleme

1. Ziel-Repository für den Push ist nicht festgelegt.
2. Alles auf Windows 11 ausführen: Build, Tests, `probe`, echte Daten, `compare`, `ml-compare`.
3. Börsenwahl nach `probe` (Binance-Verfügbarkeit am Standort: Weiß ich nicht); Gebühr 0,10 % beim Anbieter prüfen;
   Slippage/Spread sind Schätzungen.
4. Langzeit-Paper-Trading ist nur auf dem Zielrechner möglich; Startvoraussetzung ist eine freigegebene, aktive Modellversion.
5. Freqtrade-Fehler im Ersatzpfad nach fehlgeschlagenem FreqAI-Training (`DataFrame` ohne `.dtype`); abgefangen durch
   Prüfung der Vorlaufdaten, nicht behoben.
6. GPU-Nutzung nicht geprüft (für die aktuellen Modelle nicht nötig).
7. Docker-Python (3.14) und Cloud-venv (3.13) unterscheiden sich; maßgeblich ist das Docker-Image.
8. Pandas-Deprecation-Warnungen aus Freqtrade – harmlos, bei Updates beobachten.

## Nächster konkreter Schritt

Repository festlegen, dann auf dem Windows-Rechner Phase 1 mit echten Daten abschließen und `compare` / `ml-compare`
auf echten Daten ausführen (ROADMAP.md → „Nächste Schritte“).
