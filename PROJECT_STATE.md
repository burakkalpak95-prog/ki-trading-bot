# Projektstand

**Stand:** 22.10.2026 · **Phasen:** 1–8 umgesetzt (Code + Tests); **echte Daten auf dem Zielrechner ausgewertet**
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

## Testergebnisse

| Umgebung | Ergebnis |
|---|---|
| venv, Python 3.13, Pakete gemäß `constraints.txt` | 154 bestanden, 0 fehlgeschlagen (nach Korrektur eines veralteten Tests, s. u.) |
| Docker-Image `kitb:local` (Basis `freqtradeorg/freqtrade:2026.9_freqai`, Python 3.14.7) | 154 bestanden, 0 fehlgeschlagen |
| Netzwerktest `pytest -m network` (Binance) | Cloud: fehlgeschlagen (Proxy 403) · **Windows: 1 bestanden** |
| Paper-Dry-Run (Rauchtest) | Konfiguration, Strategie, Modell, DB-Anlage und geordnetes Beenden funktionieren; **keine Marktdaten** (Börse gesperrt) |
| Dashboard | Startet; Reiter „Kurse & Signale“ und „Bot A vs. Bot B“ per Browser-Screenshot geprüft, Datenfunktionen per Tests |
| Windows 11 Pro, Docker Desktop 4.96.3 (Engine 29.9.2, Compose v5.6.1), Projekt unter `C:\Projekte\ki-trading-bot` | Build ok, `check-safety` ok, **154 bestanden, 0 fehlgeschlagen** (58 s); keine Rechteprobleme mit `user_data` |

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

## Ergebnisse auf echten Daten (Binance, Windows-Rechner, 22.10.2026)

Daten: BTC/USDT und ETH/USDT, je 42 140 Stundenkerzen und 1 755 Tageskerzen, 01.01.2022 – 22.10.2026 20:00 UTC.
Validierung: OK; einziger Befund eine fehlende 1h-Kerze nach 24.03.2023 12:00 UTC (Volumen 0; vermutlich Binance-Ausfall – Unsicher).

Baseline-Backtest 01.01.2023 – heute: 608 Trades, **+1,94 %**, Max DD 29,4 %, Profit Factor 1,01, p-Wert 0,95
(nicht von Zufall unterscheidbar). Buy&Hold gleichgewichtet im selben Zeitraum: **+244 %**. Kosten inkl. Slippage 4 783 USDT.

Validation-Abschnitt 19.11.2024 – 05.11.2025 (Test-Abschnitt 05.11.2025 – 22.10.2026 **unberührt**); Buy&Hold nach Kosten **+6,97 %**:

| Strategie / Bot | Netto | Max DD | Profit Factor |
|---|---:|---:|---:|
| BaselineEmaCross | −13,77 % | 23,0 % | 0,73 |
| TrendEmaAdx | −20,53 % | 21,4 % | 0,31 |
| MomentumMacdRsi | −17,25 % | 20,5 % | 0,61 |
| MeanReversionBollinger | −3,22 % | 4,4 % | 0,35 (nur 14 Trades) |
| BreakoutDonchian | −9,30 % | 16,0 % | 0,84 |
| VolatilitySqueeze | −13,41 % | 17,4 % | 0,67 |
| Bot A (LogReg) | −9,57 % | 20,3 % | 0,86 |
| Bot B (LightGBM) | −50,47 % | 50,8 % | 0,56 |
| AgentGatedBaseline | +5,27 % | 6,2 % | 1,28 |
| AgentGatedBreakout | +2,16 % | 8,2 % | 1,06 |

- Bot A und Bot B erfüllen die Freigabekriterien **nicht** (Modellversionen #1, #2). Bot B: 40,9 Prozentpunkte schlechter als
  Bot A bei 8,8-facher Rechenzeit (39 s vs. 4,5 s).
- `agents-evaluate` meldet für AgentGatedBreakout „Nutzen belegt“ (mehr Gewinn, weniger Drawdown, 66,7 % der Fenster besser),
  für AgentGatedBaseline nicht. **Einordnung:** Der Vorteil stammt vermutlich vor allem aus weniger Trades (weniger Kosten),
  beruht auf einem einzigen Validation-Abschnitt und kehrt sich im Train-Abschnitt um (AgentGatedBreakout −11 %). Die Agenten
  bleiben deshalb **aus** (`agents.enabled=false`); das Kriterium in `agents-evaluate` ist zu schwach und sollte auch den
  Train-Abschnitt verlangen.
- **Gesamtbild:** Keine Strategie und kein Bot schlägt Buy&Hold. Kein Modell wird freigegeben, Paper-Trading wird nicht gestartet.
- Auffälligkeit: BaselineEmaCross liefert im Train-Abschnitt je nach Lauf leicht verschiedene Werte (+17,87 % / 437 Trades vs.
  +14,61 % / 439 Trades). Vermutete Ursache: unterschiedliche Vorlaufdaten je nach Strategie-Kombination (Indikator-Aufwärmphase) – Unsicher, nicht geprüft.

**Gegenprobe OKX** (gleicher Zeitraum, eigene Daten, Validierung OK, 3 Stundenkerzen ohne Volumen in 2022): Ergebnisse fast
deckungsgleich (z. B. Baseline −13,83 %, Bot A −8,28 %, Bot B −56,16 %, AgentGatedBaseline +5,09 %); gleiche Verdicts.
Das zeigt Unabhängigkeit von der Börse, aber nicht von der Marktphase (BTC/ETH kosten auf beiden Börsen fast dasselbe).

Berichte (auf dem Windows-Rechner): `user_data/reports/compare-20261022-211012-2bf0b6`, `ml-compare-20261022-211134-292bda`,
`compare-20261022-211250-69b03e`.

## Runde 2 – Ergebnisse auf echten Daten (Binance, 22.10.2026, Commit d5e65ba)

Tests auf dem Windows-Rechner: 169 bestanden. Die Aufwärmphase ist jetzt einheitlich: BaselineEmaCross liefert in allen Läufen
identische Werte (Train 437 Trades, +17,87 %).

| | Train Netto / Max DD / Rendite/DD | Validation Netto / Max DD / Rendite/DD | Trades Val. |
|---|---|---|---:|
| Buy&Hold gleichgewichtet | +41,02 % / 69,61 % / 0,59 | +6,97 % / 48,44 % / 0,14 | – |
| BaselineEmaCross (1h) | +17,87 % / 19,09 % / 0,94 | −13,77 % / 23,01 % / −0,60 | 163 |
| BreakoutDonchian (1h) | −16,07 % / 28,27 % / −0,57 | −9,30 % / 16,01 % / −0,58 | – |
| AgentGatedBaseline (1h) | +4,33 % / 10,76 % / 0,40 | +5,27 % / 6,23 % / 0,85 | – |
| DailyTrendSma (1d) | +57,35 % / 1,54 % / 37,29 (14 Trades) | −1,01 % / 7,83 % / −0,13 | 14 |
| DailyMomentum (1d) | +46,55 % / 6,30 % / 7,39 (47 Trades) | +5,53 % / 7,81 % / 0,71 | 9 |

Übrige 1h-Strategien in Train und Validation negativ. Fenster (Anteil positiv / besser als Buy&Hold): DailyMomentum 53,8 % / 38,5 %,
DailyTrendSma 53,8 % / 46,2 %. Train der Tagesstrategien beginnt wegen 200 Tagen Vorlauf erst Mitte 2022.

- **Agenten:** Mit dem schärferen Kriterium kein belegter Nutzen mehr (AgentGatedBaseline schadet im Train, AgentGatedBreakout
  nur in 53 % der Fenster besser) → bleiben aus.
- **Tagesstrategien:** risikobereinigt besser als Buy&Hold, in der Validation aber nur 9–14 Trades → **nicht belastbar**.
  Der Test-Abschnitt wird dafür noch nicht geöffnet.
- **Nächste Prüfung (vorab festgelegt):** unabhängige Vorperiode 2019–2021 (Daten 2018–2021, bisher nie angesehen, eigener
  Ordner). Kandidat gilt als robust, wenn dort gilt: Gewinn > 0, Profit Factor ≥ 1,2, Max DD ≤ 25 %, ≥ 15 Trades, und in
  mindestens 2 von 3 Einzeljahren Gewinn. Nur dann wird der Test-Abschnitt einmalig ausgewertet.

## Vorperiode 2019–2021 (Binance, Daten 2018–2021, eigener Ordner)

Buy&Hold gleichgewichtet: +1 915,8 %, Max DD 65,4 % (1d).

| Strategie | Trades | Netto | Max DD | PF | p-Wert | Jahre mit Gewinn |
|---|---:|---:|---:|---:|---:|---|
| DailyMomentum (1d) | 48 | +231,65 % | 19,74 % | 4,85 | 0,125 | 3 von 3 (2019 +37 %, 2020 +46 %, 2021 +47 %) |
| DailyTrendSma (1d) | 25 | +256,40 % | 6,12 % | 8,00 | 0,199 | 3 von 3 (2019 +13 %, 2020 +88 %, 2021 +38 %) |
| BaselineEmaCross (1h, Vergleich) | 407 | +179,86 % | 10,03 % | 1,81 | 0,002 | – |

**Vorab festgelegte Regel: beide Tagesstrategien bestehen** (Gewinn > 0, PF ≥ 1,2, DD ≤ 25 %, ≥ 15 Trades, ≥ 2 von 3 Jahren
positiv). Einschränkungen: p-Werte 0,12–0,20 (kein statistischer Beleg), netto weit hinter Buy&Hold; der Vorteil liegt nur im
Verhältnis Rendite/Drawdown. 1h-Daten 2018 mit Lücken (nur Vorlauf).

**Nächster Schritt: einmalige Auswertung des Test-Abschnitts** (05.11.2025 – 22.10.2026) für DailyTrendSma und DailyMomentum.
Vorab festgelegte Regel für den Test-Abschnitt: Gewinn > 0, Max DD ≤ 25 % und Rendite/DD höher als bei Buy&Hold im selben
Abschnitt. Besteht eine Strategie, wird sie Kandidat für Paper-Trading (eigene Freigabe nötig); sonst wird keine Variante
nachträglich angepasst und erneut am Test-Abschnitt geprüft.

## Test-Abschnitt (einmalig ausgewertet, 05.11.2025 – 22.10.2026)

| | Trades | Netto | Max DD | Rendite/DD | PF |
|---|---:|---:|---:|---:|---:|
| DailyTrendSma | 5 | +3,45 % | 1,93 % | 1,79 | 2,79 |
| DailyMomentum | 10 | −5,49 % | 8,03 % | −0,68 | 0,32 |
| Buy&Hold gleichgewichtet | – | −21,21 % | 50,38 % | −0,42 | – |

Train und Validation unverändert gegenüber dem Lauf vor der Test-Auswertung. Vorab festgelegte Regel: **DailyTrendSma besteht**
(nur 5 Trades – kein statistischer Beleg), **DailyMomentum fällt durch** (Verlust) und wird nicht weiterverfolgt. Der
Test-Abschnitt ist damit verbraucht; künftige Varianten brauchen neue, ungesehene Daten (z. B. ab jetzt Paper-Trading).

**Paper-Trading für Regel-Strategien** (ADR-015): `paper_strategies.trend_d1` = DailyTrendSma, Start nur nach namentlicher
Freigabe (`kitb strategies approve`), eigenes Compose-Profil `paper-trend`. Tests (Cloud-venv): 171 bestanden.

## Erreichte Datenquellen (22.10.2026)

| Quelle | Ergebnis |
|---|---|
| Binance, OKX, Kraken – vom Windows-Rechner | **erreichbar** (BTC/USDT und ETH/USDT verfügbar) |
| Binance, OKX, Kraken, Bybit, Bitstamp, Coinbase, CoinGecko – aus der Cloud | gesperrt (Proxy 403) |
| pypi.org, Docker Hub, raw.githubusercontent.com | erreichbar |

→ Echte Daten wurden auf dem Windows-Rechner von Binance geladen.

## Kosten

0 €. Nur Open-Source-Software und öffentliche Endpunkte; keine bezahlten Dienste genutzt.

## Offene Punkte / bekannte Probleme

1. **Kein Vorteil gegenüber Buy&Hold** auf echten Daten – weder Regel-Strategien noch KI-Bots. Ohne neue Idee (andere Features,
   Labels, Zeitintervalle oder Strategieform) ist Paper-Trading derzeit nicht sinnvoll.
2. `agents-evaluate`: Kriterium verschärft (Runde 2); Ergebnis auf echten Daten ausstehend.
3. Leicht unterschiedliche Train-Ergebnisse je nach Lauf: Ursache gefunden (Freqtrade-Vorlauf = Maximum der Strategien), in Runde 2 behoben.
4. Gebühr 0,10 % bei Binance für das eigene Konto prüfen; Slippage/Spread sind Schätzungen.
5. Repository `burakkalpak95-prog/ki-trading-bot` ist öffentlich; Umstellung auf privat empfohlen.
6. Auf Windows läuft das Projekt ohne Ubuntu direkt mit Docker Desktop aus `C:\Projekte\ki-trading-bot` (Bind-Mount langsamer als
   in WSL, für die aktuellen Datenmengen ausreichend).
7. Freqtrade-Fehler im Ersatzpfad nach fehlgeschlagenem FreqAI-Training (`DataFrame` ohne `.dtype`); abgefangen, nicht behoben.
8. GPU-Nutzung nicht geprüft (für die aktuellen Modelle nicht nötig).
9. Pandas-Deprecation-Warnungen aus Freqtrade – harmlos, bei Updates beobachten.

## Nächster konkreter Schritt

Entscheiden, ob die Strategie-/Modellentwicklung weitergeht (neue Hypothesen, nur auf Train/Validation prüfen) oder das
System vorerst als Analysewerkzeug genutzt wird. Der Test-Abschnitt (05.11.2025 – 22.10.2026) bleibt gesperrt, bis ein
Kandidat die Validation überzeugend besteht.
