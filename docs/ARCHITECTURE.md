# Architektur und Entscheidungen

## Komponenten und Schnittstellen (Stand Phase 8)

| # | Komponente | Phase | Umsetzung / Schnittstelle |
|---|---|---|---|
| 1 | Market Data Layer | 1 | `freqtrade download-data` (CCXT, Rate-Limit + Retries von Freqtrade); `kitb probe` mit eigenem Backoff |
| 2 | Data Validation Layer | 1 | `kitb.data.validate.validate_ohlcv(...) -> ValidationReport` (15 Prüfungen) |
| 3 | Historical Data Storage | 1 | Parquet über Freqtrade-Datahandler, Protokoll in SQLite (`kitb.registry.Registry`) |
| 4 | Feature Engineering | 2/3 | `kitb.indicators` (kausal); FreqAI-Features in `BotAEfficient` / `BotBExtended` |
| 5 | Strategy Engine | 2 | Freqtrade `IStrategy`: 6 Regel-Strategien, alle mit `RiskManagedMixin` |
| 6 | Machine Learning Layer | 3 | FreqAI Walk-Forward; Bot A `KitbLogisticRegression`, Bot B `KitbLightGBMClassifier` |
| 7 | Decision Logging | 4 | `kitb.decisions` (Backtest: Freqtrade-Signal-Export; Paper: `confirm_trade_entry/exit`), Feature-Einfluss über `kitb.explain` |
| 8 | Learning & Evaluation | 4 | `kitb.models` (Register, Kriterien, Freigabe, Rollback), `kitb.drift`, `kitb train-cycle` |
| 9 | Backtesting Engine | 1/2 | `kitb.backtest.run_freqtrade_backtest()`, `kitb.compare`, `kitb.evaluation` |
| 10 | Paper-Trading Engine | 7 | `kitb.paper` startet Freqtrade-Dry-Run je Bot (eigene SQLite-DB) |
| 11 | Risk Management | 7 | `kitb.risk.RiskManager` + `RiskManagedMixin`, Limits in `config/risk.json`; Not-Stopp `kitb.killswitch` |
| 12 | Multi-Agent Coordination | 5 | `kitb.agents` (azyklisch, nur Veto), **standardmäßig aus** – kein belegter Nutzen |
| 13 | Monitoring & Dashboard | 6 | Streamlit (`kitb dashboard`, nur 127.0.0.1) |
| 14 | Live-Trading-Adapter | 8 | `kitb.live.LockedLiveAdapter` – lehnt jede Aktion ab |

Datenfluss:

```
Börse (öffentlich) --download-data--> Parquet --validate--> SQLite
                                         |
                                         v
  compare / ml-compare / train-cycle: Safety-Check -> Validierung -> chronologischer Split (Train/Val, Test gesperrt)
     -> Freqtrade-Backtest bzw. FreqAI Walk-Forward (--fee = Gebühr+Slippage+½Spread, RiskManagedMixin)
     -> Kennzahlen, Buy&Hold, Marktphasen, Fenster -> Bericht (user_data/reports) + SQLite
     -> Modellkandidat -> Kriterien geprüft -> [Mensch: approve/activate] -> aktive Version
                                                                              |
  paper start: Safety-Check (Modus paper) -> Freqtrade dry-run je Bot <-------+
     -> Entscheidungen + Trades (SQLite je Bot) -> Dashboard / paper report / drift
  Not-Stopp-Datei -> Bots stoppen sich selbst (bot_loop_start), neue Einstiege werden abgelehnt
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
- Entscheidung: `docker compose` mit gepinntem Image `freqtradeorg/freqtrade:2026.9` (Python 3.14.7, CCXT 4.5.84);
  seit Phase 3 die Variante `2026.9_freqai` (gleiche Freqtrade-Version plus FreqAI-Abhängigkeiten).
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

## ADR-006: Faire Auswertung – chronologischer Split, gesperrter Test-Abschnitt, Marktphasen (Phase 2)

- Daten werden chronologisch in Train 60 % / Validation 20 % / Test 20 % geteilt. Der Test-Abschnitt wird nur mit
  `--include-test` ausgewertet, jede Nutzung wird protokolliert (Schutz vor unbewusstem Overfitting).
- Zusätzlich rollierende 90-Tage-Fenster (Stabilität) und Marktphasen (Trend über SMA50 + Steigung, Volatilität über
  expandierenden Median). Die Phase eines Tages gilt erst ab Schluss der Tageskerze (kein Blick in die Zukunft).
- Ergebnisse mit < 30 Trades werden als „zu klein“ markiert.

## ADR-007: KI-Bots über FreqAI statt eigener Trainingsschleife (Phase 3)

- FreqAI liefert Walk-Forward-Training (90 Tage Training, Neutraining alle 30 Tage) innerhalb des Freqtrade-Backtests,
  dadurch gelten für Bots und Regel-Strategien identische Kosten, Daten und Ausführungsannahmen.
- Label: Rendite über 12 Kerzen > 0,6 %. Leck-Schutz: `shuffle=false`, Features nur aus vergangenen Kerzen.
- Bot A: logistische Regression, wenige Features, nur 1h. Bot B: LightGBM, mehr Features, 1h + 1d, BTC/ETH-Korrelation.
  Beide mit `class_weight=balanced` (ohne dies erreichte Bot A nie die Einstiegsschwelle und handelte nicht).
- Ressourcen werden je Lauf gemessen (Laufzeit, CPU, Spitzen-RAM, Modellgröße).
- GPU: nicht genutzt und nicht geprüft (Cloud ohne GPU). Bei den gemessenen Laufzeiten (Sekunden) ist sie für diese
  Modelle nicht nötig. CUDA-Unterstützung der RTX 5070 Ti in Docker/WSL2: Weiß ich nicht.
- Bekannter Freqtrade-Fehler: Schlägt ein Training fehl, bricht der Ersatzpfad für leere Vorhersagen ab
  (DataFrame hat kein `.dtype`). Gegenmaßnahme: Prüfung auf ausreichende Vorlaufdaten (`min_history_days_before_backtest`).

## ADR-008: Modell-Governance mit menschlicher Freigabe (Phase 4)

- Lebenszyklus: candidate -> passed/failed (automatisch nach `config/release_criteria.json`) -> approved -> active -> retired.
  `approve`, `activate` und `rollback` verlangen `--by <Name>`; fehlende Namen, bekannte automatische Akteure
  und Namen, die mit „bot“, „agent“ oder „model“ beginnen, werden abgelehnt.
- Kriterien u. a.: ≥ 30 Trades, Profit Factor ≥ 1,05, Drawdown ≤ 25 %, Gewinn > 0, p-Wert des mittleren Trade-Gewinns
  ≤ 0,05 (t-Test aus Freqtrade), besser als Buy&Hold, Referenz-Strategie und aktive Version.
  Der p-Wert wurde ergänzt, nachdem Bot A auf Zufallsdaten die übrigen Kriterien zufällig erfüllte.
- Drift (PSI + KS-Test, Leistungsvergleich) **meldet nur**; kein automatisches Neutrainieren oder Umschalten.

## ADR-009: Agenten nur als abschaltbares Veto (Phase 5)

- Regelbasierte, lokale Agenten (Markt, Trend, Volatilität, Risiko) mit Koordinator; Abhängigkeiten werden topologisch
  sortiert, Zyklen führen zum Abbruch. Agenten können Einstiege nur verhindern, nie auslösen.
- `kitb agents-evaluate` vergleicht Strategien mit und ohne Agenten. Ergebnis auf synthetischen Daten: kein belegter
  Nutzen -> `agents.enabled=false`. Einschalten nur durch einen Menschen, wenn die Auswertung auf echten Daten einen Nutzen zeigt.

## ADR-010: Dashboard mit Streamlit; OpenTerminal verworfen (Phase 6)

- Geprüft (PyPI, 22.10.2026): Das Paket `open-terminal` (0.14.2, MIT) ist ein Fernzugriffs-Terminal mit REST-API für
  KI-Agenten, **kein Marktdaten-Terminal**. Es würde eine Shell über das Netzwerk freigeben – widerspricht der Regel
  „keine unnötigen offenen Ports“. Verworfen.
- Falls „OpenTerminal“ das OpenBB-Projekt meinte: `openbb` 5.0.0 (Apache-2.0) existiert auf PyPI, wurde aber nicht
  installiert oder getestet (Unsicher, ob es gemeint war). Für die Anforderungen genügt eine schlanke Lösung.
- Entscheidung: Streamlit 1.66.1 (Apache-2.0) + Plotly, liest nur Dateien/SQLite, bindet nur an 127.0.0.1, Telemetrie aus,
  Streamlit-Werkzeugleiste minimal. FreqUI wurde nicht genutzt, weil sie den Freqtrade-API-Server voraussetzt.
- Analysen und Tests funktionieren ohne Dashboard.

## ADR-011: Zentrales Risikomodul und Not-Stopp (Phase 7)

- `RiskManagedMixin` steckt in allen Strategien: Positionsgröße (max. 40 % je Position, 2 % Risiko je Trade bis zum
  Stoploss), Gesamtexposure ≤ 80 %, max. 2 Positionen, Tagesverlust ≤ 3 %, Drawdown-Stopp bei 15 %, veraltete Daten,
  Einsatz ≤ 1 % des 24h-Volumens. Hebel, Futures, Margin, Abhebungen: immer aus.
- Im Backtest wird der Drawdown-Stopp nur gemeldet (`risk_halt_would_trigger_at`), sonst würde ein einmaliger Stopp
  den Rest des Backtests verfälschen. Im Paper-Betrieb wird er durchgesetzt.
- Not-Stopp: Datei `user_data/KILL_SWITCH`. Laufende Bots lehnen neue Einstiege ab und beenden sich; Zurücksetzen nur
  durch einen Menschen (`kitb kill-switch off --by <Name>`).

## ADR-012: Paper-Trading = Freqtrade-Dry-Run (Phase 7)

- Je Bot ein eigener Dry-Run mit eigener SQLite-DB und virtuellem Kapital (10 000 USDT). Start nur mit aktiver,
  freigegebener Modellversion; Sicherheitsprüfung im Modus `paper` (dry_run=true erzwungen, keine Keys).
- Freqtrade simuliert Markt-Orders im Dry-Run anhand des Orderbuchs und rechnet die konfigurierte Gebühr (im Code von
  Freqtrade 2026.9 nachgesehen). Kein API-Server, keine offenen Ports.

## ADR-013: Live-Trading nur als gesperrte Schnittstelle (Phase 8)

- `LiveTradingAdapter` (Protokoll) beschreibt die spätere Schnittstelle; einzige Implementierung `LockedLiveAdapter`
  lehnt jede Aktion ab. `safety` lässt den Modus `live` nie zu.
- `kitb live readiness` zeigt die Checkliste für eine spätere Freigabe (u. a. mind. 100 Paper-Trades, bestätigte
  Risikolimits, Mindestordergrößen, unabhängige Prüfung). Eine Freischaltung erfordert Codeänderung + separate Freigabe.

## ADR-014: Fairere Vergleiche nach den ersten Ergebnissen mit echten Daten (22.10.2026)

- **Einheitliche Aufwärmphase:** Freqtrade lädt bei `--strategy-list` den Vorlauf für das Maximum aller Strategien
  (`required_startup`, im Code von Freqtrade 2026.9 nachgesehen). EMAs hängen vom Startpunkt ab, deshalb lieferte dieselbe
  Strategie je nach Zusammenstellung eines Laufs leicht andere Train-Werte (+17,87 % vs. +14,61 %). Alle 1h-Regel-Strategien
  nutzen jetzt `STARTUP_CANDLES_1H = 520`, Tagesstrategien `STARTUP_CANDLES_1D = 200`. Rollierende Fenster beginnen erst
  nach diesem Vorlauf.
- **Risikobereinigter Vergleich:** Buy&Hold ist voll investiert, die Strategien wegen der Risikolimits (2 % Risiko je Trade
  bei −10 % Stoploss → höchstens 20 % je Position) nur teilweise. Der Bericht zeigt deshalb zusätzlich den maximalen
  Drawdown von Buy&Hold und das Verhältnis Rendite/Drawdown. Die Freigabekriterien bleiben unverändert.
- **Agenten-Kriterium:** Ein Vorteil muss auch im Train-Abschnitt bestehen (`require_train_not_worse`); auf echten Daten
  hatte der Filter nur im Validation-Abschnitt geholfen.
- **Tagesstrategien:** `DailyTrendSma` (Schluss über SMA 200) und `DailyMomentum` (Schluss über dem von vor 90 Tagen) mit
  festen, vorab gewählten Parametern – bewusst keine Optimierung, um Overfitting auf den Validation-Abschnitt zu vermeiden.
  Weil die Daten am 01.01.2022 beginnen, startet ihr Train-Abschnitt erst nach 200 Tagen Vorlauf.

## ADR-015: Paper-Trading auch für Regel-Strategien, mit menschlicher Freigabe

- Anlass: DailyTrendSma bestand die vorab festgelegten Prüfungen (Vorperiode 2019–2021, Test-Abschnitt); KI-Bots nicht.
- Regel-Strategien haben keine Modellversion. Stattdessen Tabelle `strategy_approvals` (approved/revoked, Name + Begründung,
  nur Menschen). Eine Freigabe gilt nur für die konfigurierte Strategie; wird `paper_strategies` geändert, ist sie ungültig.
- Paper-Konfiguration ohne `timeframe` (kommt aus der Strategie, hier 1d) und ohne FreqAI; sonst identisch zu den Bots
  (dry_run, keine Keys, eigene DB, Risikomodul, Not-Stopp).
- Eigenes Compose-Profil `paper-trend`, damit nicht versehentlich die KI-Bots ohne Freigabe in eine Neustart-Schleife geraten.


## ADR-016: Lernende Paper-Flotte mit KI-Gedächtnis (Freigabe Burak, 22.10.2026)

- Anlass: Burak will, dass die KI im Paper-Trading laufend lernt, statt monatelang nur eine träge Regel zu beobachten.
- **Lernmodus** (`bots`, Freigabe `kitb strategies approve --id bot_a|bot_b|fleet`): KI-Bots laufen im Dry-Run ohne
  freigegebene Modellversion; FreqAI trainiert alle 24 h mit den neuesten Kursen neu (`paper_learning.retrain_hours`).
- **Datensammler-Flotte** (`paper_strategies` mit `mode: explore`): 7 Strategien unterschiedlicher Art (Trend, Momentum,
  Mean-Reversion, Ausbruch, Volatilität; 1h und 1d) plus 7 Kopien mit KI-Filter (`ai_filter: true`). Die Kopien ohne Filter
  sind die Kontrollgruppe – nur der Vergleich zeigt, ob der Filter hilft (Schutz gegen Selbstbestätigung).
- **Runden:** Erreicht ein Lern-/Sammler-Bot sein Drawdown-Limit, löst er nicht mehr den globalen Not-Stopp aus, sondern
  beendet seine Runde; `kitb paper run` archiviert die DB (`paper/<id>/rounds/`) und startet mit gleichem Startkapital
  (10.000 USDT) neu. Für `candidate` (trend_d1) und alles Weitere gilt der globale Not-Stopp unverändert.
- **KI-Gedächtnis** (`kitb.ai_memory`, Dienst `ai-memory`): täglich alle abgeschlossenen Trades aller Bots und Runden
  (+ einmalig historische Backtest-Trades als Startwissen) mit Marktlage beim Einstieg (nur abgeschlossene Tageskerzen
  davor). Klassifikator „Trade lohnt sich?“ wird auf den älteren 80 % trainiert und auf den jüngsten 20 % gemessen
  (Trefferquote, AUC, Gewinn der erlaubten Trades); danach auf allem neu trainiert und von den Filter-Bots genutzt.
  Der Filter kann nur ablehnen, nie ein Risikolimit aufheben. Versionen in `ai_filter_versions`.
- Seite „Was hat die KI gelernt?“: Trefferquote heute vs. Version 1 (%-Punkte), Vorsprung vor Raten, Filter-Vorteil je
  Trade, Gedächtnisgröße, wichtigste Merkmale, Ampel. Alles nur virtuelles Kapital; Live bleibt gesperrt.
- **Breit lernen, eng handeln** (Freigabe Burak, 22.10.2026): `learning_pairs` (BTC, ETH + SOL, XRP, BNB, ADA, DOGE, LINK)
  gelten nur für Paper-Bots im Modus explore/learning und für Backtests des KI-Gedächtnisses. Kandidaten (trend_d1) und
  alles darüber hinaus bleiben bei `allowed_pairs` (BTC/ETH); die Sicherheitsprüfung unterscheidet das je Modus.
