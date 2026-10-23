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
- **Experimente der KI** („die KI darf alles, die Bots haben feste Regeln“, Burak 22.10.2026): Die Regeln der Bots
  bleiben fest. Die KI probiert alle 4 Stunden (`paper_learning.memory_every_hours`) ~15 Lernmethoden (gradient-geboostete Bäume HistGradientBoosting/LightGBM/
  XGBoost mit zufälliger Hyperparameter-Suche, Random Forest, logistische Regression) × 3 Schwellen aus. Ausgewählt
  wird per Walk-Forward nur auf den älteren 80 % (3 Zeitabschnitte); die jüngsten 20 % bleiben für die ehrliche
  Messung unberührt. Eingesetzt wird ein Mittel der besten 3. Der Filter ist nur aktiv, wenn der Gewinner im Mittel UND
  in der Mehrheit der Abschnitte besser war als „alle Trades zulassen“; sonst lässt er alles durch. Begründung der
  Methodenwahl: Für tabellarische Daten dieser Größe gelten gradient-geboostete Bäume als starker Standard; Deep
  Learning/Reinforcement Learning würde bei wenigen tausend Trades eher Rauschen auswendig lernen (Unsicher: kein
  eigener Vergleich auf diesen Daten). Ergebnis jeder Runde in `ai_memory/experiments-latest.json`, sichtbar im
  Dashboard unter „Was die KI gerade macht“ (inkl. jüngster Filter-Entscheidungen aus den Logs der _ki-Bots).
- **KI handelt selbst** (Burak, 23.10.2026: „KI soll selber Trades betreiben, Bots dienen lediglich zur
  Zusatzinformation“): Bot `ki_trader` (Strategie `KiTrader`, LightGBM/FreqAI, 1h, 8 Lern-Coins) bekommt die aktuellen
  Einstiegssignale der sechs 1h-Regel-Bots als Merkmale. Im Paper-Lernmodus übt sie bewusst: Einstieg ab 50 %
  Gewinnwahrscheinlichkeit plus ~40 % Übungstrades ab 35 % (Tag `ki_uebt`), Ausstieg spätestens nach 12 Stunden
  (`max_hold_candles`), Neutraining alle 4 h. bot_a/bot_b üben schwächer (52 %, 15 % ab 45 %). Alle Risikolimits gelten
  unverändert (inkl. Tagesverlustlimit 3 % – eine Lockerung bräuchte Buraks ausdrückliche Freigabe). Im Backtest und
  mit freigegebenem Modell gibt es keine Übungstrades. Ansicht: Dashboard-Reiter „KI-Trades“.
- **15-Minuten-Sammler**: die sechs 1h-Regeln zusätzlich auf 15m (`timeframe` je Paper-Strategie), damit häufiger
  gehandelt wird. Regeln unverändert, nur öfter geprüft (Unsicher: Kennzahlen der Regeln sind auf 1h ausgelegt).

## ADR-017: Zwei KIs mit gemeinsamem Gedächtnis (Burak, 23.10.2026)

- **KI 1 „Muster“** (`ki_trader`, `KiTrader`): überwachtes Lernen (LightGBM über FreqAI) auf Indikatoren mehrerer
  Zeitrahmen (1h + 1d) und den Signalen der Regel-Bots; stündliches Neutraining; Signal-Trades ab 50 % plus ~40 %
  Lern-Trades ohne Signal; Ausstieg per Modell oder spätestens nach 12 h. Signal-Trades prüft zusätzlich das KI-Gedächtnis.
- **KI 2 „Belohnung“** (`ki_reward`, `KiReward`, `kitb.reward_learner`): Contextual Bandit mit Thompson Sampling auf
  15-Minuten-Kerzen. Aktionen: nicht handeln / kaufen und 1 h, 4 h oder 12 h halten. Belohnung = Gewinn nach Gebühren in %.
  Bayes'sche lineare Regression je Aktion; lernt neu, sobald das gemeinsame Gedächtnis neue Trades enthält (≈ jede
  Minute). Lern-Trades = Aktionen, die nur durch Unsicherheit/Zufall (ε 5 %) gewählt wurden.
- **Warum sie sich ergänzen:** KI 1 lernt aus jeder Kerze („steigt der Kurs?“) – viele Beispiele, aber das Ziel ist nur
  ein Stellvertreter für Gewinn. KI 2 lernt direkt aus dem echten Ergebnis inkl. Gebühren und Haltedauer – genau das
  Ziel, aber mit wenigen Beispielen; durch Thompson Sampling erkundet sie gezielt dort, wo sie unsicher ist. Anderer
  Zeitrahmen (15m vs. 1h) -> andere Fehler; beide lernen aus allen Trades beider KIs.
- **Gemeinsames Gedächtnis** (`kitb.shared_memory`): jede KI hängt Einstiegsnotizen (Marktlage, Begründung, Art,
  Aktion) nur an ihre eigene Datei `ai_memory/snapshots/<ki>.jsonl` an (fsync). Einziger Schreiber der Datenbank
  `ai_memory/shared.sqlite` ist der Dienst `ai-memory` (jede Minute `sync`, idempotent per (KI, Runde, Trade-ID)).
  Bewusst Journal-Modus DELETE statt WAL: WAL braucht geteilten Speicher (mmap), der über Docker-Desktop-Bind-Mounts
  unter Windows nicht zuverlässig ist (Unsicher); bei nur einem Schreiber und reinen Lesern ist DELETE ausreichend.
  So kann gleichzeitiges Schreiben nichts beschädigen, und nach Neustarts wird alles aus den Quellen nachgezogen.
- **Ressourcen:** beide Modelle sind klassisches ML auf der CPU (LightGBM ~ Sekunden je Training, Bandit Millisekunden).
  Kein Sprachmodell, die Grafikkarte wird nicht benötigt (gemessen: RTX 5070 Ti Laptop 12 GB, im Container nutzbar –
  Reserve für einen späteren größeren Ansatz, z. B. kleines PPO-Netz). Unsicher: genaue RAM-Last aller Container zusammen.
- **Sicherheit:** beide nur Paper (dry_run, eigenes Konto je KI mit 10.000 USDT, Runden bei Drawdown), Risikolimits je
  KI (Tagesverlust 3 %, Position max. 40 %, Gesamteinsatz 80 %, Drawdown 15 %). Ausnahme (Entscheidungskarte Burak,
  23.10.2026): KIs im Paper-Lernmodus bis 6 gleichzeitige Positionen (`paper_learning.max_open_positions_learning`),
  Einsatz je Trade = 10.000 × 80 % / 6; Kandidaten und Sammler bleiben bei 2. Nur Long (Spot ohne
  Hebel); Short würde Futures/Margin erfordern und ist laut Projektvorgaben gesperrt.
- **Mehr Lerndaten (Burak, 23.10.2026: „was am besten für die Lerndaten ist“):** 16 statt 8 Lern-Coins (zusätzlich
  AVAX, DOT, LTC, TRX, NEAR, ATOM, BCH, UNI; trend_d1 bleibt BTC/ETH). KI 1 handelt auf 15-Minuten-Kerzen
  (`bots.ki_trader.timeframe`, Merkmale 15m/1h/4h, Ziel: +0,6 % in 12 Kerzen = 3 h) und schließt Lern-Trades
  spätestens nach 16 Kerzen (4 h). `max_hold_candles` zählt jetzt Kerzen des Bot-Zeitrahmens, die Lern-Trade-Ziehung
  (`explore_draw`) läuft je Kerze. Unsicher: Binance-Abfragegrenzen und Trainingszeit bei 25 Bots × 16 Coins –
  nach dem Start beobachten.
- **Plätze (Entscheidungskarte Burak, 23.10.2026, „Ja, 16 Plätze je KI“):** KIs im Paper-Lernmodus ein Platz je
  Lern-Coin (16), Einsatz je Trade = 10.000 × 80 % / 16 = 500 USDT. Plätze werden nie größer als die Anzahl Coins.
  Tagesverlust, Positionsgröße, Gesamteinsatz und Drawdown unverändert.

## ADR-018: Internet-Daten für die KIs (Burak, 23.10.2026)

- **Ziel:** Burak: „so viele Daten wie möglich aus dem Internet, damit die KIs gut aufgestellt sind“, alle kostenlosen
  Quellen ausschöpfen.
- **Dienst `market-flow`** (`kitb flow loop`, einziger Schreiber, nur öffentliche Daten, kein Handel) →
  `user_data/market_flow/*.parquet`.
  - **Binance Spot:** Kaufdruck aus taker buy volume (15m, Verlauf ab 2022), Orderbuch-Ungleichgewicht (nur
    Aufzeichnung).
  - **Binance USDT-Terminmarkt, nur gelesen:** Finanzierungsrate (ab 2022), Long/Short aller Konten und der
    Top-Trader, offene Positionen. Über die API gibt es dafür nur ~30 Tage, Nachladen aus data.binance.vision ist
    geplant.
  - **Gesamtmarkt:** Krypto Fear & Greed (alternative.me) und FRED-CSV ohne Key (VIX, US-Zins 10 J., Dollar-Index,
    S&P 500, Nasdaq 100).
- **Kein Blick in die Zukunft:**
  - Eine Kerze sieht nur Werte, die beim Schluss der vorherigen Kerze bekannt waren.
  - FRED-Tageswerte gelten erst nach 36 h (Unsicher, wann genau veröffentlicht).
  - Zu alte Werte (Dienst aus) werden nicht weitergetragen, sondern bleiben NaN.
- **Einbau:**
  - KI 1 bekommt `%-flow_*`/`%-ctx_*`-Merkmale (fehlend = 0, sonst würde FreqAI Zeilen verwerfen).
  - Das gemeinsame Gedächtnis (KI 2 und alle Einstiegsnotizen) bekommt dieselben Werte in `SNAPSHOT_FEATURES`.
  - Ältere Einträge ohne diese Werte zählen beim Lernen als Durchschnitt.
  - Folge: KI 2 hat mehr Merkmale und braucht mehr Trades, bis sie sich sicher ist (sie probiert länger aus).
- **Ansicht „KI-Daten“:** `kitb.data_overview` → `market_flow/overview.json` (alle 5 Min.): eingeführte Daten je Quelle
  (Zeilen, Zeitraum), Lerndaten aus eigenen Trades je KI (gesamt, 24 h, heute), was jede KI pro Training nutzt
  (Schätzung).
- **Schlagzeilen:** Google-News-RSS je Coin alle 30 Min. mit einfachem Wörterbuch-Stimmungswert (nicht kalibriert),
  vorerst nur Aufzeichnung (kein rückwirkender Abruf möglich). Als Merkmal erst nach genug Verlauf.
- **Flotte:** bollinger_1h und bollinger_1h_ki sind abgeschaltet (≈0,4 Trades je Coin und Monat, seit Start kein Trade).
  Burak hat die Flotte am 23.10. an Claude delegiert.
- **Methoden-Suche (23.10.2026, echte Daten, 16 Coins, 2022–11/2025):** Keine der 19 Methoden erreichte ≥ 60 % Treffer
  UND Gewinn nach Gebühren in beiden Zeiträumen. Methoden mit hoher Trefferquote verlieren nach Gebühren (z. B.
  VWAP-Rückkehr 15m: 61 % Treffer, −59 %); die Gewinner treffen selten (Ichimoku 4h: 34 %, +58 %). Daher keine neuen
  Bots. Alle 19 Einstiegssignale fließen stattdessen als Merkmale `%-m_*`/`%-m4_*` in KI 1 ein.

## ADR-019: Ideen aus Vibe-Trading und weitere freie Quellen (23.10.2026)

- **Vibe-Trading (HKUDS, MIT)** ist ein Chat-Recherche-Assistent mit Sprachmodell (Daten holen, Strategie schreiben,
  Backtest, Bericht). Für unser selbstlernendes System nicht als Ganzes passend; Code wird **nicht** übernommen.
- Übernommen als Idee, eigene Umsetzung: **Faktor-Test** `kitb alphas bench` (`src/kitb/alphas.py`). 34 öffentliche
  Formeln (Auswahl Kakushadze 101, Qlib-Alpha158-Art, klassische Querschnittsfaktoren) werden auf Tageskerzen aller
  Lern-Coins mit dem IC (Rangkorrelation zur späteren Rendite) bewertet, früh UND spät, Test-Abschnitt unberührt.
  Nur was in beiden Zeiträumen lebt, darf als Merkmal in einen A/B-Test der KIs.
- Neue freie Quellen ohne Key (Recherche GitHub/Web): Deribit-DVOL BTC/ETH, DefiLlama Stablecoin-Menge,
  Coin Metrics Community (aktive Adressen, MVRV; CC BY-NC 4.0, private Nutzung). Als `ctx_*`-Merkmale mit
  Veröffentlichungs-Verzögerung (25–36 h) eingebaut. Antwortformate erst am PC geprüft (aus der Cloud geblockt).
- Verworfen: Scraper für Chart-Webseiten (z. B. chaindl: Nutzungsbedingungen unklar, bricht leicht), Santiment und
  CoinGlass (Key bzw. kostenpflichtig für Verlauf), Apify-Scraper (kostenpflichtig).
