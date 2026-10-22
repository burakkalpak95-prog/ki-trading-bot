# KI-Trading-Bot

Modulares, lokal betriebenes Analyse- und Paper-Trading-System für **BTC/USDT** und **ETH/USDT**
im **Spotmarkt ohne Hebel**. Stand: Phasen 1–8 umgesetzt, **Live-Trading gesperrt**.

> **Kein Echtgeld.** Paper-Trading ist ein Freqtrade-Dry-Run mit virtuellem Kapital. Live-Trading existiert nur als
> gesperrte Schnittstelle. Backtest-Ergebnisse sind keine Profitabilitätsaussage. Alle bisherigen Ergebnisse stammen
> aus **synthetischen Daten** und sind ausschließlich Funktionstests der Pipeline.

## Aufbau

| Baustein | Lösung |
|---|---|
| Trading-/Backtest-Engine, Datendownload, Paper-Trading | [Freqtrade](https://github.com/freqtrade/freqtrade) 2026.9 (GPLv3), Docker-Image `freqtradeorg/freqtrade:2026.9_freqai` |
| KI-Training (Walk-Forward) | FreqAI (in Freqtrade enthalten), scikit-learn 1.9.1, LightGBM 4.7.0 |
| Börsenanbindung | CCXT 4.5.84 (über Freqtrade), nur öffentliche Daten |
| Historische Daten | Parquet im Freqtrade-Format |
| Protokoll, Modellregister, Entscheidungen | SQLite `user_data/kitb.sqlite` |
| Dashboard | Streamlit 1.66.1 (Apache-2.0), nur `127.0.0.1` |
| Eigene Ergänzung (`src/kitb`) | Sicherheit, Validierung, Kosten, Vergleiche, Modell-Governance, Drift, Risiko, Not-Stopp, CLI |

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · Windows: [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md) ·
Sicherheit: [docs/SECURITY.md](docs/SECURITY.md) · Planung: [ROADMAP.md](ROADMAP.md) · Stand: [PROJECT_STATE.md](PROJECT_STATE.md)

## Projektstruktur

```
config/
  freqtrade.base.json     Freqtrade-Basis (dry_run=true, spot, keine Keys, kein API-Server)
  project.json            Börsen/Märkte, Kosten, Validierung, Auswertung, Bots, Agenten
  freqai.shared.json      Walk-Forward-Einstellungen (90 Tage Training, Neutraining alle 30 Tage)
  release_criteria.json   Freigabekriterien für Modellversionen
  risk.json               Risikolimits (Positionsgröße, Tagesverlust, Drawdown, ...)
src/kitb/
  cli.py                  Kommandozeile `kitb`
  safety.py               Sicherheitsprüfungen (Modi data/backtest/paper; live nie)
  data/                   Abruf, Validierung (15 Prüfungen), Speicherung, synthetische Daten
  backtest.py, costs.py   Backtest mit Gebühr + Slippage + Spread, Buy&Hold, p-Wert
  evaluation.py, compare.py  Chronologische Splits, rollierende Fenster, Marktphasen, Strategievergleich
  ml.py, ml_strategy.py   Bot A / Bot B über FreqAI, Ressourcenmessung, Vergleich
  explain.py, decisions.py  Feature-Einfluss je Vorhersage, Entscheidungsprotokoll
  models.py               Modellregister: Kandidat -> geprüft -> freigegeben (Mensch) -> aktiv, Rollback
  drift.py                Markt-Drift (PSI, KS-Test) und Leistungs-Drift (nur Meldung)
  agents/                 Regelbasierte Analyse-Agenten (nur Einstiegs-Veto, standardmäßig aus)
  risk.py                 Zentrales Risikomodul (in allen Strategien aktiv)
  killswitch.py           Not-Stopp (Datei user_data/KILL_SWITCH)
  paper.py                Paper-Trading-Steuerung (Freqtrade-Dry-Run je Bot)
  live.py                 Gesperrter Live-Adapter + Bereitschafts-Checkliste
  dashboard/              Streamlit-App (10 Reiter)
user_data/strategies/     6 Regel-Strategien (1h), 2 Tagesstrategien, Bot A/B, 2 Agenten-Varianten
user_data/freqaimodels/   KitbLogisticRegression, KitbLightGBMClassifier
tests/                    Offline-Tests; `-m network` für echte Börsenabfragen
docker/Dockerfile, docker-compose.yml, constraints.txt
```

## Schnellstart (Docker, empfohlen – auch unter Windows/WSL2)

```bash
docker compose build
docker compose run --rm kitb check-safety
docker compose run --rm --entrypoint pytest kitb -q            # Tests (offline)
```

### Daten

```bash
docker compose run --rm kitb probe binance okx kraken          # welche Quelle ist erreichbar?
docker compose run --rm kitb download --exchange binance --timerange 20220101-
docker compose run --rm kitb validate --source binance --check-fresh
# ohne Internet: synthetische, gekennzeichnete Daten (+ --offline-markets bei den folgenden Befehlen)
docker compose run --rm kitb synthetic
```

### Auswertung

```bash
docker compose run --rm kitb backtest --source binance --timerange 20230101-   # Baseline
docker compose run --rm kitb compare --source binance                          # alle Regel-Strategien
docker compose run --rm kitb compare --source binance --timeframe 1d           # Tagesstrategien (200-Tage-Trend, 90-Tage-Momentum)
docker compose run --rm kitb ml-compare --source binance                       # Bot A vs. Bot B vs. Regeln
docker compose run --rm kitb agents-evaluate --source binance                  # Nutzen der Agenten?
docker compose run --rm kitb drift --source binance
docker compose run --rm kitb decisions
docker compose run --rm kitb runs backtest_runs
```

Der Test-Abschnitt (letzte 20 % der Daten) bleibt unberührt, bis er bewusst mit `--include-test` verwendet wird
(wird protokolliert). Berichte landen in `user_data/reports/`.

### Modelle (Freigabe nur durch einen Menschen)

```bash
docker compose run --rm kitb train-cycle --source binance          # trainiert + prüft, aktiviert nie selbst
docker compose run --rm kitb models list
docker compose run --rm kitb models approve  --id 3 --by "Burak"
docker compose run --rm kitb models activate --id 3 --by "Burak"
docker compose run --rm kitb models rollback --bot bot_a --by "Burak"
```

### Dashboard und Paper-Trading

```bash
docker compose --profile dashboard up dashboard                     # http://127.0.0.1:8501
docker compose --profile paper up -d paper-bot-a paper-bot-b        # benötigt aktive, freigegebene Modelle
docker compose run --rm kitb paper status
docker compose run --rm kitb paper report
docker compose run --rm kitb kill-switch on                         # Not-Stopp; zurücksetzen: kill-switch off
docker compose --profile paper down
```

### Live

```bash
docker compose run --rm kitb live readiness       # Checkliste – Live bleibt gesperrt
```

## Lokal ohne Docker (Linux/WSL2)

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -c constraints.txt -e ".[dev,ml,dashboard]"
kitb check-safety && pytest -q
```

## Kosten

Keine. Alle Komponenten sind Open Source, die genutzten Börsen-Endpunkte sind öffentlich und ohne Konto nutzbar.
Docker Desktop ist nach meinem Kenntnisstand für private Nutzung kostenlos – bitte die aktuellen Lizenzbedingungen
von Docker selbst prüfen (Unsicher).
