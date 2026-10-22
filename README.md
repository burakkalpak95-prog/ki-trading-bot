# KI-Trading-Bot – Phase 1 (Fundament)

Modulares, lokal betriebenes Analyse- und (später) Trading-System für **BTC/USDT** und **ETH/USDT**
im **Spotmarkt ohne Hebel**. Aktueller Stand: **nur Phase 1** – Daten, Validierung, Speicherung,
regelbasierte Baseline und reproduzierbare Backtests.

> **Kein Handel.** Paper- und Live-Trading sind in Phase 1 technisch gesperrt.
> Backtest-Ergebnisse sind keine Profitabilitätsaussage. Ergebnisse auf synthetischen Daten
> sind ausschließlich Funktionstests der Pipeline.

## Aufbau

| Baustein | Lösung |
|---|---|
| Trading-/Backtest-Engine, Datendownload | [Freqtrade](https://github.com/freqtrade/freqtrade) 2026.9 (GPLv3), Docker-Image `freqtradeorg/freqtrade:2026.9` |
| Börsenanbindung | CCXT 4.5.84 (über Freqtrade) |
| Historische Daten | Parquet im Freqtrade-Format (`user_data/data/<börse>/BTC_USDT-1h.parquet`) |
| Protokoll (Abrufe, Validierungen, Backtests) | SQLite `user_data/kitb.sqlite` |
| Eigene Ergänzung (`src/kitb`) | Sicherheitsprüfungen, Datenvalidierung, Kostenmodell, Offline-Testdaten, CLI |
| Baseline | `user_data/strategies/BaselineEmaCross.py` (EMA 21/55-Kreuzung, Stoploss −10 %) |

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · Windows-Einrichtung: [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md) ·
Sicherheit: [docs/SECURITY.md](docs/SECURITY.md) · Planung: [ROADMAP.md](ROADMAP.md) · Stand: [PROJECT_STATE.md](PROJECT_STATE.md)

## Projektstruktur

```
config/
  freqtrade.base.json   Freqtrade-Konfiguration (dry_run=true, spot, keine Keys, kein API-Server)
  project.json          Freigegebene Börsen/Märkte, Zeitintervalle, Kostenmodell, Validierungsgrenzen
src/kitb/
  cli.py                Kommandozeile `kitb` (nur Daten- und Backtest-Befehle)
  safety.py             Phase-1-Sperren und Konfigurationsprüfung
  costs.py              Gebühr + Slippage + halber Spread -> effektive Gebühr je Seite
  backtest.py           Validierung -> Freqtrade-Backtest -> Kennzahlen + Buy&Hold -> SQLite
  registry.py           SQLite-Protokoll
  offline.py            Statische Marktdefinitionen für Offline-Backtests (nur synthetische Daten)
  retry.py              Retry mit exponentiellem Backoff
  data/source.py        Erreichbarkeitsprüfung (Ticker, Kerzen) + Aufruf `freqtrade download-data`
  data/validate.py      Prüfungen: UTC, Sortierung, Duplikate, Lücken, Raster, NaN, OHLC, Ausreißer, unvollständige/veraltete Kerzen
  data/storage.py       Parquet lesen/schreiben über Freqtrade-Datahandler
  data/synthetic.py     Deterministische, gekennzeichnete synthetische Daten
user_data/strategies/   Freqtrade-Strategien (Baseline)
tests/                  Unit- und Integrationstests (offline); `-m network` für echte Börsenabfragen
docker/Dockerfile       Freqtrade-Image + kitb
docker-compose.yml      Ein Dienst `kitb`, keine Ports
constraints.txt         Exakte Paketversionen aus dem Freqtrade-Image (für lokale venvs)
```

## Schnellstart (Docker, empfohlen – auch unter Windows)

```bash
docker compose build
docker compose run --rm kitb check-safety                     # Sicherheitsregeln prüfen
docker compose run --rm --entrypoint pytest kitb -q            # Tests (offline)

# Echte Daten (braucht Internet zur Börse)
docker compose run --rm kitb probe binance okx kraken          # welche Quelle ist erreichbar?
docker compose run --rm kitb download --exchange binance --timerange 20220101-
docker compose run --rm kitb validate --source binance --check-fresh
docker compose run --rm kitb backtest --source binance --timerange 20230101-

# Ohne Internet: klar gekennzeichnete synthetische Daten
docker compose run --rm kitb synthetic
docker compose run --rm kitb backtest --source synthetic --offline-markets

docker compose run --rm kitb runs backtest_runs                # Protokoll ansehen
```

## Lokal ohne Docker (Linux/WSL2)

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -c constraints.txt -e ".[dev]"
kitb check-safety && pytest -q
```

## Kosten

Keine. Alle Komponenten sind Open Source, die genutzten Börsen-Endpunkte sind öffentlich und ohne Konto
nutzbar. Docker Desktop ist nach meinem Kenntnisstand für private Nutzung kostenlos – bitte die aktuellen
Lizenzbedingungen von Docker selbst prüfen (Unsicher).
