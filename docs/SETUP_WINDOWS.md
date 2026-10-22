# Einrichtung unter Windows 11 (Legion Pro 5i)

> Diese Schritte wurden **nicht** auf Windows getestet – die Entwicklung lief in einer Linux-Cloud-Umgebung
> (Docker 29.9 + Image `freqtradeorg/freqtrade:2026.9_freqai`). Abweichungen bitte in PROJECT_STATE.md festhalten.

## 1. WSL2 und Docker Desktop

1. PowerShell **als Administrator**: `wsl --install` → Neustart. Danach Ubuntu aus dem Startmenü öffnen und Benutzer anlegen.
2. Docker Desktop installieren, in den Einstellungen „Use the WSL 2 based engine“ aktivieren und unter
   *Resources → WSL Integration* die Ubuntu-Distribution einschalten.
3. Prüfen (in Ubuntu/WSL): `docker version` und `docker compose version`.

## 2. Projekt holen

Projekt **im WSL-Dateisystem** ablegen (z. B. `~/ki-trading-bot`), nicht unter `/mnt/c/...` –
Bind-Mounts aus dem Windows-Laufwerk sind deutlich langsamer (Erfahrungswert, Unsicher in der Größenordnung).

```bash
cd ~ && git clone <REPO-URL> ki-trading-bot && cd ki-trading-bot
```

## 3. Bauen und testen

```bash
docker compose build
docker compose run --rm kitb check-safety
docker compose run --rm --entrypoint pytest kitb -q
```

Erwartet: `OK: Konfiguration erfüllt …` und `154 passed, 1 deselected` (Stand Phase 8).

Hinweis Berechtigungen: Der Container läuft als `ftuser` (UID 1000). Unter WSL hat der erste Benutzer
normalerweise ebenfalls UID 1000. Falls `user_data` nicht beschreibbar ist: `sudo chown -R 1000:1000 user_data`.

## 4. Echte Daten

```bash
docker compose run --rm kitb probe binance okx kraken      # Erreichbarkeit vom eigenen Standort
docker compose run --rm --entrypoint pytest kitb -q -m network
docker compose run --rm kitb download --exchange binance --timerange 20220101-
docker compose run --rm kitb validate --source binance --check-fresh
docker compose run --rm kitb backtest --source binance --timerange 20230101-
```

Ist Binance nicht erreichbar, eine erreichbare Börse aus `probe` wählen und in `config/freqtrade.base.json`
`exchange.name` anpassen (nur Werte aus `config/project.json → allowed_exchanges`).

## 5. Strategien und KI-Bots auf echten Daten vergleichen

```bash
docker compose run --rm kitb compare --source binance
docker compose run --rm kitb ml-compare --source binance        # dauert Minuten (Bot B ≈ 10–13× länger als Bot A)
docker compose run --rm kitb agents-evaluate --source binance
docker compose run --rm kitb models list
```

Berichte: `user_data/reports/<lauf>/report.md`. Der Test-Abschnitt bleibt gesperrt, bis `--include-test` bewusst gesetzt wird.

## 6. Modell freigeben und Paper-Trading starten

Nur wenn eine Modellversion die Kriterien erfüllt (`models list` → Status `passed`) und du sie bewusst übernehmen willst:

```bash
docker compose run --rm kitb models approve  --id <ID> --by "Burak"
docker compose run --rm kitb models activate --id <ID> --by "Burak"
docker compose --profile paper up -d paper-bot-a paper-bot-b
docker compose run --rm kitb paper status
docker compose --profile dashboard up -d dashboard        # Browser: http://127.0.0.1:8501
```

Not-Stopp: `docker compose run --rm kitb kill-switch on --reason "..." --by "Burak"` (oder im Dashboard, Reiter „Not-Stopp“).
Zurücksetzen: `kill-switch off --by "Burak"`. Beenden: `docker compose --profile paper down`.

Damit der Rechner das Paper-Trading über Wochen trägt: Energiesparmodus/Ruhezustand abschalten, Docker Desktop mit
Windows starten lassen. Die Dienste starten nach einem Absturz neu (`restart: on-failure`), nach einem Windows-Neustart
aber erst, wenn Docker Desktop läuft (Unsicher, ob Docker Desktop sie dann automatisch wieder hochfährt – bitte prüfen).

## 7. Alternative ohne Docker (in WSL2-Ubuntu)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # uv installieren (oder per pipx)
uv venv --python 3.14 .venv && source .venv/bin/activate
uv pip install -c constraints.txt -e ".[dev,ml,dashboard]"
kitb check-safety && pytest -q
```

Native Windows-Installation (ohne WSL) wird nicht empfohlen (Freqtrade-Doku: ggf. Visual-C++-Build-Tools nötig).

## GPU

Nicht benötigt: Bot A und Bot B trainieren auf der CPU in Sekunden je Modell. Ob die RTX 5070 Ti in Docker/WSL2 für
spätere größere Modelle nutzbar ist, wurde nicht geprüft (Weiß ich nicht ohne Test auf dem Gerät).
