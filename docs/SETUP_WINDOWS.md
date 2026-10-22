# Einrichtung unter Windows 11 (Legion Pro 5i)

> Diese Schritte wurden **nicht** auf Windows getestet – die Entwicklung lief in einer Linux-Cloud-Umgebung
> (Docker 29.9 + Image `freqtradeorg/freqtrade:2026.9`). Abweichungen bitte in PROJECT_STATE.md festhalten.

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

Erwartet: `OK: Konfiguration erfüllt …` und `56 passed, 1 deselected`.

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

## 5. Alternative ohne Docker (in WSL2-Ubuntu)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # uv installieren (oder per pipx)
uv venv --python 3.14 .venv && source .venv/bin/activate
uv pip install -c constraints.txt -e ".[dev]"
kitb check-safety && pytest -q
```

Native Windows-Installation (ohne WSL) wird nicht empfohlen (Freqtrade-Doku: ggf. Visual-C++-Build-Tools nötig).

## GPU

Für Phase 1 nicht nötig. Ob und wie die RTX 5070 Ti in Docker/WSL2 für spätere Modelle genutzt wird,
wird in Phase 3 geprüft (Treiber-/CUDA-Kompatibilität: Weiß ich nicht verlässlich ohne Test auf dem Gerät).
