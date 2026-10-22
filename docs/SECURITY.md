# Sicherheit – Stand Phase 1

## Umgesetzte Schutzmaßnahmen

| Risiko | Maßnahme | Test |
|---|---|---|
| Versehentliches Paper-/Live-Trading | `kitb` hat keinen Trade-Befehl; `assert_run_mode_allowed` erlaubt nur `data`/`backtest`; `freqtrade trade`/`webserver` sind als gesperrt definiert | `tests/test_safety.py` |
| Echtgeldorder durch Konfiguration | Pflichtprüfung: `dry_run` explizit `true`, `trading_mode=spot`, kein `margin_mode`, keine Hebelparameter, `initial_state=stopped`, `force_entry_enable=false` | `tests/test_safety.py` |
| Zugangsdaten im Repo | Keine Keys nötig (nur öffentliche Daten). Prüfung bricht ab, sobald `key/secret/password/uid/...` gesetzt ist. `.gitignore` schließt `.env`, `*.secret.json`, `config/*.local.json` aus | `tests/test_safety.py` |
| Offene Ports | Kein API-Server, kein Telegram; `docker-compose.yml` veröffentlicht keine Ports; api_server dürfte nur an 127.0.0.1 binden | `tests/test_safety.py` |
| Nicht freigegebene Märkte/Börsen | Whitelist in `config/project.json` | `tests/test_safety.py` |
| Fehlerhafte/veraltete Daten | Validierung vor jedem Backtest, Abbruch bei ERROR; unvollständige letzte Kerze wird nach Download entfernt | `tests/test_validate.py`, `tests/test_backtest_integration.py` |
| Offline-Märkte mit echten Daten verwechselt | `--offline-markets` nur für gekennzeichnete synthetische Daten | `tests/test_backtest_integration.py` |

## Offene Risiken / Hinweise

- `freqtrade` selbst ist im Image installiert. Wer `docker compose run --entrypoint freqtrade kitb trade …` aufruft,
  umgeht die kitb-Sperren. Gegenmaßnahme in Phase 1: Konfiguration enthält keine Keys und `dry_run=true`,
  d. h. selbst dann wäre nur ein simulierter Lauf möglich. Eine harte Sperre auf Freqtrade-Ebene ist erst bei
  Einführung von Paper-Trading (Phase 7) sinnvoll.
- Abhängigkeiten aus dem Freqtrade-Image (≈ 90 Pakete) werden nicht separat auditiert; Updates nur bewusst über
  neuen Image-Tag + `constraints.txt`.
- GPLv3-Lizenz von Freqtrade beachten, falls das System weitergegeben wird.
- Für spätere Live-Phasen nötig (nicht umgesetzt): Secret-Verwaltung, Kill-Switch, Order-Reconciliation,
  Abhebungs-Sperre auf API-Key-Ebene.
