# Sicherheit – Stand Phase 8

## Umgesetzte Schutzmaßnahmen

| Risiko | Maßnahme | Test |
|---|---|---|
| Echtgeldhandel | `safety` erlaubt nur die Modi `data`, `backtest`, `paper`; `live` nie. `freqtrade trade` nur im Modus `paper`. Live-Adapter `LockedLiveAdapter` lehnt jede Aktion ab | `tests/test_safety.py`, `tests/test_live_locked.py` |
| Echtgeldorder durch Konfiguration | Pflichtprüfung: `dry_run=true`, `trading_mode=spot`, kein `margin_mode`, keine Hebelparameter, `force_entry_enable=false`; Paper-Konfiguration wird erneut geprüft | `tests/test_safety.py`, `tests/test_risk_paper.py` |
| Hebel/Futures/Margin/Auszahlungen | In `config/risk.json` fest `false`; `RiskLimits` bricht sonst ab | `tests/test_risk_paper.py` |
| Zugangsdaten im Repo | Keine Keys nötig. Prüfung bricht ab, sobald `key/secret/password/uid/...` gesetzt ist. `.gitignore` schließt `.env`, `*.secret.json`, `config/*.local.json` aus | `tests/test_safety.py` |
| Offene Ports | Kein Freqtrade-API-Server, kein Telegram. `kitb`- und Paper-Dienste ohne Ports; Dashboard nur `127.0.0.1:8501`, Telemetrie aus | `tests/test_safety.py` |
| Verluste im Paper-/späteren Live-Betrieb | Zentrales Risikomodul in allen Strategien: Positionsgröße, Risiko je Trade, Gesamtexposure, Tagesverlust, Drawdown-Stopp, Datenalter, Volumenanteil | `tests/test_risk_paper.py` |
| Fehlverhalten im Lauf | Not-Stopp-Datei `user_data/KILL_SWITCH`: neue Einstiege abgelehnt, Bots beenden sich; Zurücksetzen nur durch Menschen | `tests/test_risk_paper.py`, `tests/test_dashboard.py` |
| KI ändert selbst Modelle/Limits/Modus | Modell-Freigabe, Freigabe von Regel-Strategien für Paper-Trading (`kitb strategies approve`), Aktivierung, Rollback, Not-Stopp-Reset und Bestätigung der Risikolimits verlangen einen menschlichen Namen; Drift und Agenten melden/vetoen nur | `tests/test_learning_system.py`, `tests/test_agents.py` |
| Overfitting / Datenleck | Chronologischer Split, Test-Abschnitt gesperrt (Nutzung protokolliert), FreqAI `shuffle=false`, p-Wert-Kriterium | `tests/test_evaluation.py`, `tests/test_compare_integration.py`, `tests/test_strategy_lookahead.py` |
| Fehlerhafte/veraltete Daten | Validierung vor jedem Backtest, Abbruch bei ERROR; Datenalter-Prüfung im Risikomodul | `tests/test_validate.py` |
| Offline-Märkte mit echten Daten verwechselt | `--offline-markets` nur für gekennzeichnete synthetische Daten | `tests/test_backtest_integration.py` |

## Offene Risiken / Hinweise

- `freqtrade` ist im Image installiert. Wer Freqtrade direkt (`--entrypoint freqtrade`) mit eigener Konfiguration und
  echten Keys startet, umgeht alle kitb-Sperren. Gegenmaßnahme: keine Keys auf dem Rechner hinterlegen, solange Live
  nicht freigegeben ist. Eine technische Sperre dagegen gibt es nicht.
- Dry-Run-Ausführung ist eine Simulation: Teilausführungen, Latenz und echte Orderbuch-Reaktionen weichen ab.
- Der Not-Stopp wirkt nur auf Bots, die über `kitb` (Risikomodul) laufen; er schließt keine offenen Positionen
  automatisch (im Paper-Betrieb ohne Folgen).
- Die Abhängigkeiten aus dem Freqtrade-Image werden nicht separat auditiert; Updates nur bewusst über neuen Image-Tag
  + `constraints.txt`.
- Das Dashboard hat keine Anmeldung. Es ist nur lokal erreichbar; wer Port 8501 nach außen freigibt, macht
  Handelsdaten und den Not-Stopp-Schalter öffentlich.
- GPLv3-Lizenz von Freqtrade beachten, falls das System weitergegeben wird (keine Rechtsberatung).
- Für einen späteren Live-Betrieb fehlen bewusst: geprüfter Live-Adapter, Order-Abgleich mit der Börse,
  Secret-Verwaltung, API-Key ohne Auszahlungsrecht, unabhängige Prüfung.
