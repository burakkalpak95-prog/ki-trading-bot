# Roadmap

Die Phasen 2–8 wurden nach Anweisung des Nutzers („nach jeder Phase selbständig fortsetzen“) ohne Einzelfreigabe
umgesetzt. **Ausnahme bleibt Live-Trading:** nur nach separater Freigabe und unabhängiger Prüfung.

| Phase | Inhalt | Status |
|---|---|---|
| 1 Fundament | Umgebung, Framework, Daten, Validierung, Speicherung, Baseline-Backtest | umgesetzt – echte Daten auf dem Zielrechner offen |
| 2 Strategien | 6 Regel-Strategien, Indikator-Modul, chronologische Splits, rollierende Fenster, Marktphasen, Vergleichsbericht | umgesetzt (nur synthetisch getestet) |
| 3 KI-Bots | Bot A (LogReg) und Bot B (LightGBM) über FreqAI, Walk-Forward, Ressourcenvergleich | umgesetzt (nur synthetisch); GPU nicht geprüft |
| 4 Lernsystem | Modellregister, Freigabekriterien, menschliche Freigabe, Rollback, Entscheidungslog mit Feature-Einfluss, Drift | umgesetzt |
| 5 Multi-Agenten | Regelbasierte Veto-Agenten + Nutzenauswertung | umgesetzt, **deaktiviert** (kein belegter Nutzen) |
| 6 Dashboard | Streamlit, nur 127.0.0.1, 10 Reiter; OpenTerminal geprüft und verworfen | umgesetzt |
| 7 Paper-Trading | Zentrales Risikomodul, Not-Stopp, Freqtrade-Dry-Run je Bot | umgesetzt; **Langzeitlauf nur auf dem Zielrechner möglich** |
| 8 Live (optional) | Gesperrte Schnittstelle, Bereitschafts-Checkliste, Bestätigung der Risikolimits | Schnittstelle umgesetzt, **Live gesperrt** |

## Nächste Schritte (in dieser Reihenfolge)

1. Ziel-Repository festlegen und Code pushen.
2. Windows 11: `docker compose build`, Tests, `kitb probe binance okx kraken`.
3. Echte BTC/ETH-Daten ab 2022 laden und validieren.
4. `kitb compare` und `kitb ml-compare` auf echten Daten – Ergebnisse in PROJECT_STATE.md eintragen.
5. Erfüllt eine Modellversion die Kriterien: bewusst freigeben und aktivieren (`kitb models approve/activate --by <Name>`).
6. Paper-Trading beider Bots über mehrere Wochen bis Monate (Dauer: Unsicher, Checkliste verlangt mind. 100 Trades).
7. `kitb agents-evaluate` auf echten Daten wiederholen; Agenten nur bei belegtem Nutzen einschalten.
8. Erst danach über Live-Trading entscheiden (separate Freigabe, unabhängige Prüfung).
