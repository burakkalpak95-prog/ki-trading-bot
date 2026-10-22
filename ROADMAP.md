# Roadmap

Die Phasen 2–8 wurden nach Anweisung des Nutzers („nach jeder Phase selbständig fortsetzen“) ohne Einzelfreigabe
umgesetzt. **Ausnahme bleibt Live-Trading:** nur nach separater Freigabe und unabhängiger Prüfung.

| Phase | Inhalt | Status |
|---|---|---|
| 1 Fundament | Umgebung, Framework, Daten, Validierung, Speicherung, Baseline-Backtest | abgeschlossen (Windows, echte Daten) |
| 2 Strategien | 6 Regel-Strategien, Indikator-Modul, chronologische Splits, rollierende Fenster, Marktphasen, Vergleichsbericht | umgesetzt, auf echten Daten ausgewertet (kein Vorteil ggü. Buy&Hold) |
| 3 KI-Bots | Bot A (LogReg) und Bot B (LightGBM) über FreqAI, Walk-Forward, Ressourcenvergleich | umgesetzt, auf echten Daten ausgewertet (Freigabe nicht erreicht); GPU nicht geprüft |
| 4 Lernsystem | Modellregister, Freigabekriterien, menschliche Freigabe, Rollback, Entscheidungslog mit Feature-Einfluss, Drift | umgesetzt |
| 5 Multi-Agenten | Regelbasierte Veto-Agenten + Nutzenauswertung | umgesetzt, **deaktiviert** (kein belegter Nutzen) |
| 6 Dashboard | Streamlit, nur 127.0.0.1, 10 Reiter; OpenTerminal geprüft und verworfen | umgesetzt |
| 7 Paper-Trading | Zentrales Risikomodul, Not-Stopp, Freqtrade-Dry-Run je Bot | umgesetzt; **Langzeitlauf nur auf dem Zielrechner möglich** |
| 8 Live (optional) | Gesperrte Schnittstelle, Bereitschafts-Checkliste, Bestätigung der Risikolimits | Schnittstelle umgesetzt, **Live gesperrt** |

## Stand nach den ersten echten Daten (22.10.2026)

Phase 1 ist auf dem Windows-Rechner mit echten Daten abgeschlossen (Binance, Gegenprobe OKX). Kein Ansatz schlägt
Buy&Hold; kein Modell wird freigegeben. Runde 2 (ADR-014): fairere Auswertung und zwei langsame Tagesstrategien.

## Nächste Schritte (in dieser Reihenfolge)

1. Runde 2 auf echten Daten ausführen: `compare` (1h und 1d), `agents-evaluate`.
2. Besteht ein Kandidat die Validation überzeugend (auch risikobereinigt und über die Fenster): einmalig den Test-Abschnitt
   prüfen (`--include-test`, wird protokolliert).
3. Erst danach: Modell bzw. Strategie bewusst freigeben und Paper-Trading über mehrere Wochen.
4. Live-Trading nur nach separater Freigabe und unabhängiger Prüfung.
