# Roadmap

Jede Phase startet erst nach **ausdrücklicher Freigabe**.

| Phase | Inhalt | Status |
|---|---|---|
| 1 Fundament | Umgebung, Framework-Wahl, Struktur, BTC/ETH-Daten, Validierung, Speicherung, Backtest-Baseline | **umgesetzt** (echte Daten noch auf dem Zielrechner zu prüfen) |
| 2 Strategien | Klassische Strategien (Trend, Momentum, Mean Reversion, Breakout, Volatilität), Indikatoren, Vergleichsmetriken, Walk-Forward-Grundlage, Marktphasen-Auswertung, erstes Risk-Modul | offen |
| 3 KI-Bots | Bot A (einfach, z. B. LogReg/RandomForest/LightGBM) und Bot B (erweitert) über FreqAI unter identischen Bedingungen; Vergleich inkl. Rechenzeit/Ressourcen; GPU-Prüfung | offen |
| 4 Lernsystem | Versionierte Modelle, Entscheidungshistorie, Drift-Erkennung, Freigabekriterien, Rollback | offen |
| 5 Multi-Agenten | Nur bei belegtem Zusatznutzen | offen |
| 6 Dashboard | OpenTerminal prüfen, sonst schlanke Lösung (z. B. Streamlit/FreqUI) | offen |
| 7 Langzeit-Paper-Trading | Freqtrade dry-run für beide Bots, Sicherheitsabschaltung | offen |
| 8 Optional Live | Nur nach separater Freigabe und Prüfung | offen |

## Vorgeschlagener nächster Schritt (Phase 1 abschließen)

1. Auf dem Windows-Rechner `docker compose build` + Tests ausführen.
2. `kitb probe binance okx kraken` → erreichbare Börse festlegen.
3. Echte BTC/ETH-Daten (1h, 1d, ab 2022) laden, validieren, Baseline-Backtest mit echten Daten ausführen.
4. Ergebnisse in PROJECT_STATE.md eintragen; danach Freigabe für Phase 2.
