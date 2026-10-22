# Haelt Bot und Uebersicht an. Offene (simulierte) Positionen bleiben in der Datenbank und laufen beim naechsten Start weiter.
$ErrorActionPreference = "Continue"  # native Befehle pruefen wir ueber $LASTEXITCODE
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
Write-Host "== KI-Trading-Bot: Stopp ==" -ForegroundColor Cyan
Write-Host "Solange der Bot gestoppt ist, trifft er keine Entscheidungen (auch keine Verkaeufe)."
$answer = Read-Host "Wirklich stoppen? (j/n)"
if ($answer -notin @("j", "J", "ja", "Ja")) { Write-Host "Abgebrochen."; exit 0 }
docker compose --profile paper-trend --profile paper-learn --profile dashboard stop
Write-Host "Gestoppt." -ForegroundColor Yellow
