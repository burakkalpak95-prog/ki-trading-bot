# Startet Paper-Trading (DailyTrendSma, nur simuliert) und die Übersicht, dann öffnet sich der Browser.
# Aufruf über "Trading-Bot starten.bat" im Projektordner (Doppelklick).
$ErrorActionPreference = "Continue"  # native Befehle pruefen wir ueber $LASTEXITCODE
Set-Location (Resolve-Path "$PSScriptRoot\..\..")

function Test-Docker { docker info *> $null; return $LASTEXITCODE -eq 0 }

Write-Host "== KI-Trading-Bot: Start (Paper-Trading, kein echtes Geld) ==" -ForegroundColor Cyan
if (-not (Test-Docker)) {
    Write-Host "Docker Desktop wird gestartet ..."
    docker desktop start *> $null
    if ($LASTEXITCODE -ne 0) {
        $candidates = @("$env:ProgramFiles\Docker\Docker\Docker Desktop.exe",
                        "$env:LOCALAPPDATA\Programs\DockerDesktop\Docker Desktop.exe",
                        "$env:LOCALAPPDATA\Docker\Docker Desktop.exe")
        $exe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
        if ($exe) { Start-Process $exe }
    }
    for ($i = 0; $i -lt 90 -and -not (Test-Docker); $i++) { Start-Sleep -Seconds 2 }
    if (-not (Test-Docker)) { throw "Docker Desktop laeuft nicht. Bitte Docker Desktop von Hand starten und erneut versuchen." }
}

Write-Host "Programm wird vorbereitet (beim ersten Mal einige Minuten) ..."
docker compose build --quiet
if ($LASTEXITCODE -ne 0) { throw "Build fehlgeschlagen." }

Write-Host "Sicherheitspruefung ..."
docker compose run --rm kitb check-safety
if ($LASTEXITCODE -ne 0) { throw "Sicherheitspruefung fehlgeschlagen - es wird nichts gestartet." }

Write-Host "Bot und Uebersicht werden gestartet ..."
docker compose --profile paper-trend --profile dashboard up -d paper-trend-d1 dashboard
if ($LASTEXITCODE -ne 0) { throw "Start fehlgeschlagen." }

$url = "http://127.0.0.1:8501"
for ($i = 0; $i -lt 60; $i++) {
    try { Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 2 *> $null; break } catch { Start-Sleep -Seconds 1 }
}
Start-Process $url
Write-Host "Fertig. Die Uebersicht ist im Browser geoeffnet: $url" -ForegroundColor Green
Write-Host "PC und Docker Desktop eingeschaltet lassen, damit der Bot weiterlaeuft."
