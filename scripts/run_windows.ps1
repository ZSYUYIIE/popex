$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

& "$PSScriptRoot\check_windows.ps1"

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
}

Write-Host "Installing or updating PopEx dependencies..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -e ".[dev]"

New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot "data") | Out-Null

Write-Host ""
Write-Host "Starting PopEx at http://localhost:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop it." -ForegroundColor DarkGray
& $venvPython -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
