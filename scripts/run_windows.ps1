param(
    # Install or update dependencies, then stop without starting the server.
    [switch]$InstallOnly
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

# The check prints what is missing; stop here rather than start a server
# that cannot prepare recordings.
& "$PSScriptRoot\check_windows.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "PopEx was not started. Fix the items marked above, reopen PowerShell, and run this script again." -ForegroundColor Red
    exit 1
}

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) {
        Write-Host "The Python virtual environment could not be created in $repoRoot\.venv." -ForegroundColor Red
        exit 1
    }
}

Write-Host "Installing or updating PopEx dependencies..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "pip could not be updated; check the network connection and try again." -ForegroundColor Red
    exit 1
}
& $venvPython -m pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) {
    Write-Host "PopEx dependencies could not be installed; the messages above say which package failed." -ForegroundColor Red
    exit 1
}

New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot "data") | Out-Null

if ($InstallOnly) {
    Write-Host "PopEx is installed. Run this script again without -InstallOnly to start it." -ForegroundColor Green
    exit 0
}

Write-Host ""
Write-Host "Starting PopEx at http://localhost:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop it." -ForegroundColor DarkGray
& $venvPython -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
