$ErrorActionPreference = "Stop"

function Test-Command {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$InstallHint
    )

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $command) {
        Write-Host "[missing] $Name" -ForegroundColor Red
        Write-Host "          $InstallHint" -ForegroundColor Yellow
        return $false
    }

    Write-Host "[ok]      $Name -> $($command.Source)" -ForegroundColor Green
    return $true
}

Write-Host "PopEx Windows dependency check" -ForegroundColor Cyan
Write-Host ""

$allPresent = $true
$allPresent = (
    Test-Command `
        -Name "python" `
        -InstallHint "Install Python 3.10+ and enable Add Python to PATH."
) -and $allPresent
$allPresent = (
    Test-Command `
        -Name "ffmpeg" `
        -InstallHint "Install FFmpeg, then reopen PowerShell so PATH is refreshed."
) -and $allPresent
$allPresent = (
    Test-Command `
        -Name "ffprobe" `
        -InstallHint "ffprobe is included with normal FFmpeg distributions."
) -and $allPresent
$allPresent = (
    Test-Command `
        -Name "node" `
        -InstallHint "Install Node.js LTS for reliable YouTube extraction support."
) -and $allPresent

if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonVersion = python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
    $pythonSupported = python -c "import sys; print('yes' if sys.version_info >= (3, 10) else 'no')"
    Write-Host "Python version: $pythonVersion"
    if ($pythonSupported -ne "yes") {
        Write-Host "[invalid] Python 3.10 or newer is required." -ForegroundColor Red
        $allPresent = $false
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$dataDir = Join-Path $repoRoot "data"
try {
    New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
    $probe = Join-Path $dataDir ".popex-write-test"
    Set-Content -Path $probe -Value "ok" -Encoding UTF8
    Remove-Item $probe -Force
    Write-Host "[ok]      writable data directory -> $dataDir" -ForegroundColor Green
} catch {
    Write-Host "[invalid] PopEx cannot write to $dataDir" -ForegroundColor Red
    $allPresent = $false
}

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    & $venvPython -c @"
import librosa
import numpy
import scipy
import soundfile
print('[ok]      local audio-analysis Python packages')
"@
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[invalid] Re-run scripts\run_windows.ps1 to install the audio-analysis dependencies." -ForegroundColor Red
        $allPresent = $false
    }
} else {
    Write-Host "[info]    .venv is not created yet; run_windows.ps1 will install analysis packages." -ForegroundColor DarkGray
}

if (-not $allPresent) {
    Write-Host ""
    Write-Host "Install or fix the missing dependencies, reopen PowerShell, and run this check again." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "All required host dependencies and local configuration are available." -ForegroundColor Green
