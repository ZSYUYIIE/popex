param(
    [ValidateSet("check", "validate")]
    [string]$Mode = "check",
    [switch]$AllowModelDownload,
    [string]$Python = "python",
    [string]$Worker = "",
    [string]$RuntimeLock = "",
    [string]$CacheRoot = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$profile = "windows-x86_64-cpu-cpython313"
$runtimeDir = if ($env:POPEX_DEMUCS_RUNTIME_DIR) {
    $env:POPEX_DEMUCS_RUNTIME_DIR
} else {
    Join-Path $env:LOCALAPPDATA "PopEx\runtimes\$profile"
}
if (-not $Worker) { $Worker = Join-Path $runtimeDir "venv\Scripts\popex-demucs-worker.exe" }
if (-not $RuntimeLock) { $RuntimeLock = Join-Path $runtimeDir "runtime-lock.json" }
if (-not $CacheRoot) { $CacheRoot = Join-Path $env:LOCALAPPDATA "PopEx\models\$profile" }

$common = @(
    "--worker", $Worker,
    "--runtime-lock", $RuntimeLock,
    "--cache-root", $CacheRoot,
    "--expected-profile", $profile,
    "--device", "cpu"
)
$doctor = Join-Path $repoRoot "scripts\popex_separation_doctor.py"

if ($Mode -eq "validate") {
    if (-not $AllowModelDownload) {
        throw "Validation requires the explicit -AllowModelDownload switch."
    }
    & $Python $doctor validate --allow-model-download @common
} else {
    & $Python $doctor check @common
}
exit $LASTEXITCODE
