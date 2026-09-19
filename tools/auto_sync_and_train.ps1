# tools/auto_sync_and_train.ps1 — Single-command automated pipeline runner.
#
# Usage:
#   pwsh tools/auto_sync_and_train.ps1
#   pwsh tools/auto_sync_and_train.ps1 -Fast            # skips full XLS-R retrain, syncs HF + ensemble
#   pwsh tools/auto_sync_and_train.ps1 -SkipPull        # runs without checking remote Railway server

param(
    [switch]$Fast,
    [switch]$SkipPull
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    $py = (Get-Command python.exe).Source
}

Write-Host "`n=== Sonave Automated Sync & Retrain ===" -ForegroundColor Cyan
Write-Host "Project Root: $root"
Write-Host "Python Executable: $py"

$extraArgs = @()
if ($Fast) {
    $extraArgs += "--skip-xlsr"
}
if ($SkipPull) {
    $extraArgs += "--skip-pull"
}

& $py src/pipeline/auto_sync_and_train.py @extraArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nPipeline failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n✓ Automated pipeline finished successfully." -ForegroundColor Green
