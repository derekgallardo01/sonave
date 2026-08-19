# tools/register_scheduler.ps1 — Register Recurring Retraining Task on Windows/Server.
#
# Usage:
#   pwsh tools/register_scheduler.ps1 -Cadence Weekly   # Every Sunday at 00:00 UTC
#   pwsh tools/register_scheduler.ps1 -Cadence Daily    # Every night at 02:00 UTC
#   pwsh tools/register_scheduler.ps1 -Unregister       # Remove scheduled task

param(
    [ValidateSet("Weekly", "Daily")]
    [string]$Cadence = "Weekly",
    [switch]$Unregister
)

$TaskName = "Sonave_Automated_Retrain_Pipeline"
$Root = Split-Path $PSScriptRoot -Parent
$PythonExe = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = (Get-Command python.exe).Source
}

if ($Unregister) {
    Write-Host "Unregistering scheduled task: $TaskName..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "✓ Task unregistered." -ForegroundColor Green
    exit 0
}

Write-Host "Configuring Automated Retraining Task: $TaskName" -ForegroundColor Cyan
Write-Host "Cadence: $Cadence" -ForegroundColor Cyan
Write-Host "Python: $PythonExe" -ForegroundColor Cyan

$Action = New-ScheduledTaskAction -Execute $PythonExe -Argument "src/pipeline/run_pipeline.py --epochs 3 --batch-size 16" -WorkingDirectory $Root

if ($Cadence -eq "Weekly") {
    $Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 00:00
} else {
    $Trigger = New-ScheduledTaskTrigger -Daily -At 02:00
}

$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "Sonave continuous deepfake model training pipeline with multi-corpus ingestion and regression gates." -Force

Write-Host "✓ Scheduled task '$TaskName' registered successfully ($Cadence)!" -ForegroundColor Green
