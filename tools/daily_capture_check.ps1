# daily_capture_check.ps1 — Mon-Fri morning nudge for the data program.
# Queries production's data odometer; if no capture landed in the last 48 h,
# pops a reminder so the bot actually gets into today's meetings.
# Registered by tools/register_schedule.ps1 (runs as the logged-on user).

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$tok = ((Get-Content (Join-Path $root '.env') | Select-String '^SONAVE_API_TOKEN=').Line -replace '^SONAVE_API_TOKEN=','')
try {
    $d = Invoke-RestMethod 'https://sonave-production-3ca2.up.railway.app/api/data_progress' `
        -Headers @{ 'X-Sonave-Token' = $tok } -TimeoutSec 30
} catch {
    Write-Host "data check failed: $($_.Exception.Message)"; exit 0   # never nag on network errors
}
$ageH = if ($d.last_capture_ts) { [int](((Get-Date -AsUTC) - (Get-Date '1970-01-01').AddSeconds($d.last_capture_ts)).TotalHours) } else { 9999 }
Write-Host ("captured: {0} h across {1} sessions - last capture {2} h ago" -f $d.hours, $d.sessions, $ageH)
if ($ageH -gt 48) {
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show(
        ("No Sonave capture in {0} h.`n`nPut the bot in today's meeting (announce recording).`nProgress: {1} h of {2} h toward M1." -f $ageH, $d.hours, $d.m1_target_hours),
        'Sonave data program', 'OK', 'Information') | Out-Null
}
