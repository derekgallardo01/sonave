# register_schedule.ps1 — install the data-program schedule on this machine.
#   pwsh tools/register_schedule.ps1          # register both tasks
#   pwsh tools/register_schedule.ps1 -Remove  # uninstall
#
# Tasks (run as the logged-on user, only while logged on — the daily one shows UI):
#   SonaveDailyCaptureCheck  Mon-Fri 09:30  tools/daily_capture_check.ps1
#   SonaveSundayRetrain      Sun     18:00  tools/sunday_retrain.ps1 -Pull  (needs the GPU box awake)
param([switch]$Remove)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
# NEVER use (Get-Command pwsh).Source here: for Store installs it resolves to a
# VERSIONED WindowsApps path that dies on the next PowerShell update (tasks then
# fail with 0x80070002 file-not-found). The per-user alias is version-stable.
$pwsh = "$env:LOCALAPPDATA\Microsoft\WindowsApps\pwsh.exe"
if (-not (Test-Path $pwsh)) { $pwsh = (Get-Command pwsh).Source }

if ($Remove) {
    foreach ($n in 'SonaveDailyCaptureCheck', 'SonaveSundayRetrain') {
        Unregister-ScheduledTask -TaskName $n -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "removed $n"
    }
    exit 0
}

$daily = New-ScheduledTaskAction -Execute $pwsh -Argument "-NoProfile -File `"$root\tools\daily_capture_check.ps1`"" -WorkingDirectory $root
$dailyTrig = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 09:30
Register-ScheduledTask -TaskName 'SonaveDailyCaptureCheck' -Action $daily -Trigger $dailyTrig -Force | Out-Null
Write-Host 'registered SonaveDailyCaptureCheck (Mon-Fri 09:30)'

$sunday = New-ScheduledTaskAction -Execute $pwsh -Argument "-NoProfile -File `"$root\tools\sunday_retrain.ps1`" -Pull" -WorkingDirectory $root
$sundayTrig = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 18:00
$settings = New-ScheduledTaskSettingsSet -WakeToRun -ExecutionTimeLimit (New-TimeSpan -Hours 6)
Register-ScheduledTask -TaskName 'SonaveSundayRetrain' -Action $sunday -Trigger $sundayTrig -Settings $settings -Force | Out-Null
Write-Host 'registered SonaveSundayRetrain (Sun 18:00, wakes the box, 6 h limit)'
