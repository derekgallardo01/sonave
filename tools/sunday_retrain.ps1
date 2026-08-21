# sunday_retrain.ps1 — the weekly retrain ritual, one command.
#
#   pwsh tools/sunday_retrain.ps1                      # use local captures
#   pwsh tools/sunday_retrain.ps1 -Pull                # pull new captures from prod first
#
# pull -> window -> retrain -> validate (both sides) -> benchmark -> regression gate
# -> metrics.json -> commit + push (CI deploys Modal and verifies the live version).
# Any failing step aborts BEFORE the commit, so a bad checkpoint never ships.
# Optional schedule (box must be awake):
#   Register-ScheduledJob -Name SonaveRetrain -Trigger (New-JobTrigger -Weekly -DaysOfWeek Sunday -At 18:00) `
#     -ScriptBlock { pwsh i:\Projects\sonave\tools\sunday_retrain.ps1 -Pull }
param([switch]$Pull)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
$stamp = Get-Date -Format 'yyyy-MM-dd'

function Step($name, $block) {
    Write-Host "`n=== $name ===" -ForegroundColor Green
    & $block
    if ($LASTEXITCODE -ne 0) { throw "step failed: $name" }
}

if ($Pull) {
    Step 'pull captures' { & $py src/pull_captures.py https://sonave-production-3ca2.up.railway.app }
}

# Training writes the new checkpoint over models/sonave_xlsr_meet IN PLACE, so
# ANY failure after that point (split validation, benchmark, regression gate,
# fast suite) must preserve the candidate and restore the deployed checkpoint —
# a bare failure left a rejected model sitting on the deployed path twice.
try {
    Step 'retrain + validate'   { & $py src/retrain_from_captures.py }
    Step 'benchmark'            { & $py src/eval_xlsr.py --model models/sonave_xlsr_meet }
    Step 'regression gate'      { & $py -m pytest -m gpu tests/test_model_regression.py -q }
} catch {
    $cand = "models\sonave_xlsr_meet_candidate_$stamp"
    New-Item -ItemType Directory -Force $cand | Out-Null
    Copy-Item models\sonave_xlsr_meet\* $cand\ -Force
    if (Test-Path models\training_lineage.json) {
        Copy-Item models\training_lineage.json "$cand\training_lineage.json" -Force
    }
    git checkout -- models/sonave_xlsr_meet models/training_lineage.json 2>$null
    Add-Content results/retrain_log.md "- $stamp — retrain attempt FAILED ($($_.Exception.Message)). Candidate preserved at $cand; deployed checkpoint restored."
    Write-Host "`nHELD/FAILED — candidate preserved at $cand, deployed checkpoint restored." -ForegroundColor Yellow
    throw
}
Step 'write metrics'        { & $py tools/write_metrics.py }
Step 'fast suite'           { & $py -m pytest -q }

$catch = (Get-Content railway/model_metrics.json | ConvertFrom-Json).unseen_tools_catch_pct
Add-Content results/retrain_log.md "- $stamp — retrained + gates passed; unseen-tools catch $catch%"

Step 'commit + push' {
    git add models/sonave_xlsr_meet railway/model_metrics.json results/retrain_log.md
    git commit -m "model: weekly retrain $stamp (unseen-tools catch $catch%)"
    git push
}
Write-Host "`nDone — CI is deploying the new checkpoint to Modal." -ForegroundColor Green
