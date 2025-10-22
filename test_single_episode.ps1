# Test single episode to diagnose why episodes don't finish naturally
# This will run just ONE episode and show detailed logs

$env:EXPERIMENT_ID="experiment_10_spec_real_debug"

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "Single Episode Diagnostic Test" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will run 1 episode to understand:" -ForegroundColor Yellow
Write-Host "  - Why episodes take 350+ seconds" -ForegroundColor Yellow
Write-Host "  - How many cloudlets are completed" -ForegroundColor Yellow
Write-Host "  - Whether episodes terminate or truncate" -ForegroundColor Yellow
Write-Host ""

# Activate virtual environment and run
& "drl-manager\.venv\Scripts\python.exe" drl-manager/mnt/entrypoint.py
