# Run Experiment 10 - SPEC Real Servers Training
# Full training with 120,000 timesteps

$env:EXPERIMENT_ID="experiment_10_spec_real"

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "Experiment 10: SPEC Real Servers" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  - 28 heterogeneous SPEC servers" -ForegroundColor White
Write-Host "  - 26 VMs (12 Small, 8 Medium, 6 Large)" -ForegroundColor White
Write-Host "  - 164 cloudlets workload" -ForegroundColor White
Write-Host "  - 120,000 timesteps (~344 episodes)" -ForegroundColor White
Write-Host "  - Energy-aware optimization" -ForegroundColor White
Write-Host ""
Write-Host "Starting training..." -ForegroundColor Green
Write-Host ""

# Use virtual environment Python
& "drl-manager\.venv\Scripts\python.exe" drl-manager/mnt/entrypoint.py
