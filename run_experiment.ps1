# RL-CloudSim Load Balancer - Experiment Runner
# Usage: .\run_experiment.ps1 -ExperimentId "experiment_2"

param(
    [string]$ExperimentId = "experiment_1",
    [string]$ConfigFile = "config.yml"
)

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " RL-CloudSim Experiment Runner" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if in project root
if (-not (Test-Path "config.yml")) {
    Write-Host "[ERROR] Please run from project root directory" -ForegroundColor Red
    Write-Host "Current: $(Get-Location)" -ForegroundColor Yellow
    exit 1
}

# Check config file
if (-not (Test-Path $ConfigFile)) {
    Write-Host "[ERROR] Config file not found: $ConfigFile" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Config File: $ConfigFile" -ForegroundColor Green
Write-Host "[INFO] Experiment ID: $ExperimentId`n" -ForegroundColor Green

# Check virtual environment
$venvPath = "drl-manager\venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvPath)) {
    Write-Host "[ERROR] Virtual environment not found" -ForegroundColor Red
    Write-Host "Please create it first:" -ForegroundColor Yellow
    Write-Host "  cd drl-manager; python -m venv venv" -ForegroundColor White
    exit 1
}

# Check Gateway (optional)
Write-Host "[INFO] Checking Java Gateway..." -ForegroundColor Yellow
$gatewayRunning = $false
try {
    $connection = Test-NetConnection -ComputerName localhost -Port 25333 -WarningAction SilentlyContinue -ErrorAction SilentlyContinue
    if ($connection.TcpTestSucceeded) {
        $gatewayRunning = $true
        Write-Host "[OK] Java Gateway is running`n" -ForegroundColor Green
    }
} catch {
    # Ignore
}

if (-not $gatewayRunning) {
    Write-Host "[WARN] Java Gateway not detected on port 25333" -ForegroundColor Yellow
    Write-Host "Please start it in another terminal:" -ForegroundColor Cyan
    Write-Host "  cd cloudsimplus-gateway" -ForegroundColor White
    Write-Host "  .\gradlew.bat run`n" -ForegroundColor White
}

# Set environment variables
$env:CONFIG_FILE = $ConfigFile
$env:EXPERIMENT_ID = $ExperimentId

Write-Host "[INFO] Starting experiment...`n" -ForegroundColor Cyan

# Activate venv and run
& $venvPath
python .\drl-manager\mnt\entrypoint.py

# Check result
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] Experiment completed!" -ForegroundColor Green
    
    # Find latest log
    $logDirs = Get-ChildItem -Path "logs\*\*" -Directory -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1
    
    if ($logDirs) {
        Write-Host "[INFO] Results saved to: $($logDirs.FullName)" -ForegroundColor Cyan
        
        Write-Host "`nKey files:" -ForegroundColor Yellow
        @("best_model.zip", "final_model.zip", "monitor.csv", "progress.csv") | ForEach-Object {
            $f = Join-Path $logDirs.FullName $_
            if (Test-Path $f) {
                $size = [math]::Round((Get-Item $f).Length / 1KB, 2)
                Write-Host "  * $_ ($size KB)" -ForegroundColor Green
            }
        }
    }
    
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "  - View training: tensorboard --logdir=logs" -ForegroundColor White
    Write-Host "  - Check logs: Get-Content logs\...\current_run.log`n" -ForegroundColor White
} else {
    Write-Host "`n[ERROR] Experiment failed (exit code: $LASTEXITCODE)" -ForegroundColor Red
    Write-Host "Check log files for details`n" -ForegroundColor Yellow
}

Write-Host "========================================`n" -ForegroundColor Cyan

