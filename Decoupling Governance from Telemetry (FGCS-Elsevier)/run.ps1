# Define the names of your Python scripts
$SimScript = "simUp.py"
$PlotScript = "plots.py"

Write-Host "----------------------------------------" -ForegroundColor Cyan
Write-Host "Starting simulation ($SimScript)..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Cyan

# Run the simulation script
python $SimScript

# Check if the simulation finished successfully (Exit Code 0)
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n----------------------------------------" -ForegroundColor Cyan
    Write-Host "Simulation complete. Generating plots ($PlotScript)..." -ForegroundColor Green
    Write-Host "----------------------------------------" -ForegroundColor Cyan
    
    # Run the plotting script
    python $PlotScript
} else {
    Write-Warning "`n[ERROR] simUp.py failed or was interrupted. Plotting aborted."
}