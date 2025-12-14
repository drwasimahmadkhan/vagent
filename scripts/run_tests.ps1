$activate = ".venv\Scripts\Activate.ps1"
if (-Not (Test-Path $activate)) { 
    Write-Host "Virtual environment not found. Run scripts/setup_env.ps1 first." -ForegroundColor Red
    exit 1 
}
. $activate

Write-Host "Running unit tests..." -ForegroundColor Cyan
pytest -v tests/

if ($LASTEXITCODE -ne 0) {
    Write-Host "Unit tests failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`nUnit tests passed! ✓" -ForegroundColor Green
Write-Host "`nTo run real-world scenarios:" -ForegroundColor Yellow
Write-Host "  1. Start backend: scripts\run_backend.ps1" -ForegroundColor Yellow
Write-Host "  2. Run scenarios: python test.py" -ForegroundColor Yellow
