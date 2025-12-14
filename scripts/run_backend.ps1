$activate = ".venv\\Scripts\\Activate.ps1"
if (-Not (Test-Path $activate)) { throw "Activate virtualenv first by running scripts/setup_env.ps1" }
. $activate

Write-Host "Starting FastAPI backend on http://127.0.0.1:8000" -ForegroundColor Cyan
uvicorn app.backend.main:app --reload
