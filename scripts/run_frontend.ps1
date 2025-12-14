$activate = ".venv\\Scripts\\Activate.ps1"
if (-Not (Test-Path $activate)) { throw "Activate virtualenv first by running scripts/setup_env.ps1" }
. $activate

Write-Host "Starting Streamlit frontend on http://localhost:8501" -ForegroundColor Cyan
streamlit run app/frontend/app.py
