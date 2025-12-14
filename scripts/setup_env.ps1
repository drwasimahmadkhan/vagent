param(
    [string]$PythonPath = "py -3.10"
)

Write-Host "Creating virtual environment (.venv) with Python 3.10..." -ForegroundColor Cyan
Invoke-Expression "$PythonPath -m venv .venv"

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
$activate = ".venv\\Scripts\\Activate.ps1"
if (-Not (Test-Path $activate)) { throw "Activation script not found: $activate" }

. $activate

Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host "Installing requirements..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "Done. To activate later, run:`n . .venv\\Scripts\\Activate.ps1" -ForegroundColor Green
