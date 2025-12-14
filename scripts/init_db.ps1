$activate = ".venv\Scripts\Activate.ps1"
if (-Not (Test-Path $activate)) { throw "Activate virtualenv first or run scripts/setup_env.ps1" }
. $activate

Write-Host "Initializing SQLite schema..." -ForegroundColor Cyan
python - <<'PY'
from app.database.db import Base, engine
import app.database.models  # noqa: F401
Base.metadata.create_all(bind=engine)
print("Tables created/verified")
PY
