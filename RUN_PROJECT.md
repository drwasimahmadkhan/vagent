# How to Run the CSV Analysis Multi-Agent System

## Prerequisites
- Python 3.10+ installed
- Virtual environment already set up (.venv exists)
- Dependencies already installed

## Quick Start

### Step 1: Configure Environment Variables
Edit the `.env` file and add your Anthropic API key:
```
ANTHROPIC_API_KEY=sk-ant-api03-YOUR_ACTUAL_API_KEY_HERE
```

### Step 2: Run the Backend (Terminal 1)

**Using PowerShell:**
```powershell
# Navigate to project directory
cd D:\Senarios\vagent

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run backend
uvicorn app.backend.main:app --reload --host 127.0.0.1 --port 8000
```

**Using Command Prompt:**
```cmd
cd D:\Senarios\vagent
.venv\Scripts\activate.bat
uvicorn app.backend.main:app --reload --host 127.0.0.1 --port 8000
```

**Using PowerShell Script:**
```powershell
.\scripts\run_backend.ps1
```

### Step 3: Run the Frontend (Terminal 2 - New Window)

**Using PowerShell:**
```powershell
# Navigate to project directory
cd D:\Senarios\vagent

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run frontend
streamlit run app/frontend/app.py
```

**Using Command Prompt:**
```cmd
cd D:\Senarios\vagent
.venv\Scripts\activate.bat
streamlit run app/frontend/app.py
```

**Using PowerShell Script:**
```powershell
.\scripts\run_frontend.ps1
```

## Access the Application

- **Frontend (Streamlit UI):** http://localhost:8501
- **Backend API:** http://127.0.0.1:8000
- **API Documentation:** http://127.0.0.1:8000/docs

## Testing the System

### Option 1: Run All Tests
```powershell
# Activate venv first
.\.venv\Scripts\Activate.ps1

# Run tests
pytest tests/ -v
```

### Option 2: Test Specific Scenarios
```powershell
pytest tests/test_scenarios.py -v
```

### Option 3: Run Demo Script
```powershell
python demo_working.py
```

## Troubleshooting

### Issue: "ANTHROPIC_API_KEY not found"
**Solution:** Make sure you've added your API key to the `.env` file

### Issue: "Module not found"
**Solution:** Install dependencies
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue: "Database not found"
**Solution:** Initialize the database
```powershell
python -c "from app.database.db import init_db; init_db()"
```

### Issue: Port already in use
**Solution:** Kill the process using the port
```powershell
# For backend (port 8000)
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# For frontend (port 8501)
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

## Project Structure
```
D:\Senarios\vagent/
├── app/
│   ├── backend/           # FastAPI backend
│   │   ├── agents/        # AI agents (validation, planning, execution, response)
│   │   ├── routers/       # API endpoints
│   │   ├── tools/         # Utility tools
│   │   └── main.py        # FastAPI app entry point
│   ├── database/          # Database models and CRUD
│   └── frontend/          # Streamlit UI
│       └── app.py         # Frontend entry point
├── scripts/               # PowerShell scripts
├── tests/                 # Test suite
├── uploads/               # User-uploaded CSV files
├── audit_logs/            # Security audit logs
├── .env                   # Environment variables
├── db.sqlite              # SQLite database
└── requirements.txt       # Python dependencies
```

## Usage Flow

1. **Start Backend** (Terminal 1)
2. **Start Frontend** (Terminal 2)
3. **Open Browser** to http://localhost:8501
4. **Upload CSV** file (≤10MB, <50k rows)
5. **Enter Question** (e.g., "What is the average revenue by region?")
6. **Submit** and wait for results (~5-60 seconds)
7. **View Results** with confidence score, insights, and recommendations

## Mock Mode (No API Key Required)

To test without API costs:

1. Edit `.env`:
   ```
   MOCK_EXECUTION=1
   ```

2. Run backend and frontend as usual

3. The system will use mock responses instead of calling Claude API

## Advanced: Docker Deployment

```bash
# Build image
docker build -t csv-agent .

# Run container
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... csv-agent
```

## Support

For issues or questions, refer to:
- `flow.txt` - Complete system documentation
- API docs at http://127.0.0.1:8000/docs
- Test files in `tests/` directory
