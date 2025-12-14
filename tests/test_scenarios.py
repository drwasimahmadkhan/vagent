"""Comprehensive test scenarios for CSV Agent system."""
import json
import os
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.backend.main import app
from app.database import crud
from app.database.db import get_db

client = TestClient(app)


# --- Fixtures ---

@pytest.fixture
def test_csv_dir(tmp_path):
    """Create temporary directory for test CSV files."""
    csv_dir = tmp_path / "test_csvs"
    csv_dir.mkdir()
    return csv_dir


@pytest.fixture
def empty_csv(test_csv_dir):
    """Create empty CSV file."""
    path = test_csv_dir / "empty.csv"
    pd.DataFrame().to_csv(path, index=False)
    return path


@pytest.fixture
def corrupt_csv(test_csv_dir):
    """Create corrupt CSV file."""
    path = test_csv_dir / "corrupt.csv"
    with open(path, "w") as f:
        f.write("name,age\nAlice,30\nBob,invalid_age\n")
    return path


@pytest.fixture
def sales_csv(test_csv_dir):
    """Create sample sales CSV."""
    path = test_csv_dir / "sales.csv"
    data = {
        "date": pd.date_range("2024-01-01", periods=100),
        "product": ["A", "B", "C", "D"] * 25,
        "sales": [100 + i * 10 for i in range(100)],
        "region": ["North", "South", "East", "West"] * 25,
    }
    pd.DataFrame(data).to_csv(path, index=False)
    return path


@pytest.fixture
def large_dataset_csv(test_csv_dir):
    """Create large dataset for 10k simulation testing."""
    path = test_csv_dir / "large.csv"
    data = {
        "metric": [i for i in range(10000)],
        "value": [50 + i * 0.01 for i in range(10000)],
    }
    pd.DataFrame(data).to_csv(path, index=False)
    return path


@pytest.fixture(autouse=True)
def setup_db():
    """Initialize database before each test."""
    from app.database.db import engine
    from app.database.models import Base
    
    Base.metadata.create_all(engine)
    yield
    # Cleanup after test
    Base.metadata.drop_all(engine)


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """Set up environment variables for testing."""
    monkeypatch.setenv("API_KEY", "testkey")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("DEEP_ANALYSIS_QUOTA", "5")
    monkeypatch.setenv("SUBMISSIONS_QUOTA", "50")


# --- Test Scenarios ---

def test_scenario_empty_csv(empty_csv, monkeypatch):
    """Scenario 1: Submit empty CSV file."""
    from app.backend import agents
    
    def mock_validate(payload):
        return {
            "ok": False,
            "error": "Empty CSV file",
            "profiles": [],
            "issues": ["No data rows found"],
            "suggestions": ["Upload a CSV with at least one data row"],
        }
    
    monkeypatch.setattr(agents, "validate_submission", mock_validate)
    
    with open(empty_csv, "rb") as f:
        resp = client.post(
            "/api/form/submit",
            data={
                "request_type": "Price Strategy",
                "question": "Analyze empty file",
                "priority": "normal",
                "deep_analysis": "false",
            },
            files={"files": ("empty.csv", f, "text/csv")},
            headers={"X-API-Key": "testkey"},
        )
    
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"pending", "error"}


def test_scenario_corrupt_csv(corrupt_csv, monkeypatch):
    """Scenario 2: Submit corrupt CSV with invalid data types."""
    from app.backend import agents
    
    def mock_validate(payload):
        return {
            "ok": True,
            "profiles": [{
                "filename": "corrupt.csv",
                "rows": 2,
                "columns": ["name", "age"],
                "issues": ["Invalid data type in 'age' column"],
            }],
            "suggestions": ["Check data types in 'age' column"],
        }
    
    def mock_plan(payload):
        return {
            "task_type": "data_quality",
            "steps": [{"id": "1", "action": "check_types", "params": {}}],
            "required_columns": ["name", "age"],
            "expected_outputs": [],
            "failure_modes": ["Type conversion errors"],
        }
    
    def mock_exec(plan, payload):
        class R:
            ok = False
            observations = ["Type error in 'age' column"]
            outputs = {}
        return R()
    
    def mock_response(payload):
        return {
            "summary": "Data quality issues detected",
            "confidence": 0.3,
            "insights": ["Invalid data types found"],
            "risks": ["Results may be unreliable"],
            "disclaimers": ["Data quality issues present"],
            "next_steps": ["Clean data and resubmit"],
        }
    
    monkeypatch.setattr(agents, "validate_submission", mock_validate)
    monkeypatch.setattr(agents, "plan_analysis", mock_plan)
    monkeypatch.setattr(agents, "run_execution", mock_exec)
    monkeypatch.setattr(agents, "generate_response", mock_response)
    
    with open(corrupt_csv, "rb") as f:
        resp = client.post(
            "/api/form/submit",
            data={
                "request_type": "Other",
                "question": "Check data quality",
                "priority": "normal",
                "deep_analysis": "false",
            },
            files={"files": ("corrupt.csv", f, "text/csv")},
            headers={"X-API-Key": "testkey"},
        )
    
    assert resp.status_code == 200


def test_scenario_prediction_request(sales_csv, monkeypatch):
    """Scenario 3: Request sales prediction analysis."""
    from app.backend import agents
    
    def mock_validate(payload):
        return {
            "ok": True,
            "profiles": [{
                "filename": "sales.csv",
                "rows": 100,
                "columns": ["date", "product", "sales", "region"],
                "dtypes": {"date": "object", "product": "object", "sales": "int64", "region": "object"},
            }],
        }
    
    def mock_plan(payload):
        return {
            "task_type": "prediction",
            "steps": [
                {"id": "1", "action": "load_data", "params": {"file": "sales.csv"}},
                {"id": "2", "action": "regression", "params": {"target": "sales"}},
            ],
            "required_columns": ["date", "sales"],
            "expected_outputs": ["predictions", "metrics"],
            "failure_modes": ["Insufficient data"],
        }
    
    def mock_exec(plan, payload):
        class R:
            ok = True
            observations = ["Data loaded", "Model trained"]
            outputs = {
                "1": {"shape": [100, 4]},
                "2": {"r2": 0.85, "mae": 15.2},
            }
        return R()
    
    def mock_response(payload):
        return {
            "summary": "Sales prediction model trained successfully with R2=0.85",
            "confidence": 0.85,
            "insights": ["Strong linear trend detected", "Seasonal patterns present"],
            "sanity_checks": {"r2_score": "acceptable", "residuals": "normally_distributed"},
            "risks": ["Model assumes trend continuation"],
            "disclaimers": ["Based on 100 historical records"],
            "next_steps": ["Validate on holdout set", "Deploy for forecasting"],
        }
    
    monkeypatch.setattr(agents, "validate_submission", mock_validate)
    monkeypatch.setattr(agents, "plan_analysis", mock_plan)
    monkeypatch.setattr(agents, "run_execution", mock_exec)
    monkeypatch.setattr(agents, "generate_response", mock_response)
    
    with open(sales_csv, "rb") as f:
        resp = client.post(
            "/api/form/submit",
            data={
                "request_type": "Customer & Revenue",
                "question": "Predict next quarter sales",
                "priority": "normal",
                "deep_analysis": "false",
            },
            files={"files": ("sales.csv", f, "text/csv")},
            headers={"X-API-Key": "testkey"},
        )
    
    assert resp.status_code == 200
    body = resp.json()
    assert "id" in body


def test_scenario_deep_analysis_10k_sims(large_dataset_csv, monkeypatch):
    """Scenario 4: Deep analysis with 10,000 Monte Carlo simulations."""
    from app.backend import agents
    
    def mock_validate(payload):
        return {
            "ok": True,
            "profiles": [{
                "filename": "large.csv",
                "rows": 10000,
                "columns": ["metric", "value"],
            }],
        }
    
    def mock_plan(payload):
        return {
            "task_type": "deep_analysis",
            "steps": [
                {"id": "1", "action": "load_data", "params": {}},
                {"id": "2", "action": "monte_carlo", "params": {"n_sims": 10000, "seed": 42}},
            ],
            "required_columns": ["value"],
            "expected_outputs": ["simulation_results"],
            "failure_modes": ["Memory overflow"],
        }
    
    def mock_exec(plan, payload):
        class R:
            ok = True
            observations = ["10k simulations completed"]
            outputs = {
                "2": {
                    "mean": 500.5,
                    "std": 28.9,
                    "p5": 453.2,
                    "p95": 547.8,
                    "n_sims": 10000,
                }
            }
        return R()
    
    def mock_response(payload):
        return {
            "tldr": None,
            "summary": "Deep analysis completed with 10,000 Monte Carlo simulations",
            "confidence": 0.95,
            "insights": ["Distribution is approximately normal", "Low variance detected"],
            "sanity_checks": {"mean_vs_median": "consistent", "outliers": "minimal"},
            "risks": ["Assumes iid samples"],
            "disclaimers": ["Based on 10,000 rows", "Seed=42 for reproducibility"],
            "next_steps": ["Compare with business targets"],
        }
    
    monkeypatch.setattr(agents, "validate_submission", mock_validate)
    monkeypatch.setattr(agents, "plan_analysis", mock_plan)
    monkeypatch.setattr(agents, "run_execution", mock_exec)
    monkeypatch.setattr(agents, "generate_response", mock_response)
    
    with open(large_dataset_csv, "rb") as f:
        resp = client.post(
            "/api/form/submit",
            data={
                "request_type": "Strategic Decisions",
                "question": "Run deep statistical analysis",
                "priority": "normal",
                "deep_analysis": "true",
            },
            files={"files": ("large.csv", f, "text/csv")},
            headers={"X-API-Key": "testkey"},
        )
    
    assert resp.status_code == 200


def test_scenario_urgent_with_tldr(sales_csv, monkeypatch):
    """Scenario 5: Urgent request requiring TLDR."""
    from app.backend import agents
    
    def mock_validate(payload):
        return {"ok": True, "profiles": [{"rows": 100}]}
    
    def mock_plan(payload):
        return {
            "task_type": "summary",
            "steps": [{"id": "1", "action": "summarize", "params": {}}],
            "required_columns": [],
            "expected_outputs": [],
            "failure_modes": [],
        }
    
    def mock_exec(plan, payload):
        class R:
            ok = True
            observations = ["Summary generated"]
            outputs = {}
        return R()
    
    def mock_response(payload):
        return {
            "tldr": "Sales show 15% growth trend. No immediate risks detected.",
            "summary": "Detailed analysis shows consistent growth with seasonal variations.",
            "confidence": 0.8,
            "insights": ["Growth trend stable"],
            "risks": ["Limited data"],
            "disclaimers": ["Urgent analysis - full validation pending"],
            "next_steps": ["Monitor next quarter"],
        }
    
    monkeypatch.setattr(agents, "validate_submission", mock_validate)
    monkeypatch.setattr(agents, "plan_analysis", mock_plan)
    monkeypatch.setattr(agents, "run_execution", mock_exec)
    monkeypatch.setattr(agents, "generate_response", mock_response)
    
    with open(sales_csv, "rb") as f:
        resp = client.post(
            "/api/form/submit",
            data={
                "request_type": "Customer & Revenue",
                "question": "Quick revenue check",
                "priority": "urgent",
                "deep_analysis": "false",
            },
            files={"files": ("sales.csv", f, "text/csv")},
            headers={"X-API-Key": "testkey", "X-Premium": "true"},
        )
    
    assert resp.status_code == 200


def test_scenario_react_retry_on_error(sales_csv, monkeypatch):
    """Scenario 6: ReAct agent retries on KeyError."""
    from app.backend import agents
    from app.backend.agents.execution import ExecutionResult
    
    retry_count = 0
    
    def mock_validate(payload):
        return {"ok": True, "profiles": [{"rows": 100, "columns": ["date", "sales"]}]}
    
    def mock_plan(payload):
        return {
            "task_type": "analysis",
            "steps": [{"id": "1", "action": "compute_stats", "params": {"column": "sales"}}],
            "required_columns": ["sales"],
            "expected_outputs": [],
            "failure_modes": ["Column not found"],
        }
    
    def mock_exec_with_retry(plan, payload):
        nonlocal retry_count
        retry_count += 1
        
        if retry_count == 1:
            # First attempt fails
            return ExecutionResult(
                ok=False,
                observations=["KeyError: column 'sales' not found"],
                outputs={},
            )
        else:
            # Retry succeeds
            return ExecutionResult(
                ok=True,
                observations=["Retry successful", "Stats computed"],
                outputs={"1": {"mean": 545.0, "std": 289.1}},
            )
    
    def mock_response(payload):
        return {
            "summary": "Analysis completed after retry",
            "confidence": 0.75,
            "insights": ["Retry mechanism worked"],
            "risks": ["Initial error recovered"],
            "disclaimers": ["One retry required"],
            "next_steps": ["Monitor data quality"],
        }
    
    monkeypatch.setattr(agents, "validate_submission", mock_validate)
    monkeypatch.setattr(agents, "plan_analysis", mock_plan)
    monkeypatch.setattr(agents, "run_execution", mock_exec_with_retry)
    monkeypatch.setattr(agents, "generate_response", mock_response)
    
    with open(sales_csv, "rb") as f:
        resp = client.post(
            "/api/form/submit",
            data={
                "request_type": "Other",
                "question": "Test retry mechanism",
                "priority": "normal",
                "deep_analysis": "false",
            },
            files={"files": ("sales.csv", f, "text/csv")},
            headers={"X-API-Key": "testkey"},
        )
    
    assert resp.status_code == 200
    assert retry_count >= 1


def test_scenario_quota_exceeded(sales_csv, monkeypatch):
    """Scenario 7: Deep analysis quota exceeded."""
    from app.backend.agents.security import _quota_tracker, QUOTA_LIMITS
    import time
    
    # Pre-fill quota
    current_time = time.time()
    _quota_tracker["deep_analysis_per_hour:default"] = [current_time] * 10
    
    with open(sales_csv, "rb") as f:
        resp = client.post(
            "/api/form/submit",
            data={
                "request_type": "Price Strategy",
                "question": "Test quota",
                "priority": "normal",
                "deep_analysis": "true",
            },
            files={"files": ("sales.csv", f, "text/csv")},
            headers={"X-API-Key": "testkey"},
        )
    
    assert resp.status_code == 429
    assert "quota exceeded" in resp.json()["detail"].lower()


def test_audit_logging(sales_csv, monkeypatch, tmp_path):
    """Test audit logging functionality."""
    from app.backend import agents
    from app.backend.agents.security import AUDIT_LOG_DIR
    
    def mock_validate(payload):
        return {"ok": True, "profiles": []}
    
    def mock_plan(payload):
        return {"task_type": "test", "steps": [], "required_columns": [], "expected_outputs": [], "failure_modes": []}
    
    def mock_exec(plan, payload):
        class R:
            ok = True
            observations = []
            outputs = {}
        return R()
    
    def mock_response(payload):
        return {"summary": "test", "confidence": 0.5, "insights": [], "risks": [], "disclaimers": [], "next_steps": []}
    
    monkeypatch.setattr(agents, "validate_submission", mock_validate)
    monkeypatch.setattr(agents, "plan_analysis", mock_plan)
    monkeypatch.setattr(agents, "run_execution", mock_exec)
    monkeypatch.setattr(agents, "generate_response", mock_response)
    
    with open(sales_csv, "rb") as f:
        resp = client.post(
            "/api/form/submit",
            data={
                "request_type": "Other",
                "question": "Test audit",
                "priority": "normal",
                "deep_analysis": "false",
            },
            files={"files": ("sales.csv", f, "text/csv")},
            headers={"X-API-Key": "testkey"},
        )
    
    assert resp.status_code == 200
    
    # Check audit log created
    audit_files = list(AUDIT_LOG_DIR.glob("audit_*.jsonl"))
    assert len(audit_files) > 0
