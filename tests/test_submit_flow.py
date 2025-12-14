import json
from fastapi.testclient import TestClient

from app.backend.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_submit_with_mocks(monkeypatch):
    # Mock API key env
    monkeypatch.setenv("API_KEY", "testkey")

    # Mock agents to avoid external calls
    from app.backend import agents

    def mock_validate(payload):
        return {"ok": True, "message": "ok", "profiles": []}

    def mock_plan(payload):
        return {"task_type": "test", "steps": [{"id": "1", "action": "noop", "params": {}}], "required_columns": [], "expected_outputs": [], "failure_modes": []}

    def mock_exec(plan, payload):
        class R:
            ok = True
            observations = ["noop"]
            outputs = {}
        return R()

    def mock_response(payload):
        return {"summary": "done"}

    monkeypatch.setattr(agents, "validate_submission", mock_validate)
    monkeypatch.setattr(agents, "plan_analysis", mock_plan)
    monkeypatch.setattr(agents, "run_execution", mock_exec)
    monkeypatch.setattr(agents, "generate_response", mock_response)

    payload = {
        "request_type": "Price Strategy",
        "question": "What is optimal pricing?",
        "priority": "normal",
        "deep_analysis": False,
    }
    resp = client.post("/api/form/submit", json=payload, headers={"X-API-Key": "testkey"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] in {"pending", "done", "error"}
    assert "id" in body

    # Poll result
    rid = body["id"]
    r2 = client.get(f"/api/form/result/{rid}")
    assert r2.status_code == 200
