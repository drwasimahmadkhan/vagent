import pytest

from app.backend.agents.validation import validate_submission


def test_validate_submission_empty_csv(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("")

    res = validate_submission({"file_paths": [str(csv_path)], "request_type": "Price Strategy", "question": "Q"})
    assert res["ok"] is False or any("empty" in issue.lower() for issue in res["checks"][0]["issues"])


def test_validate_submission_no_files():
    res = validate_submission({"file_paths": [], "request_type": "Price Strategy", "question": "Q"})
    assert res["ok"] is True
