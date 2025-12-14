import base64
import json
import os
import uuid
import logging
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
import aiofiles

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status, Request
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field, validator, ValidationError

from app.database import crud
from app.database.db import get_db
from app.backend.agents import validate_submission, plan_analysis, run_execution, process_submission
from app.backend.agents.security import check_quota, log_audit_event, sanitize_for_logging

log = structlog.get_logger()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(exist_ok=True)

REQUEST_TYPES = {
    "Price Strategy",
    "Growth & Marketing",
    "Hiring & Operations",
    "Customer & Revenue",
    "Strategic Decisions",
    "Other",
}


class RequestForm(BaseModel):
    request_type: str = Field(...)
    question: str = Field(..., max_length=2000)
    context: Optional[str] = Field(None)
    priority: str = Field("normal")
    deep_analysis: bool = Field(False)

    @validator("request_type")
    def validate_request_type(cls, v: str) -> str:
        if v not in REQUEST_TYPES:
            raise ValueError("Invalid request type")
        return v

    @validator("priority")
    def validate_priority(cls, v: str) -> str:
        if v not in {"normal", "urgent"}:
            raise ValueError("Priority must be normal or urgent")
        return v


class SubmitResponse(BaseModel):
    status: str
    id: str
    result: Optional[str]


class ResultResponse(BaseModel):
    id: str
    status: str
    result: Optional[str]


router = APIRouter(prefix="/api/form", tags=["form"])

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: Optional[str] = Depends(API_KEY_HEADER)):
    expected = os.getenv("API_KEY")
    if expected and api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return True


def run_agentic_pipeline(payload: Dict[str, Any]) -> str:
    """Run validation -> planning -> execution and return summary text."""

    validation = validate_submission(payload)
    if not validation.get("ok", True):
        return f"Validation failed: {validation.get('message') or validation.get('checks')}"

    plan = plan_analysis(payload)

    exec_res = run_execution(plan, payload)
    if not exec_res.ok:
        return f"Execution partial/failed: {exec_res.observations}"

    summary = {
        "validation": validation,
        "plan": plan,
        "execution": {
            "observations": exec_res.observations,
            "outputs": exec_res.outputs,
        },
    }
    return json.dumps(summary)


MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_ROWS = 50_000

logging.basicConfig(filename="errors.log", level=logging.ERROR)
val_log = logging.getLogger("validation")
val_handler = logging.FileHandler("validation.log")
val_handler.setLevel(logging.INFO)
val_log.addHandler(val_handler)


def _count_rows_stream(file_path: str) -> int:
    try:
        with open(file_path, "r", newline="", encoding="utf-8") as f:
            return sum(1 for _ in f) - 1  # minus header
    except Exception:
        # fallback to pandas if needed
        try:
            df = pd.read_csv(file_path)
            return int(len(df))
        except Exception:
            return -1


@router.post("/submit")
async def submit_form(
    request: Request,
    request_type: Optional[str] = Form(None),
    question: Optional[str] = Form(None),
    context: Optional[str] = Form(None),
    priority: Optional[str] = Form(None),
    file: UploadFile = File(None, alias="file"),
    files: List[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    """Submit analysis request with strict validations (10 MB, <50k rows). Supports JSON or multipart."""

    # Quota guard
    if not check_quota("submissions_per_hour", user_id="default"):
        raise HTTPException(status_code=429, detail="Submission quota exceeded. Please try again later.")

    try:
        # Detect payload type
        is_json = request.headers.get("content-type", "").startswith("application/json")
        if is_json:
            body = await request.json()
            data = {
                "request_type": body.get("request_type"),
                "question": body.get("question"),
                "context": body.get("context"),
                "priority": body.get("priority", "normal"),
            }
            files_b64 = body.get("files_base64") or []
        else:
            data = {
                "request_type": request_type,
                "question": question,
                "context": context,
                "priority": priority or "normal",
            }
            files_b64 = []

        # Enforce required fields
        if not data.get("request_type") or not data.get("question"):
            raise HTTPException(status_code=400, detail="Request Type and Question are required.")

        # Validate basic fields (no deep_analysis anymore)
        try:
            form_obj = RequestForm(**{**data, "deep_analysis": False})
        except ValidationError as ve:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=ve.errors())

        saved_paths: List[str] = []

        # Consolidate uploaded files (single or multiple)
        uploaded_files: List[UploadFile] = []
        if file:
            uploaded_files.append(file)
        if files:
            uploaded_files.extend(files)

        # If multipart and no files provided, reject politely
        if not is_json and not uploaded_files:
            raise HTTPException(status_code=400, detail="A CSV file is required.")

        # Save uploaded files with validation
        for uf in uploaded_files:
            unique_name = f"{uuid.uuid4().hex}_{uf.filename}"
            dest = UPLOAD_DIR / unique_name
            try:
                content = await uf.read()
                size = len(content)
                val_log.info(f"size_bytes={size} path={dest}")
                if size > MAX_FILE_BYTES:
                    raise HTTPException(status_code=400, detail="File exceeds 10 MB limit. Please upload a smaller file.")

                async with aiofiles.open(dest, "wb") as f:
                    await f.write(content)

                # Row validation
                rows = _count_rows_stream(str(dest))
                val_log.info(f"rows={rows} path={dest}")
                if rows < 0:
                    raise HTTPException(status_code=400, detail="Invalid or corrupt CSV. Please upload a valid CSV.")
                if rows >= MAX_ROWS:
                    raise HTTPException(status_code=400, detail="File exceeds 50,000 rows. Please upload a dataset with fewer rows.")

                # Corruption check: quick sample read
                try:
                    _ = pd.read_csv(dest, nrows=100)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid file: {str(e)}. Please upload a valid CSV ≤ 10 MB and < 50,000 rows.")

                saved_paths.append(str(dest))
            except HTTPException:
                if dest.exists():
                    dest.unlink(missing_ok=True)
                raise
            except Exception as exc:
                if dest.exists():
                    dest.unlink(missing_ok=True)
                log.error("file_save_failed", file=uf.filename, error=str(exc))
                raise HTTPException(status_code=500, detail=f"Failed to save file {uf.filename}")

        # Save base64 files if provided in JSON
        for idx, fb in enumerate(files_b64):
            try:
                filename = fb.get("filename", f"upload_{idx}")
                data_b64 = fb.get("data_b64")
                if not data_b64:
                    continue
                decoded = base64.b64decode(data_b64)
                if len(decoded) > MAX_FILE_BYTES:
                    raise HTTPException(status_code=400, detail=f"File {filename} exceeds 10 MB limit.")
                unique_name = f"{uuid.uuid4().hex}_{filename}"
                dest = UPLOAD_DIR / unique_name
                async with aiofiles.open(dest, "wb") as f:
                    await f.write(decoded)
                # Optional: row validation for base64 uploads
                rows = _count_rows_stream(str(dest))
                if rows >= MAX_ROWS:
                    dest.unlink(missing_ok=True)
                    raise HTTPException(status_code=400, detail="File exceeds 50,000 rows. Please upload a dataset with fewer rows.")
                saved_paths.append(str(dest))
            except HTTPException:
                raise
            except Exception as exc:
                log.error("file_save_failed_b64", file=fb, error=str(exc))
                raise HTTPException(status_code=500, detail=f"Failed to save base64 file {filename}")

        record = crud.create_submission(
            db,
            request_type=form_obj.request_type,
            question=form_obj.question,
            context=form_obj.context,
            priority=form_obj.priority,
            deep_analysis=False,
            file_paths=saved_paths,
            status="pending",
        )

        # Audit log: submission created
        log_audit_event(
            event_type="submission_created",
            submission_id=record.id,
            metadata={
                "request_type": form_obj.request_type,
                "priority": form_obj.priority,
                "deep_analysis": False,
                "files_count": len(saved_paths),
            },
        )

        try:
            _ = process_submission(record.id)
            record = crud.get_submission(db, sub_id=record.id)
            if not record:
                raise HTTPException(status_code=500, detail="Submission lost during processing")

            # Audit log: processing completed
            log_audit_event(
                event_type="submission_processed",
                submission_id=record.id,
                metadata={"status": record.status},
            )
        except HTTPException:
            raise
        except Exception as exc:
            log.error("processing_failed", id=record.id, error=str(exc))
            crud.update_result(db, sub_id=record.id, status="error", result=str(exc))

            # Audit log: processing failed
            log_audit_event(
                event_type="submission_failed",
                submission_id=record.id,
                metadata={"error": sanitize_for_logging(str(exc))},
            )

            raise HTTPException(status_code=500, detail="Processing failed")

        return SubmitResponse(status=record.status, id=record.id, result=record.result)

    except HTTPException:
        raise
    except Exception as exc:
        logging.error("submit_form_error", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during analysis. Please try again or check your file.")


@router.get("/result/{sub_id}", response_model=ResultResponse)
async def get_result(sub_id: str, db: Session = Depends(get_db)) -> ResultResponse:
    record = crud.get_submission(db, sub_id=sub_id)
    if not record:
        raise HTTPException(status_code=404, detail="Not found")
    return ResultResponse(id=record.id, status=record.status, result=record.result)