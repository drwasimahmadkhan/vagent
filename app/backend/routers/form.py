from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from app.database import models
from app.database.db import get_db

REQUEST_TYPES = [
    "Price Strategy",
    "Growth & Marketing",
    "Operations",
    "Product",
    "Finance",
    "Other",
]

PRIORITIES = {"normal", "urgent"}


class RequestForm(BaseModel):
    request_type: str = Field(..., description="Category of the request")
    question: str = Field(..., max_length=2000, description="Main question or task")
    additional_context: Optional[str] = Field(None, description="Optional supporting details")
    priority: str = Field("normal", description="normal (72h) or urgent (48h)")
    deep_analysis: bool = Field(False, description="Run deep analysis with simulations")
    attachments: Optional[List[str]] = Field(None, description="Uploaded file names")

    @validator("request_type")
    def validate_request_type(cls, v: str) -> str:
        if v not in REQUEST_TYPES:
            raise ValueError("Invalid request type")
        return v

    @validator("priority")
    def validate_priority(cls, v: str) -> str:
        if v not in PRIORITIES:
            raise ValueError("Priority must be 'normal' or 'urgent'")
        return v


class RequestResponse(BaseModel):
    id: int
    status: str
    message: str


router = APIRouter(prefix="/form", tags=["form"])


@router.post("/submit", response_model=RequestResponse)
def submit_form(payload: RequestForm, db: Session = Depends(get_db)) -> RequestResponse:
    if payload.priority == "urgent":
        # Placeholder rule: urgent allowed but could gate by subscription
        pass

    record = models.RequestRecord(
        request_type=payload.request_type,
        question=payload.question,
        additional_context=payload.additional_context,
        priority=payload.priority,
        deep_analysis=payload.deep_analysis,
        status="queued",
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return RequestResponse(
        id=record.id,
        status=record.status,
        message="Request received and queued for processing.",
    )
