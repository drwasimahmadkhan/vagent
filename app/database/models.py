import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, String, Text, JSON, Index

from .db import Base


class Submission(Base):
    __tablename__ = "submissions"

    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    request_type = Column(String(100), nullable=False)
    question = Column(Text, nullable=False)
    context = Column(Text, nullable=True)
    priority = Column(String(20), nullable=False, default="normal")
    deep_analysis = Column(Boolean, default=False)
    status = Column(String(20), nullable=False, default="pending", index=True)
    files = Column(JSON, nullable=True)
    result = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


Index("idx_submissions_status", Submission.status)
