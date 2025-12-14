from typing import List, Optional

from sqlalchemy.orm import Session

from . import models


def create_submission(
    db: Session,
    *,
    request_type: str,
    question: str,
    context: Optional[str],
    priority: str,
    deep_analysis: bool,
    file_paths: Optional[List[str]],
    status: str = "pending",
) -> models.Submission:
    submission = models.Submission(
        request_type=request_type,
        question=question,
        context=context,
        priority=priority,
        deep_analysis=deep_analysis,
        files=file_paths or [],
        status=status,
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)
    return submission


def update_result(db: Session, *, sub_id: str, status: str, result: Optional[str]) -> models.Submission:
    submission = db.query(models.Submission).filter(models.Submission.id == sub_id).first()
    if not submission:
        return None
    submission.status = status
    submission.result = result
    db.commit()
    db.refresh(submission)
    return submission


def get_submission(db: Session, *, sub_id: str) -> Optional[models.Submission]:
    return db.query(models.Submission).filter(models.Submission.id == sub_id).first()
