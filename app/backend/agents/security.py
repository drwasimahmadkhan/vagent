"""Security utilities: PII redaction, audit logging, quota management."""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

log = structlog.get_logger()


# --- PII Redaction (Simple Pattern-Based) ---
# For production, use presidio-analyzer for ML-based detection

PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b(\+\d{1,2}\s?)?(\d{3}[-.\s]??\d{3}[-.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-.\s]??\d{4})\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
}


def redact_pii_simple(text: str, mask: str = "[REDACTED]") -> str:
    """
    Simple pattern-based PII redaction.
    For production, integrate presidio-analyzer for ML-based detection.
    """
    import re
    
    redacted = text
    for pii_type, pattern in PII_PATTERNS.items():
        redacted = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", redacted, flags=re.IGNORECASE)
    
    return redacted


def sanitize_for_logging(data: Any, max_length: int = 500) -> str:
    """Sanitize data for safe logging (redact PII, truncate)."""
    try:
        text = json.dumps(data) if not isinstance(data, str) else data
        redacted = redact_pii_simple(text)
        if len(redacted) > max_length:
            redacted = redacted[:max_length] + "... [truncated]"
        return redacted
    except Exception:  # noqa: BLE001
        return "[UNSERIALIZABLE]"


# --- Audit Logging ---

AUDIT_LOG_DIR = Path("audit_logs")
AUDIT_LOG_DIR.mkdir(exist_ok=True)


def log_audit_event(
    event_type: str,
    user_id: Optional[str] = None,
    submission_id: Optional[str] = None,
    prompt: Optional[str] = None,
    response: Optional[str] = None,
    seed: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log audit events to structured JSON files.

    Args:
        event_type: Type of event (e.g., 'llm_invoke', 'submission_created')
        user_id: User identifier (hashed)
        submission_id: Submission UUID
        prompt: LLM prompt (sanitized)
        response: LLM response (sanitized)
        seed: Random seed used
        metadata: Additional metadata
    """
    # Wrap entire function in try-except to prevent audit logging from breaking the application
    try:
        timestamp = datetime.utcnow().isoformat()

        audit_record = {
            "timestamp": timestamp,
            "event_type": event_type,
            "user_id": user_id,
            "submission_id": submission_id,
            "seed": seed,
            "metadata": metadata or {},
        }

        # Sanitize and add prompt/response if provided
        if prompt:
            audit_record["prompt_hash"] = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            audit_record["prompt_preview"] = sanitize_for_logging(prompt, max_length=200)

        if response:
            audit_record["response_hash"] = hashlib.sha256(response.encode()).hexdigest()[:16]
            audit_record["response_preview"] = sanitize_for_logging(response, max_length=200)

        # Write to daily log file
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        log_file = AUDIT_LOG_DIR / f"audit_{date_str}.jsonl"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_record) + "\n")

        # Use print instead of logging to avoid Windows file handle issues with structlog
        print(f"[AUDIT] audit_logged: event_type={event_type}, submission_id={submission_id}")
    except Exception as exc:  # noqa: BLE001
        # Silently fail - audit logging should never break the application
        # Use basic print to stderr to at least try to report the error
        try:
            print(f"[AUDIT ERROR] audit_log_failed: {str(exc)}", flush=True)
        except:  # noqa: E722
            pass  # If even printing fails, just ignore


# --- Quota Management ---

QUOTA_LIMITS = {
    "deep_analysis_per_hour": int(os.getenv("DEEP_ANALYSIS_QUOTA", "10")),
    "submissions_per_hour": int(os.getenv("SUBMISSIONS_QUOTA", "100")),
}

# In-memory quota tracking (for production, use Redis or database)
_quota_tracker: Dict[str, List[float]] = {}


def check_quota(quota_type: str, user_id: str = "default") -> bool:
    """
    Check if user has quota remaining.
    
    Returns:
        True if within quota, False if exceeded
    """
    import time
    
    current_time = time.time()
    key = f"{quota_type}:{user_id}"
    
    # Initialize tracker
    if key not in _quota_tracker:
        _quota_tracker[key] = []
    
    # Remove entries older than 1 hour
    _quota_tracker[key] = [t for t in _quota_tracker[key] if current_time - t < 3600]
    
    # Check limit
    limit = QUOTA_LIMITS.get(quota_type, 100)
    if len(_quota_tracker[key]) >= limit:
        log.warning("quota_exceeded", quota_type=quota_type, user_id=user_id)
        return False
    
    # Record usage
    _quota_tracker[key].append(current_time)
    return True


def get_quota_usage(quota_type: str, user_id: str = "default") -> Dict[str, Any]:
    """Get current quota usage statistics."""
    import time
    
    current_time = time.time()
    key = f"{quota_type}:{user_id}"
    
    if key not in _quota_tracker:
        usage = 0
    else:
        # Count entries in last hour
        usage = len([t for t in _quota_tracker[key] if current_time - t < 3600])
    
    limit = QUOTA_LIMITS.get(quota_type, 100)
    
    return {
        "quota_type": quota_type,
        "user_id": user_id,
        "usage": usage,
        "limit": limit,
        "remaining": max(0, limit - usage),
        "reset_in_seconds": 3600,
    }
