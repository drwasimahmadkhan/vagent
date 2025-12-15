import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import structlog
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

log = structlog.get_logger()


def strip_markdown_json(content: str) -> str:
    """Remove markdown code block markers from JSON responses."""
    if not content:
        return content
    
    content = content.strip()
    
    # Remove ```json or ``` at start
    if content.startswith("```json"):
        content = content[7:]  # len("```json") + newline
    elif content.startswith("```"):
        content = content[3:]
    
    # Remove trailing ```
    if content.endswith("```"):
        content = content[:-3]
    
    return content.strip()


def profile_data(file_path: str, max_rows: int = 2000) -> Dict[str, Any]:
    """Generate data profile using pandas (lightweight alternative to ydata-profiling)."""
    try:
        df = pd.read_csv(file_path, nrows=max_rows)
        
        profile = {
            "file": file_path,
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "nulls": {col: int(df[col].isna().sum()) for col in df.columns},
            "head": df.head(5).to_dict(orient="records"),
            "stats": {},
        }
        
        # Add basic stats for numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            profile["stats"][col] = {
                "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                "std": float(df[col].std()) if not df[col].isna().all() else None,
                "min": float(df[col].min()) if not df[col].isna().all() else None,
                "max": float(df[col].max()) if not df[col].isna().all() else None,
            }
        
        return profile
    except Exception as exc:  # noqa: BLE001
        log.error("profiling_failed", file=file_path, error=str(exc))
        return {"file": file_path, "error": str(exc)}


def profile_and_validate(file_paths: List[str]) -> Dict[str, Any]:
    """Profile all files and generate validation summary with suggestions."""
    profiles = []
    issues = []
    suggestions = []
    
    for fpath in file_paths:
        if not Path(fpath).exists():
            issues.append(f"File not found: {fpath}")
            continue
            
        if fpath.lower().endswith(".csv"):
            profile = profile_data(fpath)
            if "error" in profile:
                issues.append(f"{fpath}: {profile['error']}")
                suggestions.append(f"Check file format and encoding for {Path(fpath).name}")
            elif profile.get("rows", 0) == 0:
                issues.append(f"{fpath}: Empty CSV")
                suggestions.append(f"Provide non-empty data file for {Path(fpath).name}")
            else:
                # Check for all-null columns
                null_cols = [col for col, count in profile.get("nulls", {}).items() if count == profile.get("rows", 0)]
                if null_cols:
                    issues.append(f"{fpath}: Columns with all nulls: {null_cols}")
                    suggestions.append(f"Remove or fill null columns: {', '.join(null_cols)}")
            
            profiles.append(profile)
        else:
            # Non-CSV files
            profiles.append({"file": fpath, "type": "non-csv", "size": Path(fpath).stat().st_size})
    
    valid = len(issues) == 0
    
    return {
        "valid": valid,
        "profiles": profiles,
        "issues": issues,
        "suggestions": suggestions if not valid else ["Data looks good"],
    }


@lru_cache(maxsize=1)
# def get_validator_llm() -> ChatAnthropic:
#     api_key = os.getenv("ANTHROPIC_API_KEY")
#     if not api_key:
#         raise RuntimeError("ANTHROPIC_API_KEY not set")
#     return ChatAnthropic(model="claude-sonnet-4-5", api_key=api_key, temperature=0, max_tokens=2048)

def get_validator_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    return ChatOpenAI(
        model="gpt-4.1-mini",   # deterministic & cheap
        temperature=0.0,
        max_tokens=2048
    )


def is_mock_mode() -> bool:
    """Check if mock execution mode is enabled (for testing without LLM costs)."""
    return os.getenv("MOCK_EXECUTION", "").lower() in ("1", "true", "yes")


VALIDATION_PROMPT = ChatPromptTemplate.from_template(
    """
You are a data validation assistant. Given request details and data profiles, assess if data is usable and relevant to the query.

CRITICAL CHECKS:
1. Is the data relevant to the question? (e.g., don't accept HR data for sales questions)
2. Does the data have sufficient rows and columns?
3. Are there critical data quality issues?

Request:
- Type: {request_type}
- Question: {question}
- Context: {context}

Data Profiles:
{profiles}

Issues Found:
{issues}

Respond with compact JSON: {{"ok": bool, "message": str, "suggestions": list[str], "relevance_check": str}}

If data is empty, corrupted, or completely irrelevant to the query, set ok to false.
In message, explain WHY validation failed (e.g., "File is empty", "Data contains employee records but query asks about sales revenue").
Provide actionable suggestions.
    """
)


def validate_submission(submission: Dict[str, Any]) -> Dict[str, Any]:
    """Profile files and validate with LLM guidance. REJECTS empty/irrelevant files early."""
    files: List[str] = submission.get("file_paths") or []
    
    # CRITICAL: Reject if no files provided
    if not files:
        return {
            "ok": False,
            "message": "No data file uploaded. Please provide a CSV file to analyze.",
            "profiles": [],
            "suggestions": ["Upload at least one CSV file with relevant data"],
        }
    
    profile_result = profile_and_validate(files)
    
    # CRITICAL: Reject immediately if profiling found critical issues
    if not profile_result["valid"]:
        return {
            "ok": False,
            "message": f"Data validation failed: {'; '.join(profile_result['issues'])}",
            "profiles": profile_result["profiles"],
            "suggestions": profile_result["suggestions"],
        }
    
    # Check for mock mode to save LLM costs during testing
    if is_mock_mode():
        log.info("validation_mock_mode", files=len(files))
        return {
            "ok": profile_result["valid"],
            "message": "[MOCK] Data validation passed. Columns and structure look appropriate for the query.",
            "profiles": profile_result["profiles"],
            "suggestions": profile_result["suggestions"] or ["Data looks good for analysis"],
            "relevance_check": "[MOCK] PASS - Data structure appears suitable for the requested analysis.",
        }
    
    # Use LLM to check query-data relevance
    try:
        llm = get_validator_llm()
        prompt = VALIDATION_PROMPT.format(
            request_type=submission.get("request_type"),
            question=submission.get("question"),
            context=submission.get("context") or "None",
            profiles=json.dumps(profile_result["profiles"], indent=2),
            issues=json.dumps(profile_result["issues"]),
        )
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        
        # DEBUG: Log raw response
        log.info("validation_llm_raw_response", content=content[:500] if content else "EMPTY")
        
        if not content or not content.strip():
            raise ValueError("LLM returned empty response")
        
        # Strip markdown code blocks
        content = strip_markdown_json(content)
        
        result = json.loads(content)
        
        # Log relevance check
        log.info("validation_relevance_check", 
                 relevance=result.get("relevance_check", "unknown"),
                 ok=result.get("ok", True))
        
    except Exception as exc:  # noqa: BLE001
        log.warning("validation_llm_failed", error=str(exc))
        result = {
            "ok": profile_result["valid"],
            "message": "Data profiling passed but LLM relevance check unavailable.",
            "suggestions": profile_result["suggestions"],
        }
    
    return {
        "ok": result.get("ok", profile_result["valid"]),
        "message": result.get("message", ""),
        "profiles": profile_result["profiles"],
        "suggestions": result.get("suggestions", profile_result["suggestions"]),
        "relevance_check": result.get("relevance_check", "Not performed"),
    }
