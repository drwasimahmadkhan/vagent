import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

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


@lru_cache(maxsize=1)
def get_response_llm() -> ChatAnthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return ChatAnthropic(model="claude-sonnet-4-5", api_key=api_key, temperature=0.3, max_tokens=4096)

# def get_response_llm():
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise RuntimeError("OPENAI_API_KEY not set")

#     return ChatOpenAI(
#         model="gpt-4.1",   # or gpt-4.1-mini / gpt-4o
#         temperature=0.3,
#         max_tokens=4096
#     )


def is_mock_mode() -> bool:
    """Check if mock execution mode is enabled (for testing without LLM costs)."""
    return os.getenv("MOCK_EXECUTION", "").lower() in ("1", "true", "yes")


RESPONSE_PROMPT = ChatPromptTemplate.from_template(
    """
You are a rigorous analytics verifier. Synthesize insights from validation, plan, and execution outputs.

CRITICAL REQUIREMENTS:
1. Include TLDR (2-3 sentences) if priority is urgent
2. Add confidence score (0.0-1.0) based on data quality and execution success
3. Perform sanity checks: validate bounds, units, statistical consistency
4. Add disclaimers for assumptions (e.g., "Assumes no PII", "Based on N rows")
5. Identify next_steps for further analysis
6. Flag any anomalies or data quality issues

Request Context:
- Type: {request_type}
- Question: {question}
- Priority: {priority}

Data Profile & Validation: {validation}
Execution Plan: {plan}
Execution Results: {execution}

Respond with valid JSON:
{{
  "tldr": "string (only if urgent, else null)",
  "summary": "string (comprehensive 3-5 sentences)",
  "confidence": 0.0-1.0,
  "insights": ["key finding 1", "key finding 2", ...],
  "sanity_checks": {{"metric": "status/warning", ...}},
  "risks": ["assumption 1", "limitation 1", ...],
  "disclaimers": ["Assumes X", "Based on Y rows", ...],
  "next_steps": ["recommendation 1", ...],
  "artifacts": {{"type": "list of available artifacts (e.g., csv, plot)"}}
}}

Sanity check examples:
- Bounds: "Revenue values in reasonable range (0 to 1M)"
- Units: "All timestamps in UTC"
- Statistics: "Mean within 2 std devs of median"
    """
)


def generate_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate comprehensive verified response with confidence scoring and sanity checks.
    
    Returns structured JSON optimized for UI parsing.
    """
    priority = payload.get("priority", "normal")
    execution = payload.get("execution", {})
    validation = payload.get("validation", {})
    
    # Extract data quality metrics
    profiles = validation.get("profiles", [])
    n_rows = profiles[0].get("rows", 0) if profiles else 0
    exec_ok = execution.get("ok", False)
    
    # Check for mock mode to save LLM costs during testing
    if is_mock_mode():
        log.info("response_mock_mode", exec_ok=exec_ok, n_rows=n_rows)
        
        # Extract results from execution outputs
        outputs = execution.get("outputs", {})
        insights = []
        
        for step_id, output in outputs.items():
            if output.get("ok") and output.get("result"):
                result = output.get("result", {})
                if isinstance(result, dict):
                    if "average_revenue_by_region" in result:
                        for region, avg in result["average_revenue_by_region"].items():
                            insights.append(f"{region}: Average revenue = ${avg:,.2f}")
                    elif "grouped_data" in result:
                        for item in result.get("grouped_data", [])[:3]:
                            insights.append(f"Region {item.get('region', 'Unknown')}: Mean = {item.get('mean', 'N/A')}")
        
        if not insights:
            insights = ["Analysis completed successfully", f"Processed {n_rows} rows of data"]
        
        return {
            "tldr": f"[MOCK] Analysis of {n_rows} rows completed." if priority == "urgent" else None,
            "summary": f"[MOCK] Successfully analyzed {n_rows} rows. Execution {'succeeded' if exec_ok else 'had issues'}.",
            "confidence": 0.85 if exec_ok else 0.4,
            "insights": insights,
            "sanity_checks": {
                "execution_status": "ok" if exec_ok else "error",
                "data_rows": f"{n_rows} rows processed",
            },
            "risks": ["[MOCK] Results generated without LLM verification"],
            "disclaimers": [f"Based on {n_rows} rows", "Mock mode - for testing only"],
            "next_steps": ["Review results", "Run in production mode for full analysis"],
            # Artifacts removed per requirements (no visuals)
            "artifacts": {},
        }
    
    try:
        llm = get_response_llm()
        
        # Audit log: record prompt
        log.info("verifier_llm_invoke", 
                 request_type=payload.get("request_type"),
                 priority=priority,
                 exec_ok=exec_ok,
                 n_rows=n_rows)
        
        prompt = RESPONSE_PROMPT.format(**payload)
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        
        # DEBUG: Log raw response
        log.info("response_llm_raw_output", content=content[:500] if content else "EMPTY")
        
        if not content or not content.strip():
            raise ValueError("LLM returned empty response")
        
        # Strip markdown code blocks
        content = strip_markdown_json(content)
        
        # Parse and validate response structure
        data = json.loads(content)
        
        # Ensure required fields
        data.setdefault("confidence", 0.7 if exec_ok else 0.3)
        data.setdefault("sanity_checks", {})
        data.setdefault("disclaimers", [])
        # Remove artifacts/visuals from outputs
        data.pop("artifacts", None)
        
        # Add default disclaimers if missing
        if not data.get("disclaimers"):
            disclaimers = [f"Based on {n_rows} rows" if n_rows > 0 else "Limited data available"]
            if not validation.get("ok", True):
                disclaimers.append("Data quality issues detected")
            data["disclaimers"] = disclaimers
        
        # Audit log: record response
        log.info("verifier_llm_success",
                 confidence=data.get("confidence"),
                 insights_count=len(data.get("insights", [])),
                 has_tldr=bool(data.get("tldr")))
        
    except Exception as exc:  # noqa: BLE001
        log.warning("response_llm_failed", error=str(exc))
        
        # Structured fallback with sanity checks
        data = {
            "tldr": "Analysis completed with LLM unavailable" if priority == "urgent" else None,
            "summary": f"Processed request with {n_rows if n_rows else 0} rows. Execution {'succeeded' if exec_ok else 'failed'}.",
            "confidence": 0.5 if exec_ok else 0.2,
            "insights": [
                "Validation completed" if validation.get("ok") else "Validation issues detected",
                f"Execution {'successful' if exec_ok else 'encountered errors'}",
            ],
            "sanity_checks": {
                "execution_status": "ok" if exec_ok else "error",
                "data_available": "yes" if n_rows and n_rows > 0 else "no",
            },
            "risks": [
                "LLM unavailable; using fallback response",
                "Confidence score is estimated",
            ],
            "disclaimers": [
                f"Based on {n_rows if n_rows else 0} rows" if n_rows else "No data available",
                "Assumes no PII in uploaded files",
                "Results require human validation",
            ],
            "next_steps": [
                "Review execution logs for details",
                "Validate results against business rules",
                "Provide additional context if needed",
            ],
            # No artifacts/visualizations included in fallback response
            "artifacts": {},
        }
    
    return data
