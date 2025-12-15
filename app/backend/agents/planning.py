import json
import os
import logging
from functools import lru_cache
from typing import Dict, List, Any

import structlog
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from app.backend.tools.cost import estimate_prompt_cost

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
# def get_planner_llm() -> ChatAnthropic:
#     api_key = os.getenv("ANTHROPIC_API_KEY")
#     if not api_key:
#         raise RuntimeError("ANTHROPIC_API_KEY not set")
#     return ChatAnthropic(model="claude-sonnet-4-5", api_key=api_key, temperature=0.2, max_tokens=4096)
def get_planner_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    return ChatOpenAI(
        model="gpt-4.1",   # or: gpt-4.1-mini / gpt-4o
        api_key=api_key,
        temperature=0.2,
        max_tokens=4096
    )


def is_mock_mode() -> bool:
    """Check if mock execution mode is enabled (for testing without LLM costs)."""
    return os.getenv("MOCK_EXECUTION", "").lower() in ("1", "true", "yes")


def get_mock_plan(submission: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a mock plan for testing without LLM costs."""
    file_paths = submission.get("file_paths", [])
    file_path = file_paths[0] if file_paths else "data.csv"
    
    return {
        "task_type": "analyze",
        "steps": [
            {"id": "load_data", "action": "load_data", "params": {"file": file_path}},
            {"id": "group_aggregate", "action": "group_by_aggregate", "params": {"group_by": ["region"], "aggregations": {"revenue": ["sum", "mean", "count"]}}},
            {"id": "calculate_avg_revenue", "action": "summarize", "params": {"metric": "average_revenue_per_region"}},
            {"id": "format_output", "action": "format_results", "params": {"include_visualizations": True}},
        ],
        "required_columns": ["region", "revenue"],
        "expected_outputs": ["average_revenue_by_region", "summary_table"],
        "failure_modes": ["missing_columns", "empty_data"],
    }


PLANNING_PROMPT = ChatPromptTemplate.from_template(
    """
You are an analytics planning agent. Generate a structured execution plan.

Request:
- Type: {request_type}
- Question: {question}
- Context: {context}
- Deep Analysis: {deep_analysis}

Data Profile:
{profile}

Return ONLY minified JSON with this structure:
{{
  "task_type": str (e.g., "simulate", "predict", "analyze", "optimize"),
  "steps": [
    {{
      "id": str,
      "action": str (e.g., "load_data", "monte_carlo", "regression", "summarize"),
      "params": dict (e.g., {{"n_sims": 10000, "columns": ["col1"]}})
    }}
  ],
  "required_columns": list[str],
  "expected_outputs": list[str],
  "failure_modes": list[str]
}}

If deep_analysis is true, ensure simulation steps have n_sims=10000.
    """
)


def plan_analysis(submission: Dict[str, Any]) -> Dict[str, Any]:
    """Generate structured JSON plan from submission and data profile."""
    profile = submission.get("profile") or submission.get("file_summary") or "No profile available"
    deep_analysis = submission.get("deep_analysis", False)
    
    # Check for mock mode to save LLM costs during testing
    if is_mock_mode():
        log.info("planning_mock_mode")
        return get_mock_plan(submission)
    
    try:
        llm = get_planner_llm()
        prompt = PLANNING_PROMPT.format(
            request_type=submission.get("request_type"),
            question=submission.get("question"),
            context=submission.get("context") or "None",
            deep_analysis=deep_analysis,
            profile=json.dumps(profile) if isinstance(profile, dict) else str(profile),
        )
        guidelines = "Guidelines: For datasets >= 1000 rows, avoid loading the full dataset into LLM context; plan minimal extraction (aggregates, samples, filters) and return only textual summaries. No visualizations."
        prompt += f"\n{guidelines}"
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        
        # DEBUG: Log raw response
        log.info("planning_llm_raw_response", content=content[:500] if content else "EMPTY")
        
        if not content or not content.strip():
            raise ValueError("LLM returned empty response")
        
        # Strip markdown code blocks
        content = strip_markdown_json(content)
        
        plan = json.loads(content)
        
        # Enforce deep analysis params if requested
        if deep_analysis:
            for step in plan.get("steps", []):
                if "monte_carlo" in step.get("action", "").lower() or "sim" in step.get("action", "").lower():
                    if "params" not in step:
                        step["params"] = {}
                    step["params"]["n_sims"] = 10000
        est = estimate_prompt_cost(prompt)
        logging.info({"event": "llm_prompt_estimate", **est})
        
    except Exception as exc:  # noqa: BLE001
        log.warning("planning_llm_failed", error=str(exc))
        plan = {
            "task_type": "analyze",
            "steps": [
                {"id": "step_1", "action": "load_data", "params": {}},
                {"id": "step_2", "action": "compute_stats", "params": {}},
                {"id": "step_3", "action": "summarize", "params": {}},
            ],
            "required_columns": [],
            "expected_outputs": ["summary", "insights"],
            "failure_modes": ["missing_data", "llm_unavailable"],
        }
        
        # Add simulation step if deep analysis
        if deep_analysis:
            plan["steps"].insert(2, {
                "id": "step_sim",
                "action": "monte_carlo",
                "params": {"n_sims": 10000}
            })
    
    return plan
