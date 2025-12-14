import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import structlog
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from app.database import crud
from app.database.db import get_db
from app.backend.agents import validate_submission, plan_analysis, run_execution, generate_response

log = structlog.get_logger()


def _sanitize_result_bundle(obj: Any) -> Any:
    """Sanitize result bundle to remove non-JSON-serializable objects like DataFrames."""
    import pandas as pd
    
    if isinstance(obj, dict):
        return {k: _sanitize_result_bundle(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_result_bundle(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')  # Convert DataFrame to list of dicts
    elif isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'dtype'):  # pandas dtype objects
        return str(obj)
    else:
        return obj


# --- LangGraph State for PEV Orchestration ---

class OrchestrationState(TypedDict):
    submission_id: str
    payload: Dict[str, Any]
    validation: Optional[Dict[str, Any]]
    plan: Optional[Dict[str, Any]]
    plan_validation: Optional[Dict[str, Any]]
    execution: Optional[Dict[str, Any]]
    response: Optional[Dict[str, Any]]
    hitl_required: bool
    error: Optional[str]
    status: str


# --- Node Functions ---

def validate_node(state: OrchestrationState) -> OrchestrationState:
    """Profile and validate uploaded data. REJECTS early if validation fails."""
    try:
        payload = state["payload"]
        validation = validate_submission(payload)
        state["validation"] = validation
        
        if not validation.get("ok", True):
            state["status"] = "error"
            state["error"] = validation.get("message", "Validation failed")
            
            # Generate user-friendly response for validation failures
            state["response"] = {
                "tldr": None,
                "summary": validation.get("message", "Data validation failed"),
                "confidence": 0.0,
                "insights": [],
                "sanity_checks": {"validation": "failed"},
                "risks": ["Data does not meet quality requirements"],
                "disclaimers": validation.get("suggestions", []),
                "next_steps": validation.get("suggestions", []),
                "artifacts": {},
            }
            
            log.warning("validation_rejected", 
                       id=state["submission_id"], 
                       message=validation.get("message"),
                       suggestions=validation.get("suggestions"))
        else:
            log.info("validation_passed", id=state["submission_id"])
    except Exception as exc:  # noqa: BLE001
        log.error("validate_node_error", id=state["submission_id"], error=str(exc))
        state["status"] = "error"
        state["error"] = f"Validation error: {exc}"
    
    return state


def plan_node(state: OrchestrationState) -> OrchestrationState:
    """Generate structured execution plan."""
    try:
        payload = state["payload"]
        validation = state["validation"]
        
        # Inject profiles from validation
        profiles = validation.get("profiles", []) if validation else []
        payload["profile"] = profiles
        
        plan = plan_analysis(payload)
        state["plan"] = plan
        log.info("plan_generated", id=state["submission_id"], task_type=plan.get("task_type"))
    except Exception as exc:  # noqa: BLE001
        log.error("plan_node_error", id=state["submission_id"], error=str(exc))
        state["status"] = "error"
        state["error"] = f"Planning error: {exc}"
    
    return state


def validate_plan_node(state: OrchestrationState) -> OrchestrationState:
    """Validate plan against data profile (check required_columns exist)."""
    try:
        plan = state["plan"]
        validation = state["validation"]
        
        if not plan or not validation:
            state["plan_validation"] = {"ok": False, "error": "Missing plan or validation"}
            return state
        
        required_cols = plan.get("required_columns", [])
        profiles = validation.get("profiles", [])
        
        # Extract all available columns from profiles
        available_cols = set()
        for profile in profiles:
            cols = profile.get("columns", [])
            available_cols.update(cols)
        
        missing_cols = [col for col in required_cols if col not in available_cols]
        
        if missing_cols:
            state["plan_validation"] = {
                "ok": False,
                "missing_columns": missing_cols,
                "available_columns": list(available_cols),
                "error": f"Plan requires columns {missing_cols} not found in data",
            }
            state["status"] = "error"
            state["error"] = f"Plan validation failed: missing columns {missing_cols}"
            log.warning("plan_validation_failed", id=state["submission_id"], missing=missing_cols)
        else:
            state["plan_validation"] = {
                "ok": True,
                "required_columns": required_cols,
                "available_columns": list(available_cols),
            }
            log.info("plan_validated", id=state["submission_id"])
    except Exception as exc:  # noqa: BLE001
        log.error("validate_plan_node_error", id=state["submission_id"], error=str(exc))
        state["plan_validation"] = {"ok": False, "error": str(exc)}
        state["status"] = "error"
        state["error"] = f"Plan validation error: {exc}"
    
    return state


def execute_node(state: OrchestrationState) -> OrchestrationState:
    """Execute plan using ReAct loop."""
    try:
        plan = state["plan"]
        payload = state["payload"]
        
        exec_res = run_execution(plan, payload)
        state["execution"] = {
            "ok": exec_res.ok,
            "observations": exec_res.observations,
            "outputs": exec_res.outputs,
        }
        
        if not exec_res.ok:
            state["status"] = "error"
            state["error"] = "Execution failed"
            log.warning("execution_failed", id=state["submission_id"])
        else:
            log.info("execution_completed", id=state["submission_id"])
    except Exception as exc:  # noqa: BLE001
        log.error("execute_node_error", id=state["submission_id"], error=str(exc))
        state["status"] = "error"
        state["error"] = f"Execution error: {exc}"
        state["execution"] = {"ok": False, "observations": [str(exc)], "outputs": {}}
    
    return state


def verify_node(state: OrchestrationState) -> OrchestrationState:
    """Synthesize final response and detect HITL requirement."""
    try:
        payload = state["payload"]
        validation = state["validation"]
        plan = state["plan"]
        execution = state["execution"]
        
        response = generate_response({
            "request_type": payload.get("request_type"),
            "question": payload.get("question"),
            "priority": payload.get("priority"),
            "validation": validation,
            "plan": plan,
            "execution": execution,
        })
        
        state["response"] = response
        
        # Detect HITL requirement for high-impact requests
        request_type = payload.get("request_type", "").lower()
        high_impact_types = ["strategic decisions", "strategic", "decision"]
        hitl_required = any(keyword in request_type for keyword in high_impact_types)
        
        if hitl_required:
            state["hitl_required"] = True
            response["hitl_flag"] = True
            response["hitl_reason"] = f"High-impact request type: {payload.get('request_type')}"
            log.info("hitl_flagged", id=state["submission_id"], request_type=payload.get("request_type"))
        
        if state["status"] != "error":
            state["status"] = "done"
        
        log.info("verification_completed", id=state["submission_id"], hitl=hitl_required)
    except Exception as exc:  # noqa: BLE001
        log.error("verify_node_error", id=state["submission_id"], error=str(exc))
        state["status"] = "error"
        state["error"] = f"Verification error: {exc}"
    
    return state


# --- Graph Construction ---

def create_pev_graph() -> StateGraph:
    """Create PEV (Plan-Execute-Verify) orchestration graph."""
    workflow = StateGraph(OrchestrationState)
    
    workflow.add_node("validate", validate_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("validate_plan", validate_plan_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("verify", verify_node)
    
    workflow.set_entry_point("validate")
    
    # Conditional edges
    def should_continue_after_validate(state: OrchestrationState) -> str:
        if state.get("status") == "error":
            return "end"
        return "plan"
    
    def should_continue_after_validate_plan(state: OrchestrationState) -> str:
        plan_validation = state.get("plan_validation", {})
        if not plan_validation.get("ok", False):
            return "end"
        return "execute"
    
    workflow.add_conditional_edges("validate", should_continue_after_validate, {"plan": "plan", "end": END})
    workflow.add_edge("plan", "validate_plan")
    workflow.add_conditional_edges("validate_plan", should_continue_after_validate_plan, {"execute": "execute", "end": END})
    workflow.add_edge("execute", "verify")
    workflow.add_edge("verify", END)
    
    return workflow.compile()


# --- Public Interface ---

def process_submission(sub_id: str) -> Dict[str, Any]:
    """
    Orchestrate PEV workflow using LangGraph.
    
    Workflow:
    1. Load submission from DB
    2. Validate → Plan → Validate Plan → Execute → Verify
    3. Update DB with result bundle
    4. Return status and result
    """
    db = next(get_db())
    record = crud.get_submission(db, sub_id=sub_id)
    if not record:
        raise ValueError(f"Submission {sub_id} not found")

    payload: Dict[str, Any] = {
        "id": record.id,
        "request_type": record.request_type,
        "question": record.question,
        "context": record.context,
        "priority": record.priority,
        "deep_analysis": record.deep_analysis,
        "file_paths": record.files or [],
    }

    # Priority handling: slight delay for normal requests
    if record.priority == "normal":
        time.sleep(float(os.getenv("NORMAL_DELAY_SEC", "0")))

    try:
        # Initialize state
        initial_state: OrchestrationState = {
            "submission_id": sub_id,
            "payload": payload,
            "validation": None,
            "plan": None,
            "plan_validation": None,
            "execution": None,
            "response": None,
            "hitl_required": False,
            "error": None,
            "status": "pending",
        }
        
        # Run PEV graph
        graph = create_pev_graph()
        final_state = None
        
        # Add global retry protection to avoid infinite loops
        retries = 0
        for state in graph.stream(initial_state):
            final_state = state
            retries += 1
            if retries > 1000:
                break
        
        # Extract final state from LangGraph output
        if final_state:
            last_node_state = list(final_state.values())[-1] if final_state else initial_state
        else:
            last_node_state = initial_state
            last_node_state["status"] = "error"
            last_node_state["error"] = "Graph execution failed"
        
        # Build result bundle
        result_bundle = {
            "validation": last_node_state.get("validation"),
            "plan": last_node_state.get("plan"),
            "plan_validation": last_node_state.get("plan_validation"),
            "execution": last_node_state.get("execution"),
            "response": last_node_state.get("response"),
            "hitl_required": last_node_state.get("hitl_required", False),
        }
        
        # Sanitize result bundle to remove non-JSON-serializable objects
        result_bundle = _sanitize_result_bundle(result_bundle)
        
        final_status = last_node_state.get("status", "error")
        
        # Update DB
        crud.update_result(db, sub_id=record.id, status=final_status, result=json.dumps(result_bundle))
        
        log.info("orchestration_completed", id=sub_id, status=final_status, hitl=result_bundle["hitl_required"])
        return {"status": final_status, "result": result_bundle}

    except Exception as exc:  # noqa: BLE001
        log.error("orchestration_failed", id=sub_id, error=str(exc))
        crud.update_result(db, sub_id=record.id, status="error", result=str(exc))
        return {"status": "error", "result": str(exc)}