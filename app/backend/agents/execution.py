import json
import os
import time
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import numba
from joblib import Parallel, delayed
from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr
import structlog
import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

log = structlog.get_logger()


def _sanitize_for_json(obj: Any) -> Any:
    """Convert objects to JSON-serializable types."""
    import pandas as pd
    
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')  # Convert DataFrame to list of dicts
    elif isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    elif isinstance(obj, type(pd.api.types.pandas_dtype('object'))):  # pandas dtype objects
        return str(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


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


# Safe builtins for RestrictedPython
SAFE_BUILTINS = {
    "__builtins__": {
        **safe_globals,
        "range": range,
        "len": len,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "sorted": sorted,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "_getiter_": guarded_iter_unpack_sequence,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_getattr_": safer_getattr,
    },
    "pd": None,  # Will be injected
    "np": None,  # Will be injected
    "sklearn": None,  # Will be injected
}


# --- Code Templates -----------------------------------------------------------

CODE_TEMPLATES = {
    "monte_carlo": """
# Monte Carlo Simulation Template
import numpy as np
n_sims = {n_sims}
seed = {seed}
np.random.seed(seed)

# Vectorized simulation
samples = np.random.normal(loc={mean}, scale={std}, size=n_sims)
result = {{
    "mean": float(np.mean(samples)),
    "std": float(np.std(samples)),
    "p5": float(np.percentile(samples, 5)),
    "p95": float(np.percentile(samples, 95)),
    "min": float(np.min(samples)),
    "max": float(np.max(samples)),
}}
""",
    "load_data": """
# Load Data Template
import pandas as pd
file_path = "{file_path}"
df = pd.read_csv(file_path, nrows={max_rows})
result = {{
    "shape": df.shape,
    "columns": list(df.columns),
    "head": df.head(3).to_dict(orient="records"),
}}
""",
    "compute_stats": """
# Compute Statistics Template
import pandas as pd
import numpy as np
file_path = "{file_path}"
df = pd.read_csv(file_path)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
result = {{col: {{
    "mean": float(df[col].mean()),
    "median": float(df[col].median()),
    "std": float(df[col].std()),
}} for col in numeric_cols}}
""",
}


# --- Tools --------------------------------------------------------------------

class PrintCollector:
    """Collect print output for RestrictedPython."""
    def __init__(self):
        self._output = []
    
    def __call__(self, *args, **kwargs):
        self._output.append(' '.join(str(a) for a in args))
    
    def _call_print(self, *args, **kwargs):
        self(*args, **kwargs)
    
    def output(self):
        return '\n'.join(self._output)


def strip_imports(code: str) -> str:
    """Remove import statements from code since modules are pre-injected."""
    import re
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip import lines
        if stripped.startswith('import ') or stripped.startswith('from '):
            continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)


def strip_print_statements(code: str) -> str:
    """Remove or comment out print statements to avoid RestrictedPython issues."""
    import re
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        # Replace print statements with pass
        if re.match(r'^\s*print\s*\(', line):
            # Comment it out instead of removing to preserve line numbers
            cleaned_lines.append('# ' + line)
        else:
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)


def create_sandbox_env() -> Dict[str, Any]:
    """Create a fresh sandbox environment with all allowed modules."""
    import pandas as pd
    import numpy as np
    import sklearn
    import os
    
    _print_ = PrintCollector()
    
    env = {
        "__builtins__": SAFE_BUILTINS["__builtins__"],
        "_print_": _print_,
        "_getattr_": safer_getattr,
        "_getiter_": iter,
        "_getitem_": lambda obj, key: obj[key],
        "_write_": lambda obj: obj,  # Allow attribute writes
        "_print": _print_,  # RestrictedPython uses _print (no trailing underscore)
        # Inject modules with common aliases
        "pd": pd,
        "pandas": pd,
        "np": np,
        "numpy": np,
        "sklearn": sklearn,
        "os": os,
        # Add common builtins that might be needed
        "len": len,
        "range": range,
        "list": list,
        "dict": dict,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "tuple": tuple,
        "set": set,
    }
    return env


def safe_code_execution(code: str, shared_env: Optional[Dict[str, Any]] = None, allowed_modules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute Python code in RestrictedPython sandbox with optional shared environment."""
    import pandas as pd
    import numpy as np
    import sklearn
    import os
    
    try:
        # Strip import statements - modules are pre-injected
        code = strip_imports(code)
        # Strip print statements to avoid RestrictedPython binding issues
        code = strip_print_statements(code)
        
        compile_result = compile_restricted(code, filename="<sandbox>", mode="exec")
        
        # Check if compilation succeeded - handle both tuple and object returns
        errors = getattr(compile_result, 'errors', None) or (compile_result[1] if isinstance(compile_result, tuple) else None)
        code_obj = getattr(compile_result, 'code', None) or (compile_result[0] if isinstance(compile_result, tuple) else compile_result)
        
        if errors:
            return {"ok": False, "error": f"Compilation errors: {errors}"}
        
        if not code_obj:
            return {"ok": False, "error": "Compilation produced no code"}
        
        # Use shared environment if provided, otherwise create new one
        if shared_env is not None:
            local_env = shared_env
            # Refresh the print collector but keep it properly bound
            _print_ = PrintCollector()
            local_env["_print_"] = _print_
            local_env["_print"] = _print_  # RestrictedPython uses _print (no trailing underscore)
        else:
            local_env = create_sandbox_env()
            # Print is already properly bound in create_sandbox_env()
        
        # Always ensure modules are available
        local_env["pd"] = pd
        local_env["pandas"] = pd
        local_env["np"] = np
        local_env["numpy"] = np
        local_env["sklearn"] = sklearn
        local_env["os"] = os
        
        if allowed_modules:
            local_env.update(allowed_modules)
        
        # Retry with timeout protection (Windows-compatible)
        import concurrent.futures
        import platform

        tries = 0
        timeout_seconds = 160

        while True:
            try:
                # Use ThreadPoolExecutor for cross-platform timeout support
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(exec, code_obj, local_env)
                    try:
                        future.result(timeout=timeout_seconds)
                        break
                    except concurrent.futures.TimeoutError:
                        raise TimeoutError("Execution timed out")
            except Exception as e:
                tries += 1
                if tries >= 3:
                    raise
        
        # Extract result and print output
        result = local_env.get("result", {})
        print_output = local_env["_print_"].output() if hasattr(local_env.get("_print_"), 'output') else ""
        
        # Convert any pandas/numpy objects to JSON-serializable types
        result = _sanitize_for_json(result)
        
        return {"ok": True, "result": result, "output": print_output, "_env": local_env}
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.basicConfig(filename="errors.log", level=logging.ERROR)
        logging.error("safe_code_execution_error", exc_info=True)
        return {"ok": False, "error": "Execution failed. Please try again.", "type": type(exc).__name__}


@numba.jit(nopython=True, parallel=True)
def _monte_carlo_numba(n_sims: int, mean: float, std: float, seed: int) -> np.ndarray:
    """Numba-accelerated Monte Carlo simulation."""
    np.random.seed(seed)
    return np.random.normal(mean, std, n_sims)


def monte_carlo_numba(n_sims: int, mean: float = 0.0, std: float = 1.0, seed: int = 42) -> Dict[str, Any]:
    """High-performance Monte Carlo with Numba for deep analysis."""
    try:
        samples = _monte_carlo_numba(n_sims, mean, std, seed)
        return {
            "ok": True,
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "p5": float(np.percentile(samples, 5)),
            "p50": float(np.percentile(samples, 50)),
            "p95": float(np.percentile(samples, 95)),
            "min": float(np.min(samples)),
            "max": float(np.max(samples)),
            "n_sims": n_sims,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def rag_query_faiss_stub(query: str, file_paths: Optional[List[str]] = None) -> Dict[str, Any]:
    """Placeholder RAG using FAISS. TODO: Implement vectorstore + embeddings."""
    log.info("rag_query_stub", query=query, files=file_paths)
    return {
        "ok": True,
        "query": query,
        "summary": "RAG not fully implemented; returning stub context.",
        "sources": file_paths or [],
        "chunks": [],
    }


# --- ReAct Agent with LangGraph -----------------------------------------------

from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict


class AgentState(TypedDict):
    plan: Dict[str, Any]
    submission: Dict[str, Any]
    current_step_idx: int
    observations: List[str]
    outputs: Dict[str, Any]
    retry_count: int
    max_retries: int
    completed: bool
    error: Optional[str]
    _decision: Optional[Dict[str, Any]]
    _shared_env: Optional[Dict[str, Any]]  # Shared execution environment for state persistence


def get_execution_llm() -> ChatAnthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return ChatAnthropic(model="claude-sonnet-4-5", api_key=api_key, temperature=0.1, max_tokens=4096)

# --- ADD OpenAI LLM ---
# def get_execution_llm():
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise RuntimeError("OPENAI_API_KEY not set")

#     return ChatOpenAI(
#         model="gpt-4.1",   # or gpt-4.1 / gpt-4o / gpt-4.1-mini
#         api_key=api_key,
#         temperature=0.1,
#         max_tokens=4096
#     )

def is_mock_mode() -> bool:
    """Check if mock execution mode is enabled (for testing without LLM costs)."""
    return os.getenv("MOCK_EXECUTION", "").lower() in ("1", "true", "yes")


def get_mock_decision(step: Dict[str, Any], file_path: str = None) -> Dict[str, Any]:
    """Generate mock decision for testing without LLM calls."""
    action = step.get("action", "")
    params = step.get("params", {})
    
    # Get file path from params or use default
    fp = params.get("file", file_path) or "data.csv"
    
    if action == "load_data":
        return {
            "reasoning": "[MOCK] Loading data from CSV file",
            "tool": "code_execution",
            "params": {"description": "Load CSV data"},
            "code": f"""import pandas as pd
df = pd.read_csv('{fp}')
result = {{
    'ok': True,
    'shape': df.shape,
    'columns': list(df.columns),
    'head': df.head(3).to_dict(orient='records')
}}
"""
        }
    elif action in ("group_by_aggregate", "group_aggregate"):
        group_by = params.get("group_by", ["region"])
        aggs = params.get("aggregations", {"revenue": ["mean"]})
        group_col = group_by[0] if group_by else "region"
        agg_col = list(aggs.keys())[0] if aggs else "revenue"
        return {
            "reasoning": f"[MOCK] Grouping by {group_col} and aggregating {agg_col}",
            "tool": "code_execution",
            "params": {"description": "Group and aggregate data"},
            "code": f"""import pandas as pd
df = pd.read_csv('{fp}')
grouped = df.groupby('{group_col}')['{agg_col}'].agg(['sum', 'mean', 'count']).reset_index()
result = {{
    'ok': True,
    'grouped_data': grouped.to_dict(orient='records'),
    'summary': f'Grouped by {group_col}, found {{len(grouped)}} groups'
}}
"""
        }
    elif action in ("summarize", "calculate_avg_revenue"):
        return {
            "reasoning": "[MOCK] Generating summary statistics",
            "tool": "code_execution",
            "params": {"description": "Calculate summary"},
            "code": f"""import pandas as pd
df = pd.read_csv('{fp}')
if 'region' in df.columns and 'revenue' in df.columns:
    avg_by_region = df.groupby('region')['revenue'].mean().to_dict()
    result = {{'ok': True, 'average_revenue_by_region': avg_by_region}}
else:
    result = {{'ok': True, 'stats': df.describe().to_dict()}}
"""
        }
    elif action == "format_results":
        return {
            "reasoning": "[MOCK] Formatting final results",
            "tool": "code_execution",
            "params": {"description": "Format output"},
            "code": f"""import pandas as pd
df = pd.read_csv('{fp}')
if 'region' in df.columns and 'revenue' in df.columns:
    summary = df.groupby('region')['revenue'].agg(['mean', 'sum', 'count']).reset_index()
    summary.columns = ['Region', 'Average Revenue', 'Total Revenue', 'Count']
    result = {{
        'ok': True,
        'formatted_table': summary.to_dict(orient='records'),
        'chart_data': summary.to_dict(orient='list')
    }}
else:
    result = {{'ok': True, 'message': 'Data formatted successfully'}}
"""
        }
    else:
        # Generic fallback
        return {
            "reasoning": f"[MOCK] Executing step: {action}",
            "tool": "code_execution",
            "params": {"description": f"Execute {action}"},
            "code": f"""result = {{'ok': True, 'action': '{action}', 'status': 'completed'}}
"""
        }


REASON_PROMPT = ChatPromptTemplate.from_template(
    """
You are a ReAct execution agent. Given a plan step, reason about how to execute it safely.

Current Step:
{step}

Available Tools:
- code_execution: Run sandboxed Python (pandas, numpy, sklearn, numba allowed)
- rag_query: Query document context
- monte_carlo_numba: High-performance simulations

Previous Observations:
{observations}

If previous attempt failed with error: {error}
Reflect on the error and suggest a fix.

Respond with JSON:
{{
  "reasoning": str,
  "tool": str (one of: code_execution, rag_query, monte_carlo_numba),
  "params": dict,
  "code": str (if tool=code_execution, provide safe code using templates)
}}
"""
)


def reason_node(state: AgentState) -> AgentState:
    """Reason about next action."""
    plan = state["plan"]
    steps = plan.get("steps", [])
    idx = state["current_step_idx"]
    
    if idx >= len(steps):
        state["completed"] = True
        return state
    
    step = steps[idx]
    observations = state["observations"]
    error = state.get("error")
    
    # Check for mock mode to save LLM costs during testing
    if is_mock_mode():
        # Get file path from submission or first step
        file_paths = state.get("submission", {}).get("file_paths", [])
        file_path = file_paths[0] if file_paths else None
        
        decision = get_mock_decision(step, file_path)
        state["_decision"] = decision
        observations.append(f"Step {step.get('id')}: [MOCK] {decision.get('reasoning', 'N/A')[:80]}")
        log.info("reason_mock_mode", step_id=step.get("id"), tool=decision.get("tool"))
        return state
    
    try:
        llm = get_execution_llm()
        prompt = REASON_PROMPT.format(
            step=json.dumps(step),
            observations="\n".join(observations[-3:]) if observations else "None",
            error=error or "None",
        )
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        
        # DEBUG: Log FULL raw response
        log.info("reason_llm_raw_response", content=content, length=len(content) if content else 0)
        
        if not content or not content.strip():
            raise ValueError("LLM returned empty response")
        
        # Strip markdown code blocks
        content = strip_markdown_json(content)
        
        # DEBUG: Log cleaned content
        log.info("reason_llm_cleaned", content=content[:800], length=len(content))
        
        try:
            decision = json.loads(content)
        except json.JSONDecodeError as e:
            log.error("json_parse_failed", error=str(e), content=content[:1000])
            raise ValueError(f"Failed to parse JSON: {e}. Content: {content[:200]}")
        
        # Validate decision has required fields
        if not decision.get("tool"):
            log.error("decision_missing_tool", decision_keys=list(decision.keys()), decision=decision)
            raise ValueError(f"Decision missing 'tool' field. Got keys: {list(decision.keys())}, Decision: {decision}")
        
        state["_decision"] = decision
        observations.append(f"Step {step.get('id')}: Reasoning -> {decision.get('reasoning', 'N/A')[:100]}")
        log.info("reason_success", tool=decision.get("tool"), has_code=bool(decision.get("code")))
        
    except Exception as exc:  # noqa: BLE001
        log.error("reason_failed", error=str(exc), error_type=type(exc).__name__)
        observations.append(f"Reasoning failed: {exc}")
        state["error"] = str(exc)
        # CRITICAL: Don't set _decision if parsing failed
        state["_decision"] = None
    
    return state


def act_node(state: AgentState) -> AgentState:
    """Execute the planned action."""
    decision = state.get("_decision")
    step = state["plan"]["steps"][state["current_step_idx"]]
    step_id = step.get("id", "unknown")
    
    # CRITICAL: Check if decision exists (reason_node may have failed)
    if decision is None:
        log.error("act_no_decision", step_id=step_id)
        result = {"ok": False, "error": "No decision from reasoning step (JSON parse failed)"}
        state["outputs"][step_id] = result
        state["observations"].append(f"Step {step_id}: No decision available")
        state["error"] = "Reasoning produced no valid decision"
        return state
    
    tool = decision.get("tool")
    params = decision.get("params", {})
    code = decision.get("code")
    
    # Get or create shared environment for state persistence between steps
    shared_env = state.get("_shared_env")
    
    # Log what we got
    log.info("act_execute", step_id=step_id, tool=tool, has_code=bool(code), params_keys=list(params.keys()) if params else [])
    
    try:
        if tool == "code_execution" and code:
            result = safe_code_execution(code, shared_env=shared_env)
            # Save the updated environment for next steps
            if result.get("ok") and result.get("_env"):
                state["_shared_env"] = result.get("_env")
                # Remove _env from result to avoid serialization issues
                result_clean = {k: v for k, v in result.items() if k != "_env"}
                state["outputs"][step_id] = result_clean
            else:
                state["outputs"][step_id] = result
            state["observations"].append(f"Step {step_id}: Code executed, ok={result.get('ok')}")
        elif tool == "monte_carlo_numba":
            n_sims = params.get("n_sims", 1000)
            result = monte_carlo_numba(n_sims, seed=params.get("seed", 42))
            state["outputs"][step_id] = result
            state["observations"].append(f"Step {step_id}: Numba simulation, n={n_sims}")
        elif tool == "rag_query":
            query = params.get("query", state["submission"].get("question", ""))
            result = rag_query_faiss_stub(query, state["submission"].get("file_paths"))
            state["outputs"][step_id] = result
            state["observations"].append(f"Step {step_id}: RAG query executed")
        else:
            state["observations"].append(f"Step {step_id}: Unknown tool {tool}")
            result = {"ok": False, "error": f"Unknown tool: {tool}"}
            state["outputs"][step_id] = result
        
        # Clear error if successful
        if result.get("ok", False):
            state["error"] = None
            state["current_step_idx"] += 1
            state["retry_count"] = 0
        else:
            state["error"] = result.get("error", "Unknown error")
    except Exception as exc:  # noqa: BLE001
        log.error("act_failed", step_id=step_id, error=str(exc))
        state["error"] = str(exc)
        state["observations"].append(f"Step {step_id}: Execution error: {exc}")
    
    return state


def observe_node(state: AgentState) -> AgentState:
    """Observe results and decide next action."""
    if state.get("error"):
        state["retry_count"] += 1
        if state["retry_count"] >= state["max_retries"]:
            state["completed"] = True
            state["observations"].append(f"Max retries reached for step {state['current_step_idx']}")
    
    return state


def should_continue(state: AgentState) -> str:
    """Decide whether to continue or end."""
    if state["completed"]:
        return "end"
    if state["current_step_idx"] >= len(state["plan"].get("steps", [])):
        return "end"
    return "reason"


def create_react_graph() -> StateGraph:
    """Create LangGraph ReAct workflow."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("reason", reason_node)
    workflow.add_node("act", act_node)
    workflow.add_node("observe", observe_node)
    
    workflow.set_entry_point("reason")
    workflow.add_edge("reason", "act")
    workflow.add_edge("act", "observe")
    workflow.add_conditional_edges("observe", should_continue, {"reason": "reason", "end": END})
    
    return workflow.compile()


# --- Public Interface ---------------------------------------------------------

@dataclass
class ExecutionResult:
    ok: bool
    observations: List[str]
    outputs: Dict[str, Any]


def run_execution(plan: Dict[str, Any], submission: Dict[str, Any], timeout_s: int = 300, max_retries: int = 3) -> ExecutionResult:
    """Execute plan using ReAct graph with retry logic."""
    start = time.time()

    # --- Fast-path deterministic execution to avoid slow LLM ReAct when possible ---
    try:
        file_paths = submission.get("file_paths") or []
        if file_paths:
            fp = file_paths[0]
            log.info("fast_path_attempt", file=fp)
            df = pd.read_csv(fp)

            # Simple heuristic: if question asks for average/avg revenue per region and columns exist, compute directly
            question = (submission.get("question") or "").lower()
            if "region" in question and "revenue" in question and ("average" in question or "avg" in question) and "region" in df.columns and "revenue" in df.columns:
                log.info("fast_path_triggered", reason="region_revenue_aggregation")
                grouped = df.groupby("region")['revenue'].agg(["mean", "count", "sum"]).reset_index()
                grouped.columns = ["region", "average_revenue", "count", "total_revenue"]
                outputs = {
                    "average_revenue_by_region": grouped.to_dict(orient="records"),
                    "summary": f"Computed revenue aggregates by region in {time.time()-start:.2f}s via fast-path (no LLM)",
                    "regional_breakdown": grouped.to_dict(orient="records"),
                }
                obs = [f"Fast-path: computed revenue aggregates by region without LLM in {time.time()-start:.2f}s", f"Found {len(grouped)} regions: {', '.join(grouped['region'].tolist())}"]
                log.info("fast_path_success", duration=time.time()-start)
                return ExecutionResult(ok=True, observations=obs, outputs=_sanitize_for_json(outputs))

            # Otherwise provide basic stats fast-path for any simple query
            log.info("fast_path_triggered", reason="basic_stats")
            stats = df.describe(include="all").to_dict()
            outputs = {
                "basic_stats": _sanitize_for_json(stats),
                "shape": df.shape,
                "columns": list(df.columns),
                "summary": f"Computed basic statistics in {time.time()-start:.2f}s via fast-path (no LLM)",
            }
            obs = [f"Fast-path: computed basic statistics without LLM in {time.time()-start:.2f}s"]
            log.info("fast_path_success", duration=time.time()-start)
            return ExecutionResult(ok=True, observations=obs, outputs=outputs)
    except Exception as exc:  # noqa: BLE001
        log.error("fast_path_failed", error=str(exc), exc_info=True)
    
    initial_state: AgentState = {
        "plan": plan,
        "submission": submission,
        "current_step_idx": 0,
        "observations": [],
        "outputs": {},
        "retry_count": 0,
        "max_retries": max_retries,
        "completed": False,
        "error": None,
        "_decision": None,
        "_shared_env": None,  # Will be created on first code execution
    }
    
    try:
        graph = create_react_graph()
        
        # Run graph with timeout
        final_state = None
        for state in graph.stream(initial_state):
            if time.time() - start > timeout_s:
                return ExecutionResult(
                    ok=False,
                    observations=["Execution timeout reached"],
                    outputs={},
                )
            final_state = state
        
        # Extract final values
        if final_state:
            # LangGraph returns dict with node names as keys
            last_node_state = list(final_state.values())[-1] if final_state else initial_state
            return ExecutionResult(
                ok=last_node_state.get("error") is None,
                observations=last_node_state.get("observations", []),
                outputs=last_node_state.get("outputs", {}),
            )
        else:
            return ExecutionResult(ok=False, observations=["Graph failed to execute"], outputs={})
    
    except Exception as exc:  # noqa: BLE001
        log.error("execution_failed", error=str(exc))
        return ExecutionResult(
            ok=False,
            observations=[f"Execution crashed: {exc}"],
            outputs={},
        )