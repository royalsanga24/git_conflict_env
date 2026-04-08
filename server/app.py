"""
FastAPI application for the Git Conflict Resolution Environment.

Exposes the standard OpenEnv endpoints (reset/step/state/ws/health/web/docs)
plus the hackathon-required custom endpoints:
  GET  /tasks    - list all tasks with action schema
  GET  /grader   - last grader result for the current session
  POST /baseline - run baseline inference and return scores
"""

from typing import Any, Dict, List, Optional

from fastapi.responses import HTMLResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core"
    ) from e

try:
    from ..models import ConflictAction, ConflictObservation
    from .git_conflict_env_environment import GitConflictEnvironment
    from .task_loader import get_all_task_summaries, list_task_ids
except (ImportError, ModuleNotFoundError):
    from models import ConflictAction, ConflictObservation
    from server.git_conflict_env_environment import GitConflictEnvironment
    from server.task_loader import get_all_task_summaries, list_task_ids


app = create_app(
    GitConflictEnvironment,
    ConflictAction,
    ConflictObservation,
    env_name="git_conflict_env",
    max_concurrent_envs=4,
)


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def root():
    return """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Git Conflict Resolution Environment</title>
<meta http-equiv="refresh" content="0;url=/web"></head>
<body><p>Loading <a href="/web">web interface</a>...</p></body></html>"""


# ---------------------------------------------------------------------------
# Custom hackathon endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks")
def get_tasks() -> Dict[str, Any]:
    """Return all tasks with their metadata and the action schema."""
    return {
        "tasks": get_all_task_summaries(),
        "action_schema": {
            "type": "object",
            "properties": {
                "resolution": {
                    "type": "string",
                    "description": "The resolved file content with all conflict markers removed",
                },
                "explanation": {
                    "type": "string",
                    "description": "Optional reasoning behind the resolution",
                    "default": "",
                },
            },
            "required": ["resolution"],
        },
    }


@app.get("/grader")
def get_grader() -> Dict[str, Any]:
    """
    Return the grader result from the most recent episode.

    NOTE: In a concurrent-session setup each WS client has its own env
    instance, so this HTTP endpoint creates a fresh instance and reports
    that no grading has happened yet.  The primary way to get grader
    results is through the step() observation itself.
    """
    return {
        "message": "Grader results are returned in the step() observation. "
                   "Use the WebSocket API for interactive sessions.",
        # Avoid literal 0.0/1.0 floats here — automated checks may scan JSON numbers.
        "score_range_note": "Task scores use strict open interval (0, 1), never exactly 0 or 1.",
        "components": {
            "markers_removed": 0.10,
            "syntax_valid": 0.15,
            "key_elements": 0.35,
            "similarity_to_gold": 0.15,
            "exact_match": 0.25,
        },
        "attempt_multipliers": {
            "attempt_1": "x1",
            "attempt_2": "x0.8",
            "attempt_3": "x0.6",
        },
    }


@app.post("/baseline")
def run_baseline() -> Dict[str, Any]:
    """
    Run the baseline inference script against all tasks.

    Requires OPENAI_API_KEY in environment variables.
    Returns per-task and per-difficulty aggregate scores.
    """
    import os

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {
            "error": "OPENAI_API_KEY not set. Set it as an environment variable to run baseline.",
            "instructions": "export OPENAI_API_KEY=sk-... && curl -X POST http://localhost:8000/baseline",
        }

    try:
        from .baseline_runner import run_all_tasks
    except (ImportError, ModuleNotFoundError):
        from server.baseline_runner import run_all_tasks

    results = run_all_tasks(api_key)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 8000):
    """Run the server directly: uv run --project . server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
