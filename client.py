"""Git Conflict Resolution Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import (
        ConflictAction,
        ConflictObservation,
        ConflictState,
        STRICT_SCORE_MIN,
    )
except (ImportError, ModuleNotFoundError):
    from models import (
        ConflictAction,
        ConflictObservation,
        ConflictState,
        STRICT_SCORE_MIN,
    )


class GitConflictEnv(
    EnvClient[ConflictAction, ConflictObservation, ConflictState]
):
    """
    Client for the Git Conflict Resolution Environment.

    Example (sync):
        >>> with GitConflictEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset(task_id="easy_001")
        ...     print(result.observation.conflict_file)
        ...     result = env.step(ConflictAction(resolution="resolved code"))
        ...     print(result.observation.score)

    Example (async):
        >>> async with GitConflictEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(task_id="easy_001")
        ...     result = await env.step(ConflictAction(resolution="..."))
    """

    def _step_payload(self, action: ConflictAction) -> Dict[str, Any]:
        return {
            "resolution": action.resolution,
            "explanation": action.explanation,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ConflictObservation]:
        obs_data = payload.get("observation", {})
        observation = ConflictObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            task_id=obs_data.get("task_id", ""),
            difficulty=obs_data.get("difficulty", ""),
            file_path=obs_data.get("file_path", ""),
            language=obs_data.get("language", ""),
            conflict_file=obs_data.get("conflict_file", ""),
            ours_description=obs_data.get("ours_description", ""),
            theirs_description=obs_data.get("theirs_description", ""),
            base_content=obs_data.get("base_content"),
            feedback=obs_data.get("feedback", ""),
            attempts_remaining=obs_data.get("attempts_remaining", 0),
            score=obs_data.get("score"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ConflictState:
        return ConflictState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            current_task_id=payload.get("current_task_id", ""),
            current_difficulty=payload.get("current_difficulty", ""),
            resolved=payload.get("resolved", False),
            best_score=payload.get("best_score", STRICT_SCORE_MIN),
            attempt_number=payload.get("attempt_number", 0),
            max_attempts=payload.get("max_attempts", 3),
        )
