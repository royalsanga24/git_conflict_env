"""
Git Conflict Resolution Environment.

An OpenEnv environment where agents resolve git merge conflicts.
Each episode presents a file with conflict markers; the agent
submits a resolved version and receives a deterministic score.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        ConflictAction,
        ConflictObservation,
        ConflictState,
        STRICT_SCORE_MAX,
        STRICT_SCORE_MIN,
    )
except (ImportError, ModuleNotFoundError):
    from models import (
        ConflictAction,
        ConflictObservation,
        ConflictState,
        STRICT_SCORE_MAX,
        STRICT_SCORE_MIN,
    )

try:
    from .grader import grade
    from .task_loader import get_all_tasks, get_task, list_task_ids
except (ImportError, ModuleNotFoundError):
    from grader import grade
    from task_loader import get_all_tasks, get_task, list_task_ids

MAX_ATTEMPTS = 3
ATTEMPT_MULTIPLIERS = {1: 1, 2: 0.8, 3: 0.6}
MIN_STRICT_SCORE = STRICT_SCORE_MIN
MAX_STRICT_SCORE = STRICT_SCORE_MAX


class GitConflictEnvironment(Environment):
    """
    Environment for resolving git merge conflicts.

    Episode lifecycle:
      1. reset(task_id=...) -> load a conflict task, return the observation
      2. step(resolution=...) -> grade the resolution, return score + feedback
         - If perfect score or no attempts left: done = True
         - Otherwise: done = False, agent can retry with feedback
      3. state() -> current episode metadata
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = ConflictState(episode_id=str(uuid4()), step_count=0)
        self._current_task: dict | None = None
        self._last_grader_result: dict | None = None

    def reset(self, *, task_id: str | None = None, **kwargs) -> ConflictObservation:
        """
        Start a new episode. If task_id is given, load that specific task.
        Otherwise pick the first available task.
        """
        if task_id is None:
            all_ids = list_task_ids()
            if not all_ids:
                raise RuntimeError("No tasks loaded")
            task_id = all_ids[0]

        task = get_task(task_id)
        self._current_task = task
        self._last_grader_result = None

        self._state = ConflictState(
            episode_id=str(uuid4()),
            step_count=0,
            current_task_id=task["id"],
            current_difficulty=task["difficulty"],
            resolved=False,
            best_score=MIN_STRICT_SCORE,
            attempt_number=0,
            max_attempts=MAX_ATTEMPTS,
        )

        show_base = task["difficulty"] in ("medium", "hard") and task.get("base_content")

        return ConflictObservation(
            done=False,
            reward=MIN_STRICT_SCORE,
            task_id=task["id"],
            difficulty=task["difficulty"],
            file_path=task["file_path"],
            language=task["language"],
            conflict_file=task["conflict_file"],
            ours_description=task["ours_description"],
            theirs_description=task["theirs_description"],
            base_content=task["base_content"] if show_base else None,
            feedback="Resolve the merge conflict and submit your resolution.",
            attempts_remaining=MAX_ATTEMPTS,
            score=None,
        )

    def step(self, action: ConflictAction, **kwargs) -> ConflictObservation:
        """Grade the agent's resolution and return feedback."""
        if self._current_task is None:
            return ConflictObservation(
                done=True,
                reward=MIN_STRICT_SCORE,
                feedback="ERROR: No active task. Call reset() first.",
                attempts_remaining=0,
            )

        if self._state.resolved:
            return ConflictObservation(
                done=True,
                reward=MIN_STRICT_SCORE,
                task_id=self._state.current_task_id,
                difficulty=self._state.current_difficulty,
                feedback="Episode already finished. Call reset() to start a new one.",
                attempts_remaining=0,
                score=self._state.best_score,
            )

        task = self._current_task
        self._state.step_count += 1
        self._state.attempt_number += 1
        attempt = self._state.attempt_number

        raw_score, feedback = grade(action.resolution, task)

        multiplier = ATTEMPT_MULTIPLIERS.get(attempt, 0.5)
        adjusted_score = round(raw_score * multiplier, 4)
        # Phase-2 validator requires strict (0, 1) scores/rewards.
        adjusted_score = min(max(adjusted_score, MIN_STRICT_SCORE), MAX_STRICT_SCORE)

        if adjusted_score > self._state.best_score:
            self._state.best_score = adjusted_score

        self._last_grader_result = {
            "task_id": task["id"],
            "attempt": attempt,
            "raw_score": raw_score,
            "multiplier": multiplier,
            "adjusted_score": adjusted_score,
            "best_score": self._state.best_score,
            "feedback": feedback,
        }

        attempts_left = MAX_ATTEMPTS - attempt
        is_perfect = raw_score >= 0.99
        is_done = is_perfect or attempts_left <= 0

        if is_done:
            self._state.resolved = True

        if is_perfect:
            header = f"Score: {adjusted_score:.2f} (attempt {attempt}, multiplier x{multiplier:g}) - PERFECT"
        elif is_done:
            header = f"Score: {adjusted_score:.2f} (attempt {attempt}, multiplier x{multiplier:g}) - Final attempt"
        else:
            header = (
                f"Score: {adjusted_score:.2f} (attempt {attempt}, multiplier x{multiplier:g}) - "
                f"{attempts_left} attempt(s) remaining"
            )

        full_feedback = f"{header}\n\n{feedback}"

        return ConflictObservation(
            done=is_done,
            reward=adjusted_score,
            task_id=task["id"],
            difficulty=task["difficulty"],
            file_path=task["file_path"],
            language=task["language"],
            conflict_file=task["conflict_file"],
            ours_description=task["ours_description"],
            theirs_description=task["theirs_description"],
            feedback=full_feedback,
            attempts_remaining=attempts_left,
            score=adjusted_score,
        )

    @property
    def state(self) -> ConflictState:
        return self._state

    @property
    def last_grader_result(self) -> dict | None:
        """Expose the most recent grader breakdown for the /grader endpoint."""
        return self._last_grader_result
