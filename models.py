"""
Data models for the Git Conflict Resolution Environment.

Defines typed Action, Observation, and State contracts for an environment
where agents resolve git merge conflicts in source code files.
"""

from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

# Phase-2 validators require every emitted score/reward to lie strictly in (0, 1).
STRICT_SCORE_MIN = 0.0001
STRICT_SCORE_MAX = 0.9999


class ConflictAction(Action):
    """Action submitted by the agent: a proposed conflict resolution."""

    resolution: str = Field(..., description="The resolved file content with all conflict markers removed")
    explanation: str = Field(default="", description="Optional reasoning behind the resolution")


class ConflictObservation(Observation):
    """Observation returned to the agent after reset or step."""

    task_id: str = Field(default="", description="Unique task identifier, e.g. 'easy_001'")
    difficulty: str = Field(default="", description="Task difficulty: easy, medium, or hard")
    file_path: str = Field(default="", description="Simulated file path, e.g. 'utils/helpers.py'")
    language: str = Field(default="", description="Programming language or file format")
    conflict_file: str = Field(default="", description="File content WITH conflict markers to resolve")
    ours_description: str = Field(default="", description="What the HEAD/ours branch intended")
    theirs_description: str = Field(default="", description="What the incoming/theirs branch intended")
    base_content: Optional[str] = Field(default=None, description="Common ancestor content (provided for medium+hard)")
    feedback: str = Field(default="", description="Grader feedback after a step submission")
    attempts_remaining: int = Field(default=3, description="Number of resolution attempts left")
    score: Optional[float] = Field(
        default=None,
        description=f"Grader score in open interval ({STRICT_SCORE_MIN}, {STRICT_SCORE_MAX}) after submission",
    )


class ConflictState(State):
    """Episode state tracking for the conflict resolution environment."""

    current_task_id: str = ""
    current_difficulty: str = ""
    resolved: bool = False
    best_score: float = Field(
        default=STRICT_SCORE_MIN,
        description="Best score so far; never 0.0 (strict open interval for validators).",
    )
    attempt_number: int = 0
    max_attempts: int = 3
