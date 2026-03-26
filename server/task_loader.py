"""
Task loader for the Git Conflict Resolution Environment.

Loads and indexes task definitions from the JSON files in the tasks/ directory
so the environment and grader can look up tasks by ID.
"""

import json
import os
from typing import Dict, List, Optional

_TASKS: Dict[str, dict] = {}
_TASKS_BY_DIFFICULTY: Dict[str, List[dict]] = {"easy": [], "medium": [], "hard": []}
_LOADED = False


def _find_tasks_dir() -> str:
    """Locate the tasks/ directory relative to this file or the package root."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "tasks"),
        os.path.join(here, "tasks"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"Could not find tasks/ directory; searched: {candidates}")


def _load_all():
    global _LOADED
    if _LOADED:
        return
    tasks_dir = _find_tasks_dir()
    for difficulty in ("easy", "medium", "hard"):
        path = os.path.join(tasks_dir, f"{difficulty}.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        for task in tasks:
            task_id = task["id"]
            _TASKS[task_id] = task
            _TASKS_BY_DIFFICULTY[difficulty].append(task)
    _LOADED = True


def get_task(task_id: str) -> dict:
    """Return a single task by its ID, or raise KeyError."""
    _load_all()
    if task_id not in _TASKS:
        raise KeyError(f"Unknown task_id: {task_id!r}. Available: {list(_TASKS.keys())}")
    return _TASKS[task_id]


def get_tasks_by_difficulty(difficulty: str) -> List[dict]:
    """Return all tasks for a given difficulty level."""
    _load_all()
    return _TASKS_BY_DIFFICULTY.get(difficulty, [])


def get_all_tasks() -> Dict[str, dict]:
    """Return the full task index keyed by task_id."""
    _load_all()
    return dict(_TASKS)


def list_task_ids() -> List[str]:
    """Return all known task IDs."""
    _load_all()
    return list(_TASKS.keys())


def get_task_summary(task_id: str) -> dict:
    """Return a public-facing summary of a task (no gold resolution)."""
    task = get_task(task_id)
    return {
        "id": task["id"],
        "difficulty": task["difficulty"],
        "title": task["title"],
        "description": task["description"],
        "file_path": task["file_path"],
        "language": task["language"],
    }


def get_all_task_summaries() -> List[dict]:
    """Return public summaries for every task."""
    _load_all()
    return [get_task_summary(tid) for tid in _TASKS]
