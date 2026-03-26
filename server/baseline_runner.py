"""
Baseline inference runner for the Git Conflict Resolution Environment.

Uses the OpenAI chat completions API to resolve each conflict task,
then grades the result with the deterministic grader.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from openai import OpenAI

try:
    from .grader import grade
    from .task_loader import get_all_tasks
except (ImportError, ModuleNotFoundError):
    from grader import grade
    from task_loader import get_all_tasks

SYSTEM_PROMPT = """\
You are an expert software developer resolving a git merge conflict.

You will receive:
1. A file with git conflict markers (<<<<<<< HEAD, =======, >>>>>>> branch-name)
2. A description of what each branch intended
3. The programming language / file format

Your job:
- Produce the correctly resolved file content
- Remove ALL conflict markers
- Preserve the intent of BOTH branches when possible
- If the two changes are incompatible, choose the better approach and integrate what you can from the other
- The result must be syntactically valid
- Output ONLY the resolved file content — no explanations, no markdown fences, no commentary
"""


def resolve_one(client: OpenAI, task: dict, model: str = "gpt-4o") -> str:
    """Send a single conflict task to the model and return the raw resolution."""
    user_msg = (
        f"Language: {task['language']}\n"
        f"File: {task['file_path']}\n\n"
        f"HEAD branch intent: {task['ours_description']}\n"
        f"Incoming branch intent: {task['theirs_description']}\n\n"
        f"Conflicted file:\n```\n{task['conflict_file']}\n```\n\n"
        f"Produce the resolved file:"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content or ""

    # Strip markdown fences if the model wraps the output
    lines = raw.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines) + "\n"


def run_all_tasks(
    api_key: str,
    model: str = "gpt-4o",
) -> Dict[str, Any]:
    """
    Run baseline inference over every task and return structured results.

    Returns:
        {
          "model": "gpt-4o",
          "summary": {"easy": avg, "medium": avg, "hard": avg, "overall": avg},
          "tasks": [ {id, difficulty, score, feedback}, ... ]
        }
    """
    client = OpenAI(api_key=api_key)
    all_tasks = get_all_tasks()

    task_results: List[Dict[str, Any]] = []
    by_difficulty: Dict[str, List[float]] = {"easy": [], "medium": [], "hard": []}

    for task_id, task in all_tasks.items():
        resolution = resolve_one(client, task, model=model)
        score, feedback = grade(resolution, task)
        task_results.append({
            "id": task_id,
            "difficulty": task["difficulty"],
            "title": task["title"],
            "score": score,
            "feedback": feedback,
        })
        by_difficulty[task["difficulty"]].append(score)

    summary = {}
    all_scores = []
    for diff in ("easy", "medium", "hard"):
        scores = by_difficulty[diff]
        avg = round(sum(scores) / len(scores), 4) if scores else 0.0
        summary[diff] = avg
        all_scores.extend(scores)
    summary["overall"] = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0

    return {
        "model": model,
        "summary": summary,
        "tasks": task_results,
    }
