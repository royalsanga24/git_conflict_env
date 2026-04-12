"""Regression: Phase-2 validator loop on openenv.yaml grader classes."""

import importlib
from pathlib import Path

import yaml

from git_conflict_env.server.phase2_validate import run_harness


def test_phase2_grader_harness_all_tasks_in_range():
    assert run_harness() == 0


def test_harness_probe_scores_are_not_all_identical():
    """Hackathon DQ: graders must not always return the same score."""
    root = Path(__file__).resolve().parent.parent
    data = yaml.safe_load((root / "openenv.yaml").read_text(encoding="utf-8"))
    scores = []
    for t in data.get("tasks", []):
        gpath = t["grader"]
        mod_name, cls_name = gpath.rsplit(":", 1)
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        scores.append(float(cls().grade(None)))
    assert len(set(scores)) >= 3, "probe scores should vary across tasks"
