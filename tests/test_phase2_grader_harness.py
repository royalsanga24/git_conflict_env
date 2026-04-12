"""Regression: Phase-2 validator loop on openenv.yaml grader classes."""

from git_conflict_env.server.phase2_validate import run_harness


def test_phase2_grader_harness_all_tasks_in_range():
    assert run_harness() == 0
