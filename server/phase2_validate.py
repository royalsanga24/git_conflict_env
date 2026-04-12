"""
Phase-2 grader harness — matches hackathon validator behavior.

Loads ``openenv.yaml`` ``tasks[].grader``, imports each class, runs
``float(Class().grade(None))``, and requires ``0 < score < 1``.

Run locally::

    uv run validate-phase2

Or::

    uv run python -m git_conflict_env.server.phase2_validate

In Docker (image ``WORKDIR`` is ``/app/env``; console script is on ``PATH``)::

    docker run --rm git-conflict-env validate-phase2

Or::

    docker run --rm git-conflict-env python -m git_conflict_env.server.phase2_validate
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path

import yaml


def _find_openenv_yaml() -> Path | None:
    """Locate manifest (works from repo root, Docker WORKDIR /app/env, or site-packages install)."""
    env_root = os.environ.get("OPENENV_ROOT", "").strip()
    if env_root:
        p = Path(env_root) / "openenv.yaml"
        if p.is_file():
            return p
    candidates: list[Path] = []
    cwd = Path.cwd().resolve()
    candidates.append(cwd / "openenv.yaml")
    docker_env = Path("/app/env/openenv.yaml")
    if docker_env.is_file():
        candidates.append(docker_env)
    # Dev layout: git_repo/server/phase2_validate.py -> repo root
    here = Path(__file__).resolve()
    candidates.append(here.parent.parent / "openenv.yaml")
    cur = cwd
    for _ in range(8):
        candidates.append(cur / "openenv.yaml")
        if cur.parent == cur:
            break
        cur = cur.parent
    for p in candidates:
        if p.is_file():
            return p
    return None


def _import_root_for_server_modules(yaml_path: Path) -> Path:
    """Directory that must be on sys.path for ``import server.grader``."""
    return yaml_path.parent.resolve()


def run_harness() -> int:
    """Return 0 if all tasks pass, 1 otherwise."""
    yaml_path = _find_openenv_yaml()
    if yaml_path is None:
        print(
            "error: openenv.yaml not found (set OPENENV_ROOT or run from repo / Docker WORKDIR /app/env)",
            file=sys.stderr,
        )
        return 2

    root = _import_root_for_server_modules(yaml_path)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    tasks = data.get("tasks") or []
    if not tasks:
        print("error: openenv.yaml has no tasks (Phase-2 requires tasks[].grader)", file=sys.stderr)
        return 1

    failed = False
    for t in tasks:
        tid = t.get("id", "?")
        gpath = t.get("grader", "MISSING")
        print(f"Task: {tid} | grader: {gpath}")
        try:
            mod_name, cls_name = str(gpath).rsplit(":", 1)
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            score = float(cls().grade(None))
            ok = 0 < score < 1
            print(f"  -> {score} {'OK' if ok else 'FAIL'}")
            if not ok:
                failed = True
        except Exception:
            traceback.print_exc()
            print("  -> CRASHED")
            failed = True

    return 1 if failed else 0


def main() -> None:
    raise SystemExit(run_harness())


if __name__ == "__main__":
    main()
