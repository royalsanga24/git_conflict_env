---
title: Git Conflict Resolution Environment
emoji: 🔀
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Git Conflict Resolution Environment

An OpenEnv environment that trains and evaluates AI agents on resolving real git merge conflicts. Agents receive a file with conflict markers and branch descriptions, then submit a resolved version scored by a deterministic grader.

## Motivation

Merge conflict resolution is one of the most common and time-consuming tasks in software development. Every developer encounters conflicts when merging branches, yet no standardized RL environment exists for training AI agents on this task. This environment fills that gap by providing:

- **Realistic conflicts** sourced from common development patterns (feature additions, bug fixes, refactors, dependency upgrades)
- **Multi-language support**: Python, JavaScript, CSS, JSON, YAML, TypeScript, and plain text
- **Deterministic grading** with fine-grained partial credit
- **Difficulty progression** from trivial (keep-both) to genuinely hard (semantic intent merging)

## Quick Start

```python
from git_conflict_env import ConflictAction, GitConflictEnv

with GitConflictEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_id="easy_001")
    obs = result.observation

    print(f"Task: {obs.task_id} ({obs.difficulty})")
    print(f"File: {obs.file_path} ({obs.language})")
    print(f"Ours: {obs.ours_description}")
    print(f"Theirs: {obs.theirs_description}")
    print(obs.conflict_file)

    result = env.step(ConflictAction(resolution="...your resolved code..."))
    print(f"Score: {result.observation.score}")
    print(result.observation.feedback)
```

## Action Space

| Field | Type | Description |
|---|---|---|
| `resolution` | `str` (required) | The resolved file content with all conflict markers removed |
| `explanation` | `str` (optional) | Reasoning behind the resolution |

## Observation Space

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Task identifier (e.g. `easy_001`) |
| `difficulty` | `str` | `easy`, `medium`, or `hard` |
| `file_path` | `str` | Simulated file path (e.g. `utils/helpers.py`) |
| `language` | `str` | Programming language or format |
| `conflict_file` | `str` | File with `<<<<<<<`/`=======`/`>>>>>>>` markers |
| `ours_description` | `str` | What the HEAD branch intended |
| `theirs_description` | `str` | What the incoming branch intended |
| `base_content` | `str?` | Common ancestor (medium/hard only) |
| `feedback` | `str` | Grader feedback after submission |
| `attempts_remaining` | `int` | Remaining attempts (max 3) |
| `score` | `float?` | Grader score 0.0-1.0 after submission |
| `done` | `bool` | Whether episode is over |
| `reward` | `float` | Attempt-adjusted reward |

## Tasks

### Easy (5 tasks)

Simple "keep-both" patterns where both sides added independent code.

| ID | Scenario | Language |
|---|---|---|
| `easy_001` | Both sides added different utility functions | Python |
| `easy_002` | Both sides added different imports | Python |
| `easy_003` | Both sides added entries to a JSON config list | JSON |
| `easy_004` | One side updated docstring, other fixed a bug | Python |
| `easy_005` | Both sides bumped different dependency versions | Text |

### Medium (5 tasks)

Requires understanding what both sides intended to produce a correct merge.

| ID | Scenario | Language |
|---|---|---|
| `medium_001` | Both modified function parameters differently | Python |
| `medium_002` | Variable rename on one side, new logic on other | JavaScript |
| `medium_003` | Both modified CSS class with different properties | CSS |
| `medium_004` | One added error handling, other added new code path | Python |
| `medium_005` | Both modified docker-compose service differently | YAML |

### Hard (5 tasks)

Requires understanding the *intent* behind changes and making architectural decisions.

| ID | Scenario | Language |
|---|---|---|
| `hard_001` | Function refactored into class vs. feature added to original | Python |
| `hard_002` | Both sides fixed same bug differently (choose better fix) | Python |
| `hard_003` | Three-hunk conflict with interacting changes | Python |
| `hard_004` | One deleted a function, other modified and extended it | Python |
| `hard_005` | Conflicting API endpoints with different schemas | Python |

## Reward Design

### Scoring Components

Each resolution is scored across five deterministic components:

| Component | Weight | What it Measures |
|---|---|---|
| Conflict markers removed | 0.10 | Are all `<<<<<<<`, `=======`, `>>>>>>>` removed? |
| Syntax validity | 0.15 | Does the output parse without errors? (ast/json/yaml) |
| Key elements preserved | 0.35 | Are critical code patterns from both sides present? |
| Similarity to gold | 0.15 | Textual similarity to the expected resolution |
| Exact match | 0.25 | Whitespace-normalized match to gold |

### Attempt Multipliers

Agents get up to 3 attempts per task with feedback after each. The maximum reward decreases with each attempt to reward getting it right the first time:

| Attempt | Multiplier |
|---|---|
| 1 | 1.0x |
| 2 | 0.8x |
| 3 | 0.6x |

The episode ends when the agent achieves a perfect score or exhausts all attempts.

## Setup & Usage

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- `openenv-core` package

### Install

```bash
pip install openenv-core openai pyyaml
```

### Run Locally

```bash
cd git_conflict_env
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run with Docker

```bash
cd git_conflict_env
docker build -t git-conflict-env:latest -f server/Dockerfile .
docker run -p 8000:8000 git-conflict-env:latest
```

### Run Baseline

```bash
export OPENAI_API_KEY=sk-...
python baseline.py
python baseline.py --model gpt-4o-mini
```

### Deploy to Hugging Face Spaces

```bash
cd git_conflict_env
openenv push --repo-id yourname/git-conflict-env
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/ws` | WS | WebSocket for persistent sessions |
| `/reset` | POST | Start a new episode |
| `/step` | POST | Submit a resolution |
| `/state` | GET | Get current episode state |
| `/tasks` | GET | List all tasks with action schema |
| `/grader` | GET | Scoring component breakdown |
| `/baseline` | POST | Run model inference on all tasks |
| `/web` | GET | Interactive web UI |
| `/docs` | GET | OpenAPI documentation |

## Baseline Scores

Baseline run using `gpt-4o` with `temperature=0` for reproducibility:

| Task | Difficulty | Score |
|---|---|---|
| easy_001 | easy | 0.7497 |
| easy_002 | easy | 1.0000 |
| easy_003 | easy | 1.0000 |
| easy_004 | easy | 1.0000 |
| easy_005 | easy | 1.0000 |
| medium_001 | medium | 0.7499 |
| medium_002 | medium | 1.0000 |
| medium_003 | medium | 0.6763 |
| medium_004 | medium | 0.7191 |
| medium_005 | medium | 1.0000 |
| hard_001 | hard | 0.7454 |
| hard_002 | hard | 0.5316 |
| hard_003 | hard | 0.7497 |
| hard_004 | hard | 0.7469 |
| hard_005 | hard | 0.7499 |

| Aggregate | Score |
|---|---|
| **Easy avg** | 0.9499 |
| **Medium avg** | 0.8291 |
| **Hard avg** | 0.7047 |
| **Overall avg** | 0.8279 |

## Project Structure

```
git_conflict_env/
├── __init__.py              # Package exports
├── models.py                # ConflictAction, ConflictObservation, ConflictState
├── client.py                # GitConflictEnv WebSocket client
├── baseline.py              # Standalone baseline inference script
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Package configuration
├── tasks/
│   ├── easy.json            # 5 easy conflict scenarios
│   ├── medium.json          # 5 medium conflict scenarios
│   └── hard.json            # 5 hard conflict scenarios
└── server/
    ├── environment.py       # Core reset/step/state logic
    ├── grader.py            # Deterministic scoring engine
    ├── task_loader.py       # Task index and lookup
    ├── baseline_runner.py   # Inference logic for /baseline
    ├── app.py               # FastAPI application
    ├── Dockerfile           # Container definition
    └── requirements.txt     # Server dependencies
```
