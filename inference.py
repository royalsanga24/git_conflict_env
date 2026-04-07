#!/usr/bin/env python3
"""
Inference script for the Git Conflict Resolution Environment.

Hackathon env vars:
  API_BASE_URL  - environment endpoint
  MODEL_NAME    - model identifier
  HF_TOKEN      - API key (used for OpenAI calls)

Emits structured [START], [STEP], [END] logs.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from client import GitConflictEnv
from models import ConflictAction

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
API_KEY = os.environ.get("OPENAI_API_KEY", "") or HF_TOKEN
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")

BENCHMARK = "git_conflict_env"
MAX_STEPS = 3
SUCCESS_SCORE_THRESHOLD = 0.5

TASK_IDS = [
    "easy_001", "easy_002", "easy_003", "easy_004", "easy_005",
    "medium_001", "medium_002", "medium_003", "medium_004", "medium_005",
    "hard_001", "hard_002", "hard_003", "hard_004", "hard_005",
]

SYSTEM_PROMPT = (
    "You are an expert software developer resolving a git merge conflict. "
    "Return only resolved file contents. No markdown fences, no commentary."
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] {json.dumps({'task': task, 'env': env, 'model': model})}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    entry = {
        'step': step,
        'action': action[:500] if action else '',
        'reward': reward,
        'done': done,
        'error': error,
    }
    print(f"[STEP] {json.dumps(entry)}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    entry = {
        'success': success,
        'steps': steps,
        'score': round(score, 4),
        'rewards': [round(r, 4) for r in rewards],
    }
    print(f"[END] {json.dumps(entry)}", flush=True)


def _build_prompt(obs: Any, feedback: str = "") -> str:
    parts = [
        f"Language: {getattr(obs, 'language', 'text')}",
        f"File: {getattr(obs, 'file_path', '')}",
        f"HEAD branch intent: {getattr(obs, 'ours_description', '')}",
        f"Incoming branch intent: {getattr(obs, 'theirs_description', '')}",
        f"Conflicted file:\n```\n{getattr(obs, 'conflict_file', '')}\n```",
    ]
    base_content = getattr(obs, 'base_content', None)
    if base_content:
        parts.append(f"Base content:\n```\n{base_content}\n```")
    if feedback:
        parts.append(f"Previous feedback:\n{feedback}")
    parts.append("Produce the resolved file.")
    return "\n\n".join(parts)


def _strip_fences(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).rstrip() + "\n"


def _fallback_resolution(conflict_text: str) -> str:
    out: List[str] = []
    for line in conflict_text.splitlines():
        if line.startswith("<<<<<<<") or line.startswith("=======") or line.startswith(">>>>>>>"):
            continue
        out.append(line)
    return "\n".join(out).rstrip() + "\n"


def get_model_resolution(client: OpenAI, obs: Any, feedback: str = "") -> str:
    prompt = _build_prompt(obs, feedback)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt},
            ],
            temperature=0,
        )
        text = resp.choices[0].message.content or ""
        return _strip_fences(text)
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return _fallback_resolution(getattr(obs, 'conflict_file', ''))


def run_task(task_id: str, client: OpenAI) -> Dict[str, Any]:
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        with GitConflictEnv(base_url=API_BASE_URL).sync() as env:
            result = env.reset(task_id=task_id)
            feedback = ""

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                resolution = get_model_resolution(client, result.observation, feedback)
                result = env.step(ConflictAction(resolution=resolution, explanation=""))

                reward = float(result.reward or 0.0)
                done = bool(result.done)
                feedback = getattr(result.observation, 'feedback', '')

                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=resolution, reward=reward, done=done)

                if done:
                    break

            score = max(rewards) if rewards else 0.0
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        log_step(step=steps_taken + 1, action='', reward=0.0, done=True, error=str(exc))

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {'task_id': task_id, 'score': round(score, 4), 'steps': steps_taken, 'success': success}


def main() -> None:
    if not API_KEY:
        print('[DEBUG] Warning: missing HF_TOKEN/OPENAI_API_KEY; LLM calls will fail and use fallback.', flush=True)

    kwargs: Dict[str, Any] = {'api_key': API_KEY or 'sk-placeholder'}
    if OPENAI_BASE_URL:
        kwargs['base_url'] = OPENAI_BASE_URL
    llm_client = OpenAI(**kwargs)

    summary = [run_task(task_id, llm_client) for task_id in TASK_IDS]
    overall = sum(item['score'] for item in summary) / len(summary) if summary else 0.0

    print('\n' + '=' * 60, flush=True)
    print(f'Overall score: {overall:.4f}', flush=True)
    print(json.dumps({'summary': summary, 'overall_score': round(overall, 4)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
