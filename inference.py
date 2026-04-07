#!/usr/bin/env python3
"""
Inference script for the Git Conflict Resolution Environment.

Hackathon env vars:
  API_BASE_URL  - LiteLLM proxy endpoint for OpenAI client (required)
  API_KEY       - LiteLLM proxy key (required)
  MODEL_NAME    - model identifier
  LOCAL_IMAGE_NAME - optional local Docker image for environment execution

Local testing env var:
  ENV_BASE_URL  - environment server URL (default: http://localhost:8000)

Emits structured [START], [STEP], [END] logs.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from client import GitConflictEnv
from models import ConflictAction

API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000").rstrip("/")

BENCHMARK = "git_conflict_env"
MAX_STEPS = 3
SUCCESS_SCORE_THRESHOLD = 0.5
MIN_TASK_SCORE = 0.0001
MAX_TASK_SCORE = 0.9999

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
    score = MIN_TASK_SCORE

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        env_client = (
            GitConflictEnv.from_docker_image(LOCAL_IMAGE_NAME)
            if LOCAL_IMAGE_NAME
            else GitConflictEnv(base_url=ENV_BASE_URL)
        )

        with env_client.sync() as env:
            result = env.reset(task_id=task_id)
            feedback = ""

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                resolution = get_model_resolution(client, result.observation, feedback)
                result = env.step(ConflictAction(resolution=resolution, explanation=""))

                reward = min(max(float(result.reward or MIN_TASK_SCORE), MIN_TASK_SCORE), MAX_TASK_SCORE)
                done = bool(result.done)
                feedback = getattr(result.observation, 'feedback', '')

                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=resolution, reward=reward, done=done)

                if done:
                    break

            raw_score = max(rewards) if rewards else MIN_TASK_SCORE
            # Validator requires each task score strictly within (0, 1).
            score = min(max(raw_score, MIN_TASK_SCORE), MAX_TASK_SCORE)
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        score = MIN_TASK_SCORE
        log_step(step=steps_taken + 1, action='', reward=MIN_TASK_SCORE, done=True, error=str(exc))

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {'task_id': task_id, 'score': round(score, 4), 'steps': steps_taken, 'success': success}


def main() -> None:
    if not API_BASE_URL or not API_KEY:
        print("[DEBUG] Warning: API_BASE_URL/API_KEY missing; proxy LLM call will fail and fallback will be used.", flush=True)

    # Required by validator: use injected LiteLLM proxy variables exactly.
    llm_client = OpenAI(
        base_url=API_BASE_URL or "https://invalid.local",
        api_key=API_KEY or "invalid-key",
    )

    summary = [run_task(task_id, llm_client) for task_id in TASK_IDS]
    overall = sum(item['score'] for item in summary) / len(summary) if summary else 0.0

    print('\n' + '=' * 60, flush=True)
    print(f'Overall score: {overall:.4f}', flush=True)
    print(json.dumps({'summary': summary, 'overall_score': round(overall, 4)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
