"""
Deterministic grader for the Git Conflict Resolution Environment.

Scoring components (summed, capped at 1.0):
  0.10  - All conflict markers removed
  0.15  - Output is syntactically valid for the file's language
  0.35  - Key elements from the task are present (proportional)
  0.15  - Normalized textual similarity to the gold resolution
  0.25  - Exact match (after whitespace normalization)

Each component also produces a human-readable feedback line.
"""

from __future__ import annotations

import ast
import json
import re
from difflib import SequenceMatcher
from typing import List, Tuple

try:
    from ..models import STRICT_SCORE_MAX, STRICT_SCORE_MIN
except (ImportError, ModuleNotFoundError):
    from models import STRICT_SCORE_MAX, STRICT_SCORE_MIN

CONFLICT_MARKERS = ["<<<<<<<", "=======", ">>>>>>>"]

W_MARKERS = 0.10
W_SYNTAX = 0.15
W_ELEMENTS = 0.35
W_SIMILARITY = 0.15
W_EXACT = 0.25
MIN_STRICT_SCORE = STRICT_SCORE_MIN
MAX_STRICT_SCORE = STRICT_SCORE_MAX


def grade(agent_resolution: str, task: dict) -> Tuple[float, str]:
    """
    Grade an agent's conflict resolution against a task's gold standard.

    Returns (score, feedback) where score is in [0.0, 1.0] and feedback
    is a multi-line human-readable breakdown.
    """
    gold = task["gold_resolution"]
    language = task.get("language", "text")
    key_elements = task.get("key_elements", [])

    score = 0.0
    feedback: List[str] = []

    # --- Component 1: conflict markers removed ---
    markers_score, markers_fb = _check_markers(agent_resolution)
    score += markers_score
    feedback.append(markers_fb)

    # --- Component 2: syntactic validity ---
    syntax_score, syntax_fb = _check_syntax(agent_resolution, language)
    score += syntax_score
    feedback.append(syntax_fb)

    # --- Component 3: key elements preserved ---
    elements_score, elements_fb = _check_key_elements(agent_resolution, key_elements)
    score += elements_score
    feedback.extend(elements_fb)

    # --- Component 4: similarity to gold ---
    sim_score, sim_fb = _check_similarity(agent_resolution, gold)
    score += sim_score
    feedback.append(sim_fb)

    # --- Component 5: exact match ---
    exact_score, exact_fb = _check_exact_match(agent_resolution, gold)
    score += exact_score
    feedback.append(exact_fb)

    # Phase-2 validator requires strict (0, 1) task scores.
    final_score = round(min(max(score, MIN_STRICT_SCORE), MAX_STRICT_SCORE), 4)
    return final_score, "\n".join(feedback)


# ---------------------------------------------------------------------------
# Component helpers
# ---------------------------------------------------------------------------

def _check_markers(text: str) -> Tuple[float, str]:
    found = [m for m in CONFLICT_MARKERS if m in text]
    if not found:
        return W_MARKERS, f"[+{W_MARKERS:.2f}] PASS: All conflict markers removed"
    return 0.0, f"[+none] FAIL: Conflict markers still present: {', '.join(found)}"


def _check_syntax(text: str, language: str) -> Tuple[float, str]:
    lang = language.lower().strip()
    if lang == "python":
        return _syntax_python(text)
    if lang == "json":
        return _syntax_json(text)
    if lang in ("yaml", "yml"):
        return _syntax_yaml(text)
    # For languages we can't parse (js, css, text, etc.) give credit if
    # the output is non-empty and has no conflict markers.
    if text.strip():
        return W_SYNTAX, f"[+{W_SYNTAX:.2f}] PASS: Non-empty output (syntax not checked for {lang})"
    return 0.0, "[+none] FAIL: Empty output"


def _syntax_python(text: str) -> Tuple[float, str]:
    try:
        ast.parse(text)
        return W_SYNTAX, f"[+{W_SYNTAX:.2f}] PASS: Valid Python syntax"
    except SyntaxError as e:
        return 0.0, f"[+none] FAIL: Python syntax error at line {e.lineno}: {e.msg}"


def _syntax_json(text: str) -> Tuple[float, str]:
    try:
        json.loads(text)
        return W_SYNTAX, f"[+{W_SYNTAX:.2f}] PASS: Valid JSON"
    except json.JSONDecodeError as e:
        return 0.0, f"[+none] FAIL: Invalid JSON: {e.msg} (line {e.lineno})"


def _syntax_yaml(text: str) -> Tuple[float, str]:
    try:
        import yaml  # noqa: F811
        yaml.safe_load(text)
        return W_SYNTAX, f"[+{W_SYNTAX:.2f}] PASS: Valid YAML"
    except Exception as e:
        return 0.0, f"[+none] FAIL: Invalid YAML: {e}"


def _check_key_elements(
    text: str, key_elements: List[dict]
) -> Tuple[float, List[str]]:
    if not key_elements:
        return 0.0, ["[+none] SKIP: No key elements defined"]

    found = 0
    feedback: List[str] = []

    for elem in key_elements:
        desc = elem["description"]

        if "anti_pattern" in elem:
            # Anti-patterns: the text should NOT contain this
            if elem["anti_pattern"] not in text:
                found += 1
                feedback.append(f"  PASS: {desc}")
            else:
                feedback.append(f"  MISS: {desc} (unwanted pattern found)")
        else:
            pattern = elem["pattern"]
            if pattern in text:
                found += 1
                feedback.append(f"  PASS: {desc}")
            else:
                feedback.append(f"  MISS: {desc}")

    ratio = found / len(key_elements)
    element_score = round(W_ELEMENTS * ratio, 4)
    header = f"[+{element_score:.2f}] Key elements: {found}/{len(key_elements)}"
    return element_score, [header] + feedback


def _check_similarity(agent: str, gold: str) -> Tuple[float, str]:
    ratio = SequenceMatcher(None, _normalize(agent), _normalize(gold)).ratio()
    sim_score = round(W_SIMILARITY * ratio, 4)
    return sim_score, f"[+{sim_score:.2f}] Similarity to gold: {ratio:.1%}"


def _check_exact_match(agent: str, gold: str) -> Tuple[float, str]:
    if _normalize(agent) == _normalize(gold):
        return W_EXACT, f"[+{W_EXACT:.2f}] PERFECT: Exact match with expected resolution"
    return 0.0, "[+none] No exact match"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Normalize whitespace for comparison: strip trailing per-line, collapse blank lines."""
    lines = [line.rstrip() for line in text.splitlines()]
    # Remove leading/trailing blank lines
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)
