#!/usr/bin/env python3
"""
Baseline inference entry point required at repo root for hackathon automated checks.

Runs the same OpenAI-based evaluation as `baseline.py`: all tasks, reproducible
scores (temperature=0), credentials from OPENAI_API_KEY.

Usage:
    export OPENAI_API_KEY=sk-...
    python inference.py
    python inference.py --model gpt-4o-mini --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline inference on all conflict tasks (submission entry point)"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted table",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        print("  export OPENAI_API_KEY=sk-...", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from server.baseline_runner import run_all_tasks

    print(f"Running inference with model={args.model} across all tasks...\n")
    results = run_all_tasks(api_key, model=args.model)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print(f"Model: {results['model']}")
    print("=" * 60)
    print(f"{'Task':<15} {'Difficulty':<10} {'Score':<8}")
    print("-" * 60)
    for t in results["tasks"]:
        print(f"{t['id']:<15} {t['difficulty']:<10} {t['score']:<8.4f}")
    print("-" * 60)
    summary = results["summary"]
    print(f"{'Easy avg':<25} {summary['easy']:.4f}")
    print(f"{'Medium avg':<25} {summary['medium']:.4f}")
    print(f"{'Hard avg':<25} {summary['hard']:.4f}")
    print(f"{'Overall avg':<25} {summary['overall']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
