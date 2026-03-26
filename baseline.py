#!/usr/bin/env python3
"""
Baseline inference script for the Git Conflict Resolution Environment.

Runs a model against all tasks using the OpenAI API and reports scores.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline.py
    python baseline.py --model gpt-4o-mini
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Run baseline inference on all conflict tasks")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use (default: gpt-4o)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted table")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        print("  export OPENAI_API_KEY=sk-...", file=sys.stderr)
        sys.exit(1)

    # Import from sibling package
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from server.baseline_runner import run_all_tasks

    print(f"Running baseline with model={args.model} across all tasks...\n")
    results = run_all_tasks(api_key, model=args.model)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    # Formatted output
    print(f"Model: {results['model']}")
    print(f"{'='*60}")
    print(f"{'Task':<15} {'Difficulty':<10} {'Score':<8}")
    print(f"{'-'*60}")
    for t in results["tasks"]:
        print(f"{t['id']:<15} {t['difficulty']:<10} {t['score']:<8.4f}")
    print(f"{'-'*60}")
    summary = results["summary"]
    print(f"{'Easy avg':<25} {summary['easy']:.4f}")
    print(f"{'Medium avg':<25} {summary['medium']:.4f}")
    print(f"{'Hard avg':<25} {summary['hard']:.4f}")
    print(f"{'Overall avg':<25} {summary['overall']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
