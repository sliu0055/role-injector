#!/usr/bin/env python3
"""
general_expert.py — Query OpenAI with a generic role injection prompt.

For every question in a data/questions_*.json file, sends:
  user:   "Assume you are a {role} with 20 years of experience. {question}"

Results are saved to tool/results/<role>.json with the structure:
  [{"question_id": ..., "role": ..., "question": ..., "answer": ...}, ...]

Usage:
    export OPENAI_API_KEY="your-key"

    # Run a single role file
    python tool/general_expert.py --input data/questions_economist.json

    # Run all question files
    python tool/general_expert.py --all

    # Limit questions and choose model
    python tool/general_expert.py --input data/questions_physicist.json \\
        --num-questions 5 --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

RESULTS_DIR = Path("tool/results")
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_DELAY = 0.3  # seconds between API calls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_user_prefix(role: str) -> str:
    # Convert slug like "software-architect" → "software architect"
    role_display = role.replace("-", " ")
    return f"Assume you are a {role_display} with 20 years of experience."


def load_questions(path: Path, num_questions: int) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data[:num_questions]


def load_completed(output_path: Path) -> set[str]:
    """Return set of already-answered question IDs so we can resume."""
    if not output_path.exists():
        return set()
    with open(output_path) as f:
        existing = json.load(f)
    return {row["question_id"] for row in existing}


def save_results(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

def make_client(api_key: str):
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def call_openai(client, question: str, user_prefix: str, model: str) -> str:
    user_content = f"{user_prefix} {question}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": user_content},
        ],
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_file(input_path: Path, api_key: str, model: str, num_questions: int, delay: float) -> None:
    questions = load_questions(input_path, num_questions)
    if not questions:
        print(f"  No questions found in {input_path}")
        return

    # Derive role from filename: questions_software-architect.json → software-architect
    stem = input_path.stem  # e.g. "questions_software-architect"
    role = stem.removeprefix("questions_")

    output_path = RESULTS_DIR / f"{role}.json"
    completed = load_completed(output_path)

    # Load existing rows so we can append
    existing_rows: list[dict] = []
    if output_path.exists():
        with open(output_path) as f:
            existing_rows = json.load(f)

    pending = [q for q in questions if q["id"] not in completed]
    if not pending:
        print(f"  [{role}] All {len(questions)} questions already answered. Skipping.")
        return

    print(f"  [{role}] {len(completed)} done, {len(pending)} remaining — model={model}")
    user_prefix = build_user_prefix(role)
    client = make_client(api_key)

    rows = list(existing_rows)
    for i, q in enumerate(pending, 1):
        try:
            answer = call_openai(client, q["question"], user_prefix, model)
        except Exception as e:
            answer = f"ERROR: {e}"

        rows.append({
            "question_id": q["id"],
            "role": role,
            "question": q["question"],
            "answer": answer,
        })
        save_results(output_path, rows)

        print(f"    [{i}/{len(pending)}] {q['id']}")
        if i < len(pending):
            time.sleep(delay)

    print(f"  [{role}] Done. Results → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query OpenAI with a generic role injection prompt for all questions in a data file."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input",
        type=Path,
        metavar="FILE",
        help="Path to a single questions JSON file (e.g. data/questions_economist.json)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all data/questions_*.json files",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    parser.add_argument("--num-questions", type=int, default=30, help="Max questions per role (default: 30)")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Seconds between API calls (default: 0.3)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    if args.all:
        data_dir = Path("data")
        files = sorted(data_dir.glob("questions_*.json"))
        if not files:
            print(f"ERROR: No questions_*.json files found in {data_dir}/")
            sys.exit(1)
        print(f"Found {len(files)} question file(s). Running all...\n")
        for f in files:
            run_file(f, api_key, args.model, args.num_questions, args.delay)
    else:
        if not args.input.exists():
            print(f"ERROR: File not found: {args.input}")
            sys.exit(1)
        run_file(args.input, api_key, args.model, args.num_questions, args.delay)

    print("\nAll done.")


if __name__ == "__main__":
    main()
