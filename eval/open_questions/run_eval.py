#!/usr/bin/env python3
"""
run_eval.py — Evaluate role injection on open-ended domain questions.

Answering model : GPT-4o mini  (OpenAI)
Judge model     : Claude Haiku (Anthropic)
Default condition: baseline (no system prompt / no role injected)

Usage:
    export OPENAI_API_KEY="your-key"
    export ANTHROPIC_API_KEY="your-key"
    python eval/open_questions/run_eval.py

    # Custom options
    python eval/open_questions/run_eval.py \\
        --questions data/questions_others.json \\
        --output eval/open_questions/results.csv \\
        --condition baseline \\
        --answerer-model gpt-4o-mini \\
        --judge-model claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_QUESTIONS = "data/questions_others.json"
DEFAULT_OUTPUT    = "eval/open_questions/results.csv"
DEFAULT_ANSWERER  = "gpt-4o-mini"
DEFAULT_JUDGE     = "claude-haiku-4-5-20251001"
DEFAULT_CONDITION = "baseline"

JUDGE_PROMPT_TEMPLATE = """\
You are evaluating the quality of an AI response to a domain-specific expert question.

Domain: {domain}
Expected expertise: {role}
Question: {question}

Response to evaluate:
{response}

Rate this response on a scale of 1-5:
1 - Poor: vague, generic, or factually incorrect
2 - Below average: some relevant content but lacks depth or has notable errors
3 - Average: addresses the question but missing key expert insights
4 - Good: demonstrates solid domain knowledge with appropriate depth and accuracy
5 - Excellent: expert-level response that is precise, accurate, and practically useful

Reply with ONLY this format — a digit, a pipe, then one sentence (no other text):
DIGIT|ONE_SENTENCE_JUSTIFICATION

Example: 4|The response correctly explains the mechanism and covers the main trade-offs."""


# ---------------------------------------------------------------------------
# Questions loader
# ---------------------------------------------------------------------------

def load_questions(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: Questions file not found: {path}")
        sys.exit(1)
    with open(p) as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {path}")
    return questions


# ---------------------------------------------------------------------------
# Provider: OpenAI (answerer)
# ---------------------------------------------------------------------------

def make_openai_client(api_key: str):
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def call_openai(client, question: str, system_prompt: str | None, model: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Provider: Anthropic Claude (judge)
# ---------------------------------------------------------------------------

def make_claude_client(api_key: str):
    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic")
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


def call_claude(client, prompt: str, model: str) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def judge_response(
    claude_client,
    judge_model: str,
    domain: str,
    role: str,
    question: str,
    response: str,
) -> tuple[int | None, str]:
    """Ask Claude Haiku to score a response. Returns (score 1-5, justification)."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        domain=domain,
        role=role,
        question=question,
        response=response,
    )
    try:
        raw = call_claude(claude_client, prompt, judge_model)
        m = re.match(r"([1-5])\|(.+)", raw.strip())
        if m:
            return int(m.group(1)), m.group(2).strip()
        # Fallback: try to find a digit
        m2 = re.search(r"([1-5])", raw)
        if m2:
            return int(m2.group(1)), raw
        return None, raw
    except Exception as e:
        return None, f"JUDGE_ERROR: {e}"


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "question_id", "domain", "role", "question_type", "condition",
    "question", "response", "judge_score", "judge_justification",
]


def load_completed(output_path: Path) -> tuple[set, list[dict]]:
    completed = set()
    rows = []
    if output_path.exists():
        with open(output_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                completed.add((row["question_id"], row["condition"]))
        print(f"Resuming: {len(completed)} results already recorded.")
    return completed, rows


# ---------------------------------------------------------------------------
# System prompt for each condition
# ---------------------------------------------------------------------------

def resolve_system_prompt(condition: str) -> str | None:
    if condition == "baseline":
        return None
    # Future conditions (e.g. fixed_role, embedding_selected) can be added here
    return None


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(
    questions: list[dict],
    condition: str,
    openai_client,
    claude_client,
    answerer_model: str,
    judge_model: str,
    output_path: Path,
    delay: float,
):
    completed, existing_rows = load_completed(output_path)
    system_prompt = resolve_system_prompt(condition)

    remaining = [q for q in questions if (q["id"], condition) not in completed]
    if not remaining:
        print("All questions already evaluated.")
        return

    print(f"Condition : {condition} (system_prompt={'None' if system_prompt is None else repr(system_prompt[:60])})")
    print(f"Answerer  : {answerer_model}")
    print(f"Judge     : {judge_model}")
    print(f"Remaining : {len(remaining)} questions\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)

        for i, q in enumerate(remaining, 1):
            # 1. Get answer from GPT-4o mini
            try:
                response = call_openai(openai_client, q["question"], system_prompt, answerer_model)
            except Exception as e:
                response = f"ANSWER_ERROR: {e}"

            # 2. Judge with Claude Haiku
            score, justification = judge_response(
                claude_client, judge_model,
                q["domain"], q["role"], q["question"], response,
            )

            writer.writerow({
                "question_id":       q["id"],
                "domain":            q["domain"],
                "role":              q["role"],
                "question_type":     q.get("type", ""),
                "condition":         condition,
                "question":          q["question"],
                "response":          response,
                "judge_score":       score if score is not None else "PARSE_FAIL",
                "judge_justification": justification,
            })
            f.flush()

            score_str = str(score) if score is not None else "?"
            print(f"  [{i}/{len(remaining)}] {q['id'][:40]:40s} | score={score_str}/5 | {justification[:60]}")
            time.sleep(delay)

    print(f"\nResults saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate role injection on open-ended domain questions"
    )
    parser.add_argument(
        "--questions", default=DEFAULT_QUESTIONS,
        help=f"Path to questions JSON (default: {DEFAULT_QUESTIONS})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--condition", default=DEFAULT_CONDITION,
        help=f"Eval condition (default: {DEFAULT_CONDITION})",
    )
    parser.add_argument(
        "--answerer-model", default=DEFAULT_ANSWERER,
        help=f"Model to answer questions (default: {DEFAULT_ANSWERER})",
    )
    parser.add_argument(
        "--judge-model", default=DEFAULT_JUDGE,
        help=f"Model to judge responses (default: {DEFAULT_JUDGE})",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds between question pairs (default: 0.5)",
    )
    args = parser.parse_args()

    # Validate API keys
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    questions   = load_questions(args.questions)
    openai_cli  = make_openai_client(openai_key)
    claude_cli  = make_claude_client(anthropic_key)

    run_eval(
        questions=questions,
        condition=args.condition,
        openai_client=openai_cli,
        claude_client=claude_cli,
        answerer_model=args.answerer_model,
        judge_model=args.judge_model,
        output_path=Path(args.output),
        delay=args.delay,
    )
    print("Done.")
