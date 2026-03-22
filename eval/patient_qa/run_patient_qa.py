#!/usr/bin/env python3
"""
run_patient_qa.py — Evaluate role injection on patient-style open-ended questions.

Three conditions per question:
  baseline        — no system prompt
  role_injected   — General Practitioner role from roles/medical/doctor.md
  generic_expert  — "You are a medical expert."

Usage:
    export OPENAI_API_KEY="your-key"
    python eval/patient_qa/run_patient_qa.py
    python eval/patient_qa/run_patient_qa.py --num-questions 10 --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Role prompts (from roles/medical/doctor.md → Prompt Template)
# ---------------------------------------------------------------------------

ROLE_PROMPT = (
    "You are a board-certified General Practitioner with 20 years of broad clinical "
    "experience. You have working knowledge across all medical specialties including "
    "cardiology, neurology, pulmonology, gastroenterology, endocrinology, nephrology, "
    "and infectious disease. When answering, reason through the clinical picture "
    "systematically, apply relevant evidence-based guidelines, flag any red flags or "
    "urgency signals, and note important caveats a non-specialist might miss. Always "
    "remind the user that your response is informational and does not replace "
    "in-person evaluation by a licensed clinician."
)

GENERIC_PROMPT = "You are a medical expert."

CONDITIONS = [
    ("baseline", None),
    ("role_injected", ROLE_PROMPT),
    ("generic_expert", GENERIC_PROMPT),
]

FIELDNAMES = ["question_id", "domain", "question", "condition", "answer"]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_questions(path: Path, num_questions: int) -> list[dict]:
    with open(path) as f:
        questions = json.load(f)
    return questions[:num_questions]


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_completed(output_path: Path) -> tuple[set, list[dict]]:
    completed = set()
    rows = []
    if output_path.exists():
        with open(output_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                completed.add((int(row["question_id"]), row["condition"]))
        print(f"Resuming: {len(completed)} results already recorded.")
    return completed, rows


# ---------------------------------------------------------------------------
# OpenAI client
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
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    questions: list[dict],
    api_key: str,
    model: str,
    delay: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed, existing_rows = load_completed(output_path)
    client = make_openai_client(api_key)

    total_remaining = len(questions) * len(CONDITIONS) - len(completed)
    if total_remaining == 0:
        print("All questions already answered. Run judge_patient_qa.py to evaluate.")
        return

    print(f"Model: {model}")
    print(f"Running {total_remaining} API calls...")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)

        done = 0
        for q in questions:
            for cond_name, system_prompt in CONDITIONS:
                if (q["id"], cond_name) in completed:
                    continue

                try:
                    answer = call_openai(client, q["question"], system_prompt, model)
                except Exception as e:
                    answer = f"ERROR: {e}"

                writer.writerow({
                    "question_id": q["id"],
                    "domain": q["domain"],
                    "question": q["question"],
                    "condition": cond_name,
                    "answer": answer,
                })
                f.flush()

                done += 1
                print(f"  [{done}/{total_remaining}] Q{q['id']:02d} | {cond_name:16s} | domain={q['domain']}")
                time.sleep(delay)

    print(f"\nResults saved to {output_path}")
    print("Next: python eval/patient_qa/judge_patient_qa.py")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate role injection on patient questions")
    parser.add_argument("--num-questions", type=int, default=30)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--output", default="eval/patient_qa/results.csv")
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument(
        "--questions-file",
        default="eval/patient_qa/questions.json",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    questions_path = Path(args.questions_file)
    if not questions_path.exists():
        print(f"ERROR: Questions file not found: {questions_path}")
        sys.exit(1)

    questions = load_questions(questions_path, args.num_questions)
    print(f"Loaded {len(questions)} questions from {questions_path}")

    run_eval(questions, api_key, args.model, args.delay, Path(args.output))
