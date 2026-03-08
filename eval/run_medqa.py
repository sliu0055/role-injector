#!/usr/bin/env python3
"""
run_eval.py — Evaluate role injection effectiveness on MedQA using Google Gemini.

Usage:
    export GEMINI_API_KEY="your-key"
    python eval/run_eval.py --num-questions 200 --output eval/results.csv
"""

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Role prompts (from roles/medical/cardiologist.md → Prompt Template)
# ---------------------------------------------------------------------------

ROLE_PROMPT = (
    "You are a board-certified Cardiologist with 15 years of clinical experience. "
    "You have deep expertise in heart disease, cardiac imaging, arrhythmias, and "
    "cardiovascular risk management. When answering, systematically consider the "
    "clinical picture, reference ACC/AHA guidelines where applicable, and clearly "
    "flag any red flags or urgency signals. Note important caveats a non-specialist "
    "might miss."
)

GENERIC_PROMPT = "You are a medical expert."

ANSWER_INSTRUCTION = (
    "\n\nRespond with ONLY the letter of the correct answer (A, B, C, or D). "
    "Do not include any explanation."
)

CONDITIONS = [
    ("baseline", None),
    ("role_injected", ROLE_PROMPT),
    ("generic_expert", GENERIC_PROMPT),
]


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str | None:
    """Pull a single letter answer (A-D) from model response."""
    text = text.strip()
    if re.fullmatch(r"[A-D]", text):
        return text
    m = re.match(r"^([A-D])[.\):\s]", text)
    if m:
        return m.group(1)
    m = re.search(r"(?:answer|correct)\s*(?:is|:)\s*([A-D])\b", text, re.IGNORECASE)
    if m:
        return m.group(1)
    for line in reversed(text.splitlines()):
        line = line.strip()
        if re.fullmatch(r"[A-D]", line):
            return line
    return None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_medqa(num_questions: int, split: str = "test") -> list[dict]:
    """Load MedQA English 4-option questions from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    print(f"Loading MedQA (med_qa_en_4options_source, split={split})...")
    ds = load_dataset(
        "bigbio/med_qa",
        "med_qa_en_4options_source",
        split=split,
        trust_remote_code=True,
    )

    questions = []
    for i, row in enumerate(ds):
        if i >= num_questions:
            break

        # options is list of {"key": "A", "value": "Ampicillin"}
        options_list = row["options"]
        options_text = "\n".join(
            f"{opt['key']}) {opt['value']}" for opt in options_list
        )

        questions.append({
            "id": i,
            "question": row["question"],
            "options": options_text,
            "correct_answer": row["answer_idx"],
        })

    print(f"Loaded {len(questions)} questions.")
    return questions


# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------

def make_client(api_key: str):
    """Create a Gemini client."""
    try:
        from google import genai
    except ImportError:
        print("ERROR: pip install google-genai")
        sys.exit(1)

    return genai.Client(api_key=api_key)


def call_gemini(
    client,
    prompt: str,
    system_prompt: str | None,
    model_name: str,
) -> str:
    """Call Gemini and return the text response."""
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
    ) if system_prompt else None

    response = client.models.generate_content(
        model=model_name,
        config=config,
        contents=prompt,
    )
    return response.text


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "question_id", "condition", "correct_answer",
    "model_answer", "raw_response", "is_correct",
]


def load_completed(output_path: Path) -> tuple[set, list[dict]]:
    """Load already-completed question/condition pairs from existing CSV."""
    completed = set()
    rows = []
    if output_path.exists():
        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                completed.add((int(row["question_id"]), row["condition"]))
        print(f"Resuming: {len(completed)} results already recorded.")
    return completed, rows


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    questions: list[dict],
    api_key: str,
    model_name: str,
    delay: float,
    output_path: str,
):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    completed, existing_rows = load_completed(output_file)
    client = make_client(api_key)

    total_remaining = len(questions) * len(CONDITIONS) - len(completed)
    if total_remaining == 0:
        print("All questions already evaluated. Use analyze.py to view results.")
        return

    print(f"Running {total_remaining} API calls (~{total_remaining * delay / 60:.0f} min)...")
    done = 0

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)

        for q in questions:
            prompt = f"{q['question']}\n\n{q['options']}{ANSWER_INSTRUCTION}"

            for cond_name, system_prompt in CONDITIONS:
                if (q["id"], cond_name) in completed:
                    continue

                try:
                    raw = call_gemini(client, prompt, system_prompt, model_name)
                except Exception as e:
                    raw = f"ERROR: {e}"

                model_answer = extract_answer(raw)
                is_correct = (
                    model_answer is not None
                    and model_answer.upper() == q["correct_answer"].upper()
                )

                writer.writerow({
                    "question_id": q["id"],
                    "condition": cond_name,
                    "correct_answer": q["correct_answer"],
                    "model_answer": model_answer or "PARSE_FAIL",
                    "raw_response": raw[:500],
                    "is_correct": is_correct,
                })
                f.flush()

                done += 1
                status = "correct" if is_correct else "wrong"
                print(
                    f"  [{done}/{total_remaining}] Q{q['id']} | {cond_name:16s} | "
                    f"expected={q['correct_answer']} got={model_answer} ({status})"
                )

                time.sleep(delay)

    print(f"\nResults saved to {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate role injection on MedQA")
    parser.add_argument(
        "--num-questions", type=int, default=200,
        help="Number of questions to evaluate (default: 200)",
    )
    parser.add_argument(
        "--model", default="gemini-2.5-flash",
        help="Gemini model name (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--output", default="eval/results.csv",
        help="Output CSV path (default: eval/results.csv)",
    )
    parser.add_argument(
        "--delay", type=float, default=4.5,
        help="Seconds between API calls (default: 4.5)",
    )
    parser.add_argument(
        "--split", default="test",
        help="Dataset split (default: test)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable.")
        print("  export GEMINI_API_KEY='your-key'")
        sys.exit(1)

    questions = load_medqa(args.num_questions, args.split)
    run_eval(questions, api_key, args.model, args.delay, args.output)
    print("\nDone. Run: python eval/analyze_medqa.py eval/results.csv")
