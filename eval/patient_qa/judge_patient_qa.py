#!/usr/bin/env python3
"""
judge_patient_qa.py — LLM-as-judge evaluation of three role conditions.

For each question, the judge scores three answers (baseline, role_injected,
generic_expert) on four criteria, then picks an overall winner.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python eval/patient_qa/judge_patient_qa.py
    python eval/patient_qa/judge_patient_qa.py --results eval/patient_qa/results.csv --model claude-3-haiku-20240307

Output:
    eval/patient_qa/judge_results.csv   — per-question scores
    eval/patient_qa/judge_summary.txt   — aggregate statistics printed + saved
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
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are an impartial medical education expert evaluating AI-generated responses "
    "to patient questions. You assess whether each response would genuinely help a "
    "patient understand their situation, know when to seek care, and make informed "
    "decisions. You have no preference for any AI system."
)

JUDGE_TEMPLATE = """\
A patient asked the following question to an AI assistant:

<question>
{question}
</question>

Three different AI responses were generated (labeled A, B, C in random order — the \
labels do NOT correspond to the condition names):

<response_A>
{answer_A}
</response_A>

<response_B>
{answer_B}
</response_B>

<response_C>
{answer_C}
</response_C>

Score each response on the following four criteria using a scale of 1–5:
  1 = very poor, 2 = poor, 3 = acceptable, 4 = good, 5 = excellent

Criteria:
- medical_accuracy: Is the information factually correct and consistent with current evidence?
- patient_helpfulness: Does it actually help the patient understand their situation and next steps?
- safety_guidance: Does it appropriately flag red flags, urgency, and when to seek in-person care?
- clarity: Is it written in plain language that a non-medical patient can understand?

Then pick an overall_winner (A, B, or C) — the single response you would most want a \
real patient to receive.

Return ONLY valid JSON in exactly this format (no extra text before or after):
{{
  "A": {{
    "medical_accuracy": <1-5>,
    "patient_helpfulness": <1-5>,
    "safety_guidance": <1-5>,
    "clarity": <1-5>
  }},
  "B": {{
    "medical_accuracy": <1-5>,
    "patient_helpfulness": <1-5>,
    "safety_guidance": <1-5>,
    "clarity": <1-5>
  }},
  "C": {{
    "medical_accuracy": <1-5>,
    "patient_helpfulness": <1-5>,
    "safety_guidance": <1-5>,
    "clarity": <1-5>
  }},
  "overall_winner": "<A|B|C>",
  "reasoning": "<one or two sentences explaining the winner choice>"
}}
"""

CRITERIA = ["medical_accuracy", "patient_helpfulness", "safety_guidance", "clarity"]

JUDGE_FIELDNAMES = [
    "question_id", "domain",
    # Per-condition scores
    "baseline_medical_accuracy", "baseline_patient_helpfulness",
    "baseline_safety_guidance", "baseline_clarity", "baseline_avg",
    "role_injected_medical_accuracy", "role_injected_patient_helpfulness",
    "role_injected_safety_guidance", "role_injected_clarity", "role_injected_avg",
    "generic_expert_medical_accuracy", "generic_expert_patient_helpfulness",
    "generic_expert_safety_guidance", "generic_expert_clarity", "generic_expert_avg",
    # Winner
    "overall_winner",
    "reasoning",
]


# ---------------------------------------------------------------------------
# Load results CSV
# ---------------------------------------------------------------------------

def load_results(results_path: Path) -> dict[int, dict]:
    """Returns {question_id: {condition: {question, domain, answer}}}"""
    data: dict[int, dict] = {}
    with open(results_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row["question_id"])
            if qid not in data:
                data[qid] = {"domain": row["domain"], "question": row["question"]}
            data[qid][row["condition"]] = row["answer"]
    return data


# ---------------------------------------------------------------------------
# Randomise label assignment so judge can't guess by order
# ---------------------------------------------------------------------------

LABEL_ORDERS = [
    ["baseline", "role_injected", "generic_expert"],
    ["baseline", "generic_expert", "role_injected"],
    ["role_injected", "baseline", "generic_expert"],
    ["role_injected", "generic_expert", "baseline"],
    ["generic_expert", "baseline", "role_injected"],
    ["generic_expert", "role_injected", "baseline"],
]


def assign_labels(question_id: int) -> dict[str, str]:
    """Deterministically pick a label order per question. Returns {label: condition}."""
    order = LABEL_ORDERS[question_id % len(LABEL_ORDERS)]
    return {"A": order[0], "B": order[1], "C": order[2]}


# ---------------------------------------------------------------------------
# Claude judge
# ---------------------------------------------------------------------------

def make_claude_client(api_key: str):
    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic")
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


def call_judge(client, question: str, answers: dict[str, str], model: str) -> str:
    prompt = JUDGE_TEMPLATE.format(
        question=question,
        answer_A=answers["A"],
        answer_B=answers["B"],
        answer_C=answers["C"],
    )
    response = client.messages.create(
        model=model,
        max_tokens=800,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def parse_judge_response(raw: str) -> dict | None:
    """Extract JSON from judge response."""
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try to find JSON block
    m = re.search(r"\{[\s\S]+\}", raw)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_judged(judge_path: Path) -> set[int]:
    judged = set()
    if judge_path.exists():
        with open(judge_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                judged.add(int(row["question_id"]))
        print(f"Resuming: {len(judged)} questions already judged.")
    return judged


# ---------------------------------------------------------------------------
# Aggregate + print summary
# ---------------------------------------------------------------------------

def print_summary(judge_path: Path, summary_path: Path) -> None:
    conditions = ["baseline", "role_injected", "generic_expert"]
    scores: dict[str, dict[str, list]] = {c: {cr: [] for cr in CRITERIA} for c in conditions}
    avgs: dict[str, list] = {c: [] for c in conditions}
    wins: dict[str, int] = {c: 0 for c in conditions}
    total = 0

    with open(judge_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            wins[row["overall_winner"]] += 1
            for cond in conditions:
                avg = float(row[f"{cond}_avg"])
                avgs[cond].append(avg)
                for cr in CRITERIA:
                    scores[cond][cr].append(float(row[f"{cond}_{cr}"]))

    lines = []
    lines.append(f"\n{'='*60}")
    lines.append("JUDGE EVALUATION SUMMARY")
    lines.append(f"{'='*60}")
    lines.append(f"Questions evaluated: {total}\n")

    lines.append(f"{'Condition':<20} {'Avg Score':>10}  " + "  ".join(f"{c[:8]:>8}" for c in CRITERIA))
    lines.append("-" * 70)
    for cond in conditions:
        avg_score = sum(avgs[cond]) / len(avgs[cond]) if avgs[cond] else 0
        criterion_avgs = [
            sum(scores[cond][cr]) / len(scores[cond][cr]) if scores[cond][cr] else 0
            for cr in CRITERIA
        ]
        crit_str = "  ".join(f"{v:>8.2f}" for v in criterion_avgs)
        lines.append(f"{cond:<20} {avg_score:>10.2f}  {crit_str}")

    lines.append(f"\n{'Overall Winner Frequency':}")
    lines.append("-" * 40)
    for cond in conditions:
        pct = wins[cond] / total * 100 if total else 0
        lines.append(f"  {cond:<20} {wins[cond]:>3} wins ({pct:.0f}%)")

    lines.append(f"\nJudge results saved to: {judge_path}")
    lines.append(f"Summary saved to:       {summary_path}")

    output = "\n".join(lines)
    print(output)
    summary_path.write_text(output)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_judge(
    results_path: Path,
    judge_path: Path,
    api_key: str,
    model: str,
    delay: float,
) -> None:
    judge_path.parent.mkdir(parents=True, exist_ok=True)
    data = load_results(results_path)
    judged = load_judged(judge_path)
    client = make_claude_client(api_key)

    question_ids = sorted(data.keys())
    remaining = [qid for qid in question_ids if qid not in judged]
    print(f"Questions to judge: {len(remaining)}")

    # Existing rows for resume
    existing_rows: list[dict] = []
    if judge_path.exists():
        with open(judge_path) as f:
            existing_rows = list(csv.DictReader(f))

    with open(judge_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JUDGE_FIELDNAMES)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)

        for i, qid in enumerate(remaining):
            entry = data[qid]
            label_map = assign_labels(qid)  # {label: condition}
            cond_map = {v: k for k, v in label_map.items()}  # {condition: label}

            answers = {
                label: entry.get(cond, "ERROR: answer not found")
                for label, cond in label_map.items()
            }

            try:
                raw = call_judge(client, entry["question"], answers, model)
                parsed = parse_judge_response(raw)
            except Exception as e:
                print(f"  ERROR judging Q{qid}: {e}")
                time.sleep(delay)
                continue

            if not parsed:
                print(f"  WARNING: Could not parse judge response for Q{qid}. Raw:\n{raw[:300]}")
                time.sleep(delay)
                continue

            # Translate winner label back to condition name
            winner_label = parsed.get("overall_winner", "")
            winner_cond = label_map.get(winner_label, winner_label)

            row_out: dict = {
                "question_id": qid,
                "domain": entry["domain"],
                "overall_winner": winner_cond,
                "reasoning": parsed.get("reasoning", ""),
            }
            for label, cond in label_map.items():
                label_scores = parsed.get(label, {})
                criterion_values = [
                    label_scores.get(cr, 0) for cr in CRITERIA
                ]
                avg = sum(criterion_values) / len(criterion_values) if criterion_values else 0
                for cr, val in zip(CRITERIA, criterion_values):
                    row_out[f"{cond}_{cr}"] = val
                row_out[f"{cond}_avg"] = round(avg, 2)

            writer.writerow(row_out)
            f.flush()

            print(
                f"  [{i+1}/{len(remaining)}] Q{qid:02d} ({entry['domain']:20s}) "
                f"winner={winner_cond}"
            )
            time.sleep(delay)

    summary_path = judge_path.parent / "judge_summary.txt"
    print_summary(judge_path, summary_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-judge for patient_qa results")
    parser.add_argument("--results", default="eval/patient_qa/results.csv")
    parser.add_argument("--output", default="eval/patient_qa/judge_results.csv")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        print("Run run_patient_qa.py first.")
        sys.exit(1)

    run_judge(results_path, Path(args.output), api_key, args.model, args.delay)
