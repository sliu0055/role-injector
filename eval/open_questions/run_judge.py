#!/usr/bin/env python3
"""
Unified judge script: scores all conditions with the same Claude Haiku judge.

Usage:
    python run_judge.py                         # Judge all unjudged 1140q CSVs
    python run_judge.py --files baseline_1140q.csv hybrid_1140q.csv
    python run_judge.py --workers 20            # 20 parallel API calls
    python run_judge.py --delay 0.1             # Shorter delay per worker
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

JUDGE_MODEL = "claude-haiku-4-5-20251001"

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

# Default 1140q files to judge
DEFAULT_FILES = [
    "baseline_1140q.csv",
    "embedding_1140q.csv",
    "general_expert_1140q.csv",
    "hybrid_1140q.csv",
]

DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Claude API
# ---------------------------------------------------------------------------

def call_claude(client: anthropic.Anthropic, prompt: str, model: str) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def judge_one(client: anthropic.Anthropic, row: dict) -> tuple[int | None, str]:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        domain=row.get("domain", ""),
        role=row.get("role", ""),
        question=row.get("question", ""),
        response=row.get("response", ""),
    )
    for attempt in range(3):
        try:
            raw = call_claude(client, prompt, JUDGE_MODEL)
            m = re.match(r"(\d)\|(.+)", raw)
            if m:
                return int(m.group(1)), m.group(2).strip()
            return None, f"PARSE_FAIL: {raw[:120]}"
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return None, f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def judge_file(client: anthropic.Anthropic, csv_path: Path, delay: float, workers: int):
    """Judge a single CSV file. Writes results to <name>_judged.csv, resumable."""

    out_path = csv_path.with_name(csv_path.stem + "_judged.csv")

    # Load source rows
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        src_fields = reader.fieldnames
        src_rows = list(reader)

    # Determine output fieldnames — ensure judge columns present
    out_fields = list(src_fields)
    for col in ("judge_score", "judge_justification"):
        if col not in out_fields:
            out_fields.append(col)

    # Load already-judged rows for resume
    judged = {}
    if out_path.exists():
        with open(out_path, newline="") as f:
            for row in csv.DictReader(f):
                qid = row.get("question_id", "")
                if qid and row.get("judge_score", ""):
                    judged[qid] = row

    remaining = [r for r in src_rows if r["question_id"] not in judged]
    total = len(src_rows)
    done = len(judged)

    print(f"\n{'='*60}")
    print(f"File     : {csv_path.name}")
    print(f"Total    : {total}")
    print(f"Already  : {done}")
    print(f"Remaining: {len(remaining)}")
    print(f"Workers  : {workers}")
    print(f"Output   : {out_path.name}")
    print(f"{'='*60}")

    if not remaining:
        print("Nothing to judge — skipping.")
        return

    # Judge remaining rows in parallel
    results = {}  # question_id -> judged row
    counter_lock = threading.Lock()
    done_count = [done]  # mutable counter for threads

    def _judge_row(row):
        time.sleep(delay)  # per-request throttle
        score, justification = judge_one(client, row)
        out_row = dict(row)
        out_row["judge_score"] = score if score is not None else "PARSE_FAIL"
        out_row["judge_justification"] = justification
        with counter_lock:
            done_count[0] += 1
            if done_count[0] % 50 == 0:
                print(f"  [{csv_path.stem}] {done_count[0]}/{total} judged")
        return row["question_id"], out_row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_judge_row, row): row for row in remaining}
        for future in as_completed(futures):
            qid, out_row = future.result()
            results[qid] = out_row

    # Write output in original order
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for row in src_rows:
            qid = row["question_id"]
            if qid in judged:
                writer.writerow(judged[qid])
            elif qid in results:
                writer.writerow(results[qid])

    print(f"  [{csv_path.stem}] {done_count[0]}/{total} judged")
    print(f"Done: {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Unified judge for all conditions")
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES,
                        help="CSV files to judge (relative to eval/open_questions/)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of parallel API calls (default 10)")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Delay per request in seconds (default 0.1)")
    args = parser.parse_args()

    client = anthropic.Anthropic()

    for fname in args.files:
        path = DIR / fname
        if not path.exists():
            print(f"SKIP: {fname} not found")
            continue
        judge_file(client, path, args.delay, args.workers)

    print("\nAll done.")


if __name__ == "__main__":
    main()
