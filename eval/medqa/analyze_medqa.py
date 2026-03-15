#!/usr/bin/env python3
"""
analyze.py — Analyze role injection evaluation results.

Usage:
    python eval/analyze.py eval/results.csv
"""

import csv
import sys
from collections import defaultdict


def load_results(path: str) -> dict[str, list[dict]]:
    """Load CSV and group rows by condition."""
    by_condition = defaultdict(list)
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            by_condition[row["condition"]].append(row)
    return dict(by_condition)


def accuracy(rows: list[dict]) -> tuple[int, int, float]:
    """Return (correct, total, accuracy%)."""
    correct = sum(1 for r in rows if r["is_correct"] == "True")
    total = len(rows)
    return correct, total, (correct / total * 100) if total else 0.0


def parse_failures(rows: list[dict]) -> int:
    """Count rows where answer extraction failed."""
    return sum(1 for r in rows if r["model_answer"] == "PARSE_FAIL")


def paired_comparison(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Compare two conditions on the same questions (paired by question_id)."""
    by_id_a = {r["question_id"]: r["is_correct"] == "True" for r in rows_a}
    by_id_b = {r["question_id"]: r["is_correct"] == "True" for r in rows_b}

    common_ids = sorted(set(by_id_a) & set(by_id_b), key=int)
    n = len(common_ids)

    # McNemar contingency counts
    both_correct = 0
    a_only = 0  # A correct, B wrong
    b_only = 0  # B correct, A wrong
    both_wrong = 0

    for qid in common_ids:
        a_ok = by_id_a[qid]
        b_ok = by_id_b[qid]
        if a_ok and b_ok:
            both_correct += 1
        elif a_ok and not b_ok:
            a_only += 1
        elif not a_ok and b_ok:
            b_only += 1
        else:
            both_wrong += 1

    return {
        "n": n,
        "both_correct": both_correct,
        "a_only": a_only,
        "b_only": b_only,
        "both_wrong": both_wrong,
    }


def print_summary(results: dict[str, list[dict]]):
    """Print accuracy table and pairwise comparisons."""
    condition_order = ["baseline", "role_injected", "generic_expert"]
    present = [c for c in condition_order if c in results]

    print()
    print("=" * 60)
    print("  ROLE INJECTION EVALUATION RESULTS")
    print("=" * 60)

    # Accuracy table
    print()
    print(f"  {'Condition':<20s} {'Correct':>8s} {'Total':>6s} {'Accuracy':>9s} {'Parse Fail':>11s}")
    print(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*9} {'-'*11}")

    accuracies = {}
    for cond in present:
        correct, total, acc = accuracy(results[cond])
        fails = parse_failures(results[cond])
        accuracies[cond] = acc
        print(f"  {cond:<20s} {correct:>8d} {total:>6d} {acc:>8.1f}% {fails:>11d}")

    # Deltas
    print()
    print("  COMPARISONS")
    print(f"  {'-'*56}")

    if "baseline" in accuracies and "role_injected" in accuracies:
        delta = accuracies["role_injected"] - accuracies["baseline"]
        print(f"  role_injected vs baseline:      {delta:+.1f}%")

    if "baseline" in accuracies and "generic_expert" in accuracies:
        delta = accuracies["generic_expert"] - accuracies["baseline"]
        print(f"  generic_expert vs baseline:     {delta:+.1f}%")

    if "generic_expert" in accuracies and "role_injected" in accuracies:
        delta = accuracies["role_injected"] - accuracies["generic_expert"]
        print(f"  role_injected vs generic_expert: {delta:+.1f}%")

    # Paired analysis (McNemar table)
    print()
    print("  PAIRED ANALYSIS (McNemar contingency)")
    print(f"  {'-'*56}")

    pairs = [
        ("baseline", "role_injected"),
        ("baseline", "generic_expert"),
        ("generic_expert", "role_injected"),
    ]

    for cond_a, cond_b in pairs:
        if cond_a not in results or cond_b not in results:
            continue
        comp = paired_comparison(results[cond_a], results[cond_b])
        print(f"\n  {cond_a} vs {cond_b} (n={comp['n']}):")
        print(f"    Both correct:    {comp['both_correct']}")
        print(f"    Only {cond_a}: {comp['a_only']}")
        print(f"    Only {cond_b}: {comp['b_only']}")
        print(f"    Both wrong:      {comp['both_wrong']}")

        # Discordant pairs are what matter for McNemar
        disc = comp["a_only"] + comp["b_only"]
        if disc > 0:
            # McNemar chi-squared (without continuity correction)
            chi2 = (comp["a_only"] - comp["b_only"]) ** 2 / disc
            print(f"    McNemar chi2:    {chi2:.2f} (discordant={disc})")
            if chi2 > 3.84:
                print(f"    => Statistically significant (p < 0.05)")
            else:
                print(f"    => Not significant (chi2 < 3.84)")

    print()
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval/analyze.py <results.csv>")
        sys.exit(1)

    results = load_results(sys.argv[1])
    if not results:
        print(f"No data found in {sys.argv[1]}")
        sys.exit(1)

    print_summary(results)
