#!/usr/bin/env python3
"""
Statistical analysis of judged results across experimental conditions.

Compares 4 conditions on 1140 open-ended questions:
  - baseline: no role prompt
  - embedding: embedding top-1 role selection
  - general_expert: one-line generic expert prompt
  - hybrid: embedding shortlist + LLM pick

Usage:
    python eval/open_questions/run_analysis.py
    python eval/open_questions/run_analysis.py --by-domain
    python eval/open_questions/run_analysis.py --by-metric
    python eval/open_questions/run_analysis.py --similarity
    python eval/open_questions/run_analysis.py --all
"""
from __future__ import annotations

import argparse
import csv
import random
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DIR = Path(__file__).parent
EVAL_DIR = DIR.parent

FILES = {
    "baseline": EVAL_DIR / "judged" / "baseline_1140q_judged.csv",
    "embedding": EVAL_DIR / "judged" / "embedding_1140q_judged.csv",
    "general_expert": EVAL_DIR / "judged" / "general_expert_1140q_judged.csv",
    "hybrid": EVAL_DIR / "judged" / "hybrid_1140q_judged.csv",
}

CONDITIONS = list(FILES.keys())

METRICS = [
    "accuracy",
    "expertise_depth",
    "relevance",
    "safety",
    "clarity",
    "time_sensitive_correctness",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> dict[str, dict[str, dict]]:
    """Load all judged CSVs. Returns {condition: {question_id: row}}."""
    data = {}
    for name, path in FILES.items():
        with open(path, newline="") as f:
            data[name] = {r["question_id"]: r for r in csv.DictReader(f)}
    return data


def get_common_ids(data: dict) -> list[str]:
    common = set(data[CONDITIONS[0]].keys())
    for c in CONDITIONS[1:]:
        common &= set(data[c].keys())
    return sorted(common)


def get_scores(data: dict, common: list[str], condition: str, metric: str) -> np.ndarray:
    """Extract numeric scores, returning NaN for non-numeric values."""
    scores = []
    for qid in common:
        v = data[condition][qid].get(metric, "")
        try:
            scores.append(float(v))
        except (ValueError, TypeError):
            scores.append(np.nan)
    return np.array(scores)


# ---------------------------------------------------------------------------
# Overall summary
# ---------------------------------------------------------------------------

def print_overall_summary(data: dict, common: list[str]):
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Questions: {len(common)}")
    print(f"Conditions: {', '.join(CONDITIONS)}\n")

    header = f"{'Condition':20s}" + "".join(f"{m[:10]:>12s}" for m in METRICS + ["avg_score"])
    print(header)
    print("-" * len(header))

    for c in CONDITIONS:
        vals = []
        for m in METRICS + ["avg_score"]:
            scores = get_scores(data, common, c, m)
            valid = scores[~np.isnan(scores)]
            vals.append(f"{np.mean(valid):.3f}" if len(valid) > 0 else "N/A")
        print(f"{c:20s}" + "".join(f"{v:>12s}" for v in vals))


# ---------------------------------------------------------------------------
# Friedman + Wilcoxon tests
# ---------------------------------------------------------------------------

def print_statistical_tests(data: dict, common: list[str]):
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS (on avg_score)")
    print("=" * 80)

    # Build paired arrays (drop questions with any NaN)
    arrays = {}
    for c in CONDITIONS:
        arrays[c] = get_scores(data, common, c, "avg_score")

    mask = np.ones(len(common), dtype=bool)
    for c in CONDITIONS:
        mask &= ~np.isnan(arrays[c])

    n_valid = mask.sum()
    print(f"Valid paired observations: {n_valid}/{len(common)}\n")

    filtered = {c: arrays[c][mask] for c in CONDITIONS}

    # Friedman test
    stat, p = stats.friedmanchisquare(*[filtered[c] for c in CONDITIONS])
    print(f"Friedman chi-square = {stat:.4f}, p = {p:.2e}")
    print(f"Result: {'SIGNIFICANT' if p < 0.05 else 'NOT significant'} (alpha=0.05)\n")

    # Pairwise Wilcoxon
    pairs = [(a, b) for i, a in enumerate(CONDITIONS) for b in CONDITIONS[i + 1:]]
    n_tests = len(pairs)
    bonferroni = 0.05 / n_tests

    print(f"Pairwise Wilcoxon Signed-Rank Tests (Bonferroni threshold: {bonferroni:.4f})")
    print(f"{'Pair':40s} {'W-stat':>10s} {'p-value':>12s} {'p-adj':>12s} {'Sig?':>6s} {'Mean diff':>10s}")
    print("-" * 92)

    for a, b in pairs:
        diff = filtered[a] - filtered[b]
        w, p_w = stats.wilcoxon(diff)
        p_adj = min(p_w * n_tests, 1.0)
        sig = "YES" if p_adj < 0.05 else "no"
        mean_diff = np.mean(diff)
        print(f"{a + ' vs ' + b:40s} {w:10.1f} {p_w:12.6f} {p_adj:12.6f} {sig:>6s} {mean_diff:+10.4f}")


# ---------------------------------------------------------------------------
# Per-metric statistical tests
# ---------------------------------------------------------------------------

def print_per_metric_tests(data: dict, common: list[str]):
    print("\n" + "=" * 80)
    print("PER-METRIC FRIEDMAN TESTS")
    print("=" * 80)

    for metric in METRICS:
        arrays = {c: get_scores(data, common, c, metric) for c in CONDITIONS}
        mask = np.ones(len(common), dtype=bool)
        for c in CONDITIONS:
            mask &= ~np.isnan(arrays[c])

        n_valid = mask.sum()
        if n_valid < 10:
            print(f"\n{metric}: skipped (only {n_valid} valid observations)")
            continue

        filtered = {c: arrays[c][mask] for c in CONDITIONS}
        stat, p = stats.friedmanchisquare(*[filtered[c] for c in CONDITIONS])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        means = "  ".join(f"{c[:8]}={np.mean(filtered[c]):.3f}" for c in CONDITIONS)
        print(f"\n{metric}: chi2={stat:.2f}, p={p:.2e} {sig}")
        print(f"  {means}")

        if p < 0.05:
            pairs = [(a, b) for i, a in enumerate(CONDITIONS) for b in CONDITIONS[i + 1:]]
            n_tests = len(pairs)
            for a, b in pairs:
                diff = filtered[a] - filtered[b]
                if np.all(diff == 0):
                    continue
                w, p_w = stats.wilcoxon(diff)
                p_adj = min(p_w * n_tests, 1.0)
                if p_adj < 0.05:
                    print(f"    {a} vs {b}: p_adj={p_adj:.4f}, diff={np.mean(diff):+.3f}")


# ---------------------------------------------------------------------------
# By-domain breakdown
# ---------------------------------------------------------------------------

def get_domain(qid: str) -> str:
    return qid.split("_")[0]


def print_by_domain(data: dict, common: list[str]):
    print("\n" + "=" * 80)
    print("BY-DOMAIN BREAKDOWN (avg_score)")
    print("=" * 80)

    domains = sorted(set(get_domain(qid) for qid in common))

    header = f"{'Domain':15s} {'N':>5s}" + "".join(f"{c[:12]:>14s}" for c in CONDITIONS) + "    Best"
    print(header)
    print("-" * len(header))

    for domain in domains:
        dom_ids = [qid for qid in common if get_domain(qid) == domain]
        means = {}
        for c in CONDITIONS:
            scores = [float(data[c][qid]["avg_score"]) for qid in dom_ids
                      if data[c][qid].get("avg_score", "") not in ("", "0.0")]
            means[c] = np.mean(scores) if scores else 0.0

        best = max(means, key=means.get)
        row = f"{domain:15s} {len(dom_ids):5d}"
        for c in CONDITIONS:
            marker = " *" if c == best else ""
            row += f"{means[c]:12.3f}{marker:2s}"
        print(row)


# ---------------------------------------------------------------------------
# Response similarity
# ---------------------------------------------------------------------------

def print_similarity(data: dict, common: list[str], sample_size: int = 200):
    print("\n" + "=" * 80)
    print(f"RESPONSE TEXT SIMILARITY (SequenceMatcher, sample={sample_size})")
    print("=" * 80)

    random.seed(42)
    sample = random.sample(common, min(sample_size, len(common)))

    # Load response text from source files (judged CSVs have response column)
    pairs = [(a, b) for i, a in enumerate(CONDITIONS) for b in CONDITIONS[i + 1:]]

    print(f"{'Pair':40s} {'Mean':>8s} {'Median':>8s} {'<0.3':>8s} {'0.3-0.7':>8s} {'>0.7':>8s}")
    print("-" * 80)

    for a, b in pairs:
        sims = []
        for qid in sample:
            r1 = data[a][qid].get("response", "")
            r2 = data[b][qid].get("response", "")
            if r1 and r2:
                sims.append(SequenceMatcher(None, r1, r2).ratio())

        if not sims:
            continue

        sims.sort()
        mean = np.mean(sims)
        median = np.median(sims)
        low = sum(1 for s in sims if s < 0.3) / len(sims) * 100
        mid = sum(1 for s in sims if 0.3 <= s <= 0.7) / len(sims) * 100
        high = sum(1 for s in sims if s > 0.7) / len(sims) * 100
        print(f"{a + ' vs ' + b:40s} {mean:8.3f} {median:8.3f} {low:7.1f}% {mid:7.1f}% {high:7.1f}%")

    print("\nInterpretation: <0.3 = very different, 0.3-0.7 = somewhat similar, >0.7 = highly similar")


# ---------------------------------------------------------------------------
# Win/loss/tie analysis
# ---------------------------------------------------------------------------

def print_win_loss(data: dict, common: list[str]):
    print("\n" + "=" * 80)
    print("WIN / LOSS / TIE ANALYSIS (avg_score, pairwise)")
    print("=" * 80)

    pairs = [(a, b) for i, a in enumerate(CONDITIONS) for b in CONDITIONS[i + 1:]]

    print(f"{'Pair':40s} {'A wins':>8s} {'B wins':>8s} {'Tie':>8s}")
    print("-" * 68)

    for a, b in pairs:
        a_wins = b_wins = ties = 0
        for qid in common:
            sa = float(data[a][qid].get("avg_score", 0))
            sb = float(data[b][qid].get("avg_score", 0))
            if sa > sb:
                a_wins += 1
            elif sb > sa:
                b_wins += 1
            else:
                ties += 1
        total = len(common)
        print(f"{a + ' vs ' + b:40s} {a_wins:5d} ({a_wins/total*100:4.1f}%) "
              f"{b_wins:5d} ({b_wins/total*100:4.1f}%) {ties:5d} ({ties/total*100:4.1f}%)")


# ---------------------------------------------------------------------------
# By-domain statistical tests
# ---------------------------------------------------------------------------

def print_domain_stats(data: dict, common: list[str]):
    print("\n" + "=" * 90)
    print("BY-DOMAIN STATISTICAL TESTS (avg_score)")
    print("=" * 90)

    domains = sorted(set(get_domain(qid) for qid in common))

    for domain in domains:
        dom_ids = [q for q in common if get_domain(q) == domain]
        arrays = {c: np.array([float(data[c][q]["avg_score"]) for q in dom_ids]) for c in CONDITIONS}

        stat, p = stats.friedmanchisquare(*[arrays[c] for c in CONDITIONS])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        means = {c: np.mean(arrays[c]) for c in CONDITIONS}
        best = max(means, key=means.get)

        print(f"\n{domain.upper()} (n={len(dom_ids)}): Friedman chi2={stat:.2f}, p={p:.4f} {sig}")
        print(f"  Means: " + "  ".join(f"{c[:8]}={means[c]:.3f}" for c in CONDITIONS))
        print(f"  Best: {best}")

        if p < 0.05:
            pairs = [(a, b) for i, a in enumerate(CONDITIONS) for b in CONDITIONS[i + 1:]]
            n_tests = len(pairs)
            for a, b in pairs:
                diff = arrays[a] - arrays[b]
                if np.all(diff == 0):
                    continue
                w, p_w = stats.wilcoxon(diff)
                p_adj = min(p_w * n_tests, 1.0)
                if p_adj < 0.05:
                    print(f"    {a} vs {b}: p_adj={p_adj:.4f}, diff={np.mean(diff):+.3f}")


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def print_effect_sizes(data: dict, common: list[str]):
    print("\n" + "=" * 90)
    print("EFFECT SIZES — avg_score (Cohen's d, paired)")
    print("=" * 90)

    pairs = [(a, b) for i, a in enumerate(CONDITIONS) for b in CONDITIONS[i + 1:]]
    print(f"{'Pair':40s} {'Cohen d':>10s} {'Magnitude':>12s}")
    print("-" * 65)

    for a, b in pairs:
        sa = np.array([float(data[a][q]["avg_score"]) for q in common])
        sb = np.array([float(data[b][q]["avg_score"]) for q in common])
        diff = sa - sb
        d = np.mean(diff) / np.std(diff)
        mag = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        print(f"{a + ' vs ' + b:40s} {d:+10.4f} {mag:>12s}")

    print("\nEFFECT SIZES — expertise_depth (Cohen's d, paired)")
    print("-" * 65)

    for a, b in pairs:
        vals_a, vals_b = [], []
        for q in common:
            va = data[a][q].get("expertise_depth", "")
            vb = data[b][q].get("expertise_depth", "")
            if va not in ("", "N/A", "ERROR", "PARSE_FAIL") and vb not in ("", "N/A", "ERROR", "PARSE_FAIL"):
                vals_a.append(float(va))
                vals_b.append(float(vb))
        sa, sb = np.array(vals_a), np.array(vals_b)
        diff = sa - sb
        d = np.mean(diff) / np.std(diff)
        mag = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        print(f"{a + ' vs ' + b:40s} {d:+10.4f} {mag:>12s}")


# ---------------------------------------------------------------------------
# Score distributions
# ---------------------------------------------------------------------------

def print_distributions(data: dict, common: list[str]):
    print("\n" + "=" * 90)
    print("SCORE DISTRIBUTION (avg_score)")
    print("=" * 90)

    for c in CONDITIONS:
        scores = np.array([float(data[c][q]["avg_score"]) for q in common])
        print(f"{c:20s}  mean={np.mean(scores):.3f}  std={np.std(scores):.3f}  "
              f"min={np.min(scores):.2f}  Q1={np.percentile(scores, 25):.2f}  "
              f"med={np.median(scores):.2f}  Q3={np.percentile(scores, 75):.2f}  max={np.max(scores):.2f}")


# ---------------------------------------------------------------------------
# Per-metric by-domain
# ---------------------------------------------------------------------------

def print_metric_by_domain(data: dict, common: list[str]):
    domains = sorted(set(get_domain(qid) for qid in common))

    for metric in ("expertise_depth", "clarity"):
        label = "EXPERTISE_DEPTH" if metric == "expertise_depth" else "CLARITY"
        print(f"\n{'=' * 100}")
        print(f"{label} BY DOMAIN")
        print(f"{'=' * 100}")
        print(f"{'Domain':15s} {'N':>5s}" + "".join(f"{c[:12]:>14s}" for c in CONDITIONS) + f"  {'Best':>14s}  {'vs BL':>8s}")
        print("-" * 95)

        for domain in domains:
            dom_ids = [q for q in common if get_domain(q) == domain]
            means = {}
            for c in CONDITIONS:
                scores = [float(data[c][q][metric]) for q in dom_ids
                          if data[c][q].get(metric, "") not in ("", "N/A", "ERROR", "PARSE_FAIL")]
                means[c] = np.mean(scores) if scores else 0.0
            best = max(means, key=means.get)
            diff = means[best] - means["baseline"]
            row = f"{domain:15s} {len(dom_ids):5d}"
            for c in CONDITIONS:
                row += f"{means[c]:14.3f}"
            print(f"{row}  {best:>14s}  {diff:+8.3f}")


# ---------------------------------------------------------------------------
# Question-level extremes
# ---------------------------------------------------------------------------

def print_extremes(data: dict, common: list[str], top_n: int = 10):
    print("\n" + "=" * 100)
    print(f"TOP {top_n} QUESTIONS WHERE HYBRID >> BASELINE")
    print("=" * 100)

    diffs = []
    for qid in common:
        bs = float(data["baseline"][qid]["avg_score"])
        hs = float(data["hybrid"][qid]["avg_score"])
        diffs.append((hs - bs, qid))
    diffs.sort(reverse=True)

    for diff, qid in diffs[:top_n]:
        q = data["baseline"][qid]["question"][:100]
        be = float(data["baseline"][qid].get("expertise_depth", 0))
        he = float(data["hybrid"][qid].get("expertise_depth", 0))
        print(f"  diff={diff:+.2f}  BL_exp={be:.0f} HY_exp={he:.0f}  {qid[:35]:35s} {q}")

    print(f"\n{'=' * 100}")
    print(f"TOP {top_n} QUESTIONS WHERE BASELINE >> HYBRID")
    print("=" * 100)

    for diff, qid in diffs[-top_n:]:
        q = data["baseline"][qid]["question"][:100]
        bc = float(data["baseline"][qid].get("clarity", 0))
        hc = float(data["hybrid"][qid].get("clarity", 0))
        print(f"  diff={diff:+.2f}  BL_clr={bc:.0f} HY_clr={hc:.0f}  {qid[:35]:35s} {q}")


# ---------------------------------------------------------------------------
# By question type
# ---------------------------------------------------------------------------

def print_by_question_type(data: dict, common: list[str]):
    print("\n" + "=" * 100)
    print("BY QUESTION TYPE (avg_score)")
    print("=" * 100)

    # Infer question type from the results CSVs
    qtypes = {}
    results_file = EVAL_DIR / "results" / "baseline_1140q.csv"
    if results_file.exists():
        with open(results_file) as f:
            for r in csv.DictReader(f):
                qtypes[r["question_id"]] = r.get("question_type", "")

    types = sorted(set(qtypes.values()) - {""})
    if not types:
        print("No question_type data available.")
        return

    print(f"{'Type':15s} {'N':>5s}" + "".join(f"{c[:12]:>14s}" for c in CONDITIONS) + "  Best")
    print("-" * 85)

    for qtype in types:
        type_ids = [q for q in common if qtypes.get(q, "") == qtype]
        if not type_ids:
            continue
        means = {c: np.mean([float(data[c][q]["avg_score"]) for q in type_ids]) for c in CONDITIONS}
        best = max(means, key=means.get)
        print(f"{qtype:15s} {len(type_ids):5d}" + "".join(f"{means[c]:14.3f}" for c in CONDITIONS) + f"  {best}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze judged evaluation results")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    parser.add_argument("--by-domain", action="store_true", help="Breakdown by domain")
    parser.add_argument("--by-metric", action="store_true", help="Per-metric Friedman tests")
    parser.add_argument("--similarity", action="store_true", help="Response text similarity")
    parser.add_argument("--win-loss", action="store_true", help="Win/loss/tie counts")
    parser.add_argument("--domain-stats", action="store_true", help="By-domain statistical tests")
    parser.add_argument("--effect-sizes", action="store_true", help="Cohen's d effect sizes")
    parser.add_argument("--extremes", action="store_true", help="Questions with largest score differences")
    parser.add_argument("--by-type", action="store_true", help="Breakdown by question type")
    args = parser.parse_args()

    run_all = args.all
    run_default = not any([args.by_domain, args.by_metric, args.similarity,
                           args.win_loss, args.domain_stats, args.effect_sizes,
                           args.extremes, args.by_type, args.all])

    data = load_data()
    common = get_common_ids(data)

    # Always print summary and main stats
    print_overall_summary(data, common)
    print_statistical_tests(data, common)

    if run_all or run_default or args.by_metric:
        print_per_metric_tests(data, common)

    if run_all or run_default or args.by_domain:
        print_by_domain(data, common)

    if run_all or run_default or args.win_loss:
        print_win_loss(data, common)

    if run_all or run_default or args.similarity:
        print_similarity(data, common)

    if run_all or args.domain_stats:
        print_domain_stats(data, common)

    if run_all or args.effect_sizes:
        print_effect_sizes(data, common)
        print_distributions(data, common)

    if run_all or args.extremes:
        print_extremes(data, common)

    if run_all or args.by_type:
        print_by_question_type(data, common)

    if run_all or args.by_metric:
        print_metric_by_domain(data, common)


if __name__ == "__main__":
    main()
