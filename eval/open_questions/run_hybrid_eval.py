#!/usr/bin/env python3
"""
run_hybrid_eval.py — Evaluate hybrid role selection on open-ended questions.

Selector model  : Gemini Flash (picks best role from ChromaDB top-5)
Answering model : GPT-4o mini  (same as baseline and generic conditions)
Judge model     : Claude Haiku (same judge as weikai's baseline eval)

Usage:
    export OPENAI_API_KEY="your-key"
    export ANTHROPIC_API_KEY="your-key"
    export GEMINI_API_KEY="your-key"
    python eval/open_questions/run_hybrid_eval.py

    # Custom options
    python eval/open_questions/run_hybrid_eval.py \
        --questions data/questions_others.json \
        --output eval/open_questions/hybrid_results.csv \
        --num-questions 300
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_QUESTIONS = "data/questions_others.json"
DEFAULT_OUTPUT    = "eval/open_questions/hybrid_results.csv"
DEFAULT_ANSWERER  = "gpt-4o-mini"
DEFAULT_JUDGE     = "claude-haiku-4-5-20251001"
SELECTOR_MODEL    = "gemini-2.5-flash"
DEFAULT_DB_PATH   = os.path.expanduser("~/.role-injector/roledb")
SIMILARITY_THRESHOLD = 0.60  # Below this, fall back to generic domain prompt

# Same judge prompt as weikai's baseline eval for comparability
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

def load_questions(path: str, num_questions: int) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: Questions file not found: {path}")
        sys.exit(1)
    with open(p) as f:
        questions = json.load(f)
    questions = questions[:num_questions]
    print(f"Loaded {len(questions)} questions from {path}")
    return questions


# ---------------------------------------------------------------------------
# ChromaDB + Hybrid selector
# ---------------------------------------------------------------------------

def init_chromadb(db_path: str):
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print("ERROR: pip install chromadb sentence-transformers")
        sys.exit(1)

    if not Path(db_path).exists():
        print(f"ERROR: ChromaDB not found at {db_path}. Run scripts/init_roledb.py first.")
        sys.exit(1)

    client = chromadb.PersistentClient(path=db_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_collection("roles", embedding_function=ef)
    print(f"ChromaDB loaded: {collection.count()} roles indexed.")
    return collection


def get_embedding_candidates(collection, question: str, top_k: int = 5) -> list[dict]:
    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count()),
        include=["metadatas", "distances"],
    )
    candidates = []
    for role_id, metadata, distance in zip(
        results["ids"][0], results["metadatas"][0], results["distances"][0]
    ):
        similarity = 1 - (distance / 2)
        candidates.append({
            "id": role_id,
            "name": metadata.get("name", role_id),
            "domain": metadata.get("domain", ""),
            "subdomain": metadata.get("subdomain", ""),
            "similarity": similarity,
            "prompt_template": metadata.get("prompt_template", ""),
        })
    return candidates


def make_domain_fallback(domain: str) -> str:
    """Generate a generic expert prompt based on domain."""
    return f"You are a knowledgeable expert in {domain}. Answer with depth, accuracy, and practical insight."


def hybrid_select(collection, gemini_client, question: str, domain: str) -> tuple[str, str]:
    """Select best role via ChromaDB top-5 + Gemini Flash pick.
    Falls back to a generic domain prompt if best similarity is below threshold.
    Returns (prompt_template, role_id).
    """
    candidates = get_embedding_candidates(collection, question, top_k=5)

    # No candidates or all below threshold → domain fallback
    if not candidates or candidates[0]["similarity"] < SIMILARITY_THRESHOLD:
        return make_domain_fallback(domain), f"generic-{domain}"

    # Filter to only candidates above threshold
    candidates = [c for c in candidates if c["similarity"] >= SIMILARITY_THRESHOLD]
    if not candidates:
        return make_domain_fallback(domain), f"generic-{domain}"
    if len(candidates) == 1:
        return candidates[0]["prompt_template"], candidates[0]["id"]

    role_list = "\n".join(
        f"{i+1}. {c['name']} ({c['subdomain']})"
        for i, c in enumerate(candidates)
    )
    selector_prompt = (
        f"Given this question, which expert role is best suited to answer it?\n\n"
        f"Question: {question}\n\n"
        f"Available roles:\n{role_list}\n\n"
        f"Respond with ONLY the number (1-{len(candidates)}) of the best matching role."
    )

    try:
        raw = call_gemini(gemini_client, selector_prompt)
        m = re.search(r"(\d+)", raw.strip())
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]["prompt_template"], candidates[idx]["id"]
    except Exception as e:
        print(f"    [selector error: {e}, falling back to embedding top-1]")

    return candidates[0]["prompt_template"], candidates[0]["id"]


# ---------------------------------------------------------------------------
# Provider: Gemini (selector)
# ---------------------------------------------------------------------------

def make_gemini_client(api_key: str):
    try:
        from google import genai
    except ImportError:
        print("ERROR: pip install google-genai")
        sys.exit(1)
    return genai.Client(api_key=api_key)


def call_gemini(client, prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=SELECTOR_MODEL,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 2 ** (attempt + 1)
                print(f"    [Gemini rate limited, retrying in {wait}s...]")
                time.sleep(wait)
            else:
                raise
    raise Exception("Gemini rate limited after max retries")


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


def call_openai(client, question: str, system_prompt: str, model: str, max_retries: int = 3) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=800,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = 2 ** (attempt + 1)
                print(f"    [OpenAI rate limited, retrying in {wait}s...]")
                time.sleep(wait)
            else:
                raise
    raise Exception("OpenAI rate limited after max retries")


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
        m2 = re.search(r"([1-5])", raw)
        if m2:
            return int(m2.group(1)), raw
        return None, raw
    except Exception as e:
        return None, f"JUDGE_ERROR: {e}"


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

FIELDNAMES_ANSWERS = [
    "question_id", "domain", "role", "question_type", "condition",
    "selected_role", "question", "response",
]

FIELDNAMES_JUDGED = FIELDNAMES_ANSWERS + ["judge_score", "judge_justification"]


def load_completed(output_path: Path) -> tuple[set, list[dict]]:
    completed = set()
    rows = []
    if output_path.exists():
        with open(output_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                completed.add(row["question_id"])
        print(f"Resuming: {len(completed)} results already recorded.")
    return completed, rows


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_answers(
    questions: list[dict],
    collection,
    gemini_client,
    openai_client,
    answerer_model: str,
    output_path: Path,
    delay: float,
    workers: int = 10,
):
    """Phase 1: Select roles and collect answers (no judging). Runs in parallel."""
    completed, existing_rows = load_completed(output_path)

    remaining = [q for q in questions if q["id"] not in completed]
    if not remaining:
        print("All questions already answered.")
        return

    print(f"Condition : hybrid_selected (ChromaDB top-5 → Gemini Flash)")
    print(f"Answerer  : {answerer_model}")
    print(f"Selector  : {SELECTOR_MODEL}")
    print(f"Workers   : {workers}")
    print(f"Remaining : {len(remaining)} questions\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    counter_lock = threading.Lock()
    done_count = [0]
    total = len(remaining)

    def _process_question(q):
        time.sleep(delay)
        # 1. Hybrid select: ChromaDB top-5 → Gemini Flash pick
        system_prompt, selected_role = hybrid_select(
            collection, gemini_client, q["question"], q["domain"]
        )
        # 2. Get answer from GPT-4o mini with selected role as system prompt
        try:
            response = call_openai(
                openai_client, q["question"], system_prompt, answerer_model
            )
        except Exception as e:
            response = f"ANSWER_ERROR: {e}"

        row = {
            "question_id":   q["id"],
            "domain":        q["domain"],
            "role":          q["role"],
            "question_type": q.get("type", ""),
            "condition":     "hybrid_selected",
            "selected_role": selected_role,
            "question":      q["question"],
            "response":      response,
        }

        with counter_lock:
            done_count[0] += 1
            match = "✓" if selected_role == q["role"] else "✗"
            if done_count[0] % 50 == 0 or done_count[0] == total:
                print(f"  [{done_count[0]}/{total}] last: {q['id'][:40]:40s} | selected={selected_role:30s} {match}")

        return q["id"], row

    # Run in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_question, q): q for q in remaining}
        for future in as_completed(futures):
            qid, row = future.result()
            results[qid] = row

    # Write output in original question order
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES_ANSWERS)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)
        for q in remaining:
            if q["id"] in results:
                writer.writerow(results[q["id"]])

    print(f"\nAnswers saved to {output_path}")


def run_judge(
    answers_path: Path,
    claude_client,
    judge_model: str,
    output_path: Path,
    delay: float,
    workers: int = 10,
):
    """Phase 2: Score existing answers with Claude Haiku judge. Runs in parallel."""
    if not answers_path.exists():
        print(f"ERROR: Answers file not found: {answers_path}")
        sys.exit(1)

    with open(answers_path) as f:
        reader = csv.DictReader(f)
        answer_rows = list(reader)
    print(f"Loaded {len(answer_rows)} answers from {answers_path}")

    # Check what's already judged
    judged_map = {}
    if output_path.exists():
        with open(output_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                judged_map[row["question_id"]] = row
        print(f"Resuming: {len(judged_map)} already judged.")

    remaining = [r for r in answer_rows if r["question_id"] not in judged_map]
    if not remaining:
        print("All answers already judged.")
        return

    print(f"Judge     : {judge_model}")
    print(f"Workers   : {workers}")
    print(f"Remaining : {len(remaining)} answers to judge\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    counter_lock = threading.Lock()
    done_count = [len(judged_map)]
    total = len(answer_rows)

    def _judge_row(row):
        time.sleep(delay)
        score, justification = judge_response(
            claude_client, judge_model,
            row["domain"], row["role"], row["question"], row["response"],
        )
        judged_row = dict(row)
        judged_row["judge_score"] = score if score is not None else "PARSE_FAIL"
        judged_row["judge_justification"] = justification

        with counter_lock:
            done_count[0] += 1
            if done_count[0] % 50 == 0:
                print(f"  [{done_count[0]}/{total}] judged")

        return row["question_id"], judged_row

    # Run in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_judge_row, row): row for row in remaining}
        for future in as_completed(futures):
            qid, judged_row = future.result()
            results[qid] = judged_row

    # Write in original order
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES_JUDGED)
        writer.writeheader()
        for row in answer_rows:
            qid = row["question_id"]
            if qid in judged_map:
                writer.writerow(judged_map[qid])
            elif qid in results:
                writer.writerow(results[qid])

    print(f"  [{done_count[0]}/{total}] judged")
    print(f"\nJudged results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid role selection on open-ended domain questions"
    )
    subparsers = parser.add_subparsers(dest="command", help="Phase to run")

    # Phase 1: collect answers
    answer_parser = subparsers.add_parser("answer", help="Select roles and collect answers")
    answer_parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    answer_parser.add_argument("--output", default=DEFAULT_OUTPUT)
    answer_parser.add_argument("--num-questions", type=int, default=300)
    answer_parser.add_argument("--answerer-model", default=DEFAULT_ANSWERER)
    answer_parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    answer_parser.add_argument("--workers", type=int, default=10,
                               help="Number of parallel workers (default 10)")
    answer_parser.add_argument("--delay", type=float, default=0.1,
                               help="Delay per request in seconds (default 0.1)")

    # Phase 2: judge answers
    judge_parser = subparsers.add_parser("judge", help="Score answers with Claude Haiku")
    judge_parser.add_argument("--answers", default=DEFAULT_OUTPUT,
                              help="Path to answers CSV from phase 1")
    judge_parser.add_argument("--output", default="eval/open_questions/hybrid_results_judged.csv")
    judge_parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    judge_parser.add_argument("--workers", type=int, default=10,
                              help="Number of parallel workers (default 10)")
    judge_parser.add_argument("--delay", type=float, default=0.1,
                              help="Delay per request in seconds (default 0.1)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "answer":
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            print("ERROR: Set OPENAI_API_KEY environment variable.")
            sys.exit(1)
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            print("ERROR: Set GEMINI_API_KEY environment variable.")
            sys.exit(1)

        collection = init_chromadb(args.db_path)
        gemini_cli = make_gemini_client(gemini_key)
        openai_cli = make_openai_client(openai_key)
        questions = load_questions(args.questions, args.num_questions)

        run_answers(
            questions=questions,
            collection=collection,
            gemini_client=gemini_cli,
            openai_client=openai_cli,
            answerer_model=args.answerer_model,
            output_path=Path(args.output),
            delay=args.delay,
            workers=args.workers,
        )

    elif args.command == "judge":
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_key:
            print("ERROR: Set ANTHROPIC_API_KEY environment variable.")
            sys.exit(1)

        claude_cli = make_claude_client(anthropic_key)

        run_judge(
            answers_path=Path(args.answers),
            claude_client=claude_cli,
            judge_model=args.judge_model,
            output_path=Path(args.output),
            delay=args.delay,
            workers=args.workers,
        )

    print("Done.")
