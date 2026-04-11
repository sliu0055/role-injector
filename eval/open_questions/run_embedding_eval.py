#!/usr/bin/env python3
"""
run_embedding_eval.py — Evaluate embedding-selected role injection on open-ended questions.

Role selection: ChromaDB top-1 cosine similarity match (no LLM picker).
Fallback: if best match similarity < SIMILARITY_THRESHOLD, use a generic domain prompt.

Answering model : GPT-4o mini  (OpenAI)
Judge model     : Claude Haiku (Anthropic)
Condition       : embedding_selected

Usage:
    export OPENAI_API_KEY="your-key"
    export ANTHROPIC_API_KEY="your-key"
    python eval/open_questions/run_embedding_eval.py

    # Custom options
    python eval/open_questions/run_embedding_eval.py \\
        --questions data/questions_others.json \\
        --output eval/open_questions/embedding_results.csv \\
        --db-path ~/.role-injector/roledb \\
        --similarity-threshold 0.60
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

DEFAULT_QUESTIONS   = "data/questions_others.json"
DEFAULT_OUTPUT      = "eval/open_questions/embedding_results.csv"
DEFAULT_DB_PATH     = os.path.expanduser("~/.role-injector/roledb")
DEFAULT_ANSWERER    = "gpt-4o-mini"
DEFAULT_JUDGE       = "claude-haiku-4-5-20251001"
SIMILARITY_THRESHOLD = 0.60
CONDITION           = "embedding_selected"

GENERIC_PROMPT_TEMPLATE = "You are a knowledgeable expert in {domain}. Answer clearly and accurately based on domain expertise."

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
# ChromaDB role selector
# ---------------------------------------------------------------------------

def init_chromadb(db_path: str):
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print("ERROR: pip install chromadb sentence-transformers")
        sys.exit(1)

    if not Path(db_path).exists():
        print(f"ERROR: ChromaDB not found at {db_path}. Run: python scripts/init_roledb.py")
        sys.exit(1)

    client = chromadb.PersistentClient(path=db_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    try:
        collection = client.get_collection("roles", embedding_function=ef)
    except Exception as e:
        print(f"ERROR: Could not load 'roles' collection: {e}")
        sys.exit(1)

    print(f"ChromaDB ready — {collection.count()} roles indexed at {db_path}")
    return collection


def select_role(collection, question: str, domain: str, threshold: float) -> tuple[str, str, float, bool]:
    """
    Query ChromaDB for top-1 embedding match (no usage boost, no LLM picker).

    Returns:
        system_prompt  : the system prompt to inject
        selected_role  : role id used (or 'generic_fallback')
        similarity     : cosine similarity of top-1 match (0.0 if fallback)
        used_fallback  : True if similarity < threshold
    """
    results = collection.query(
        query_texts=[question],
        n_results=1,
        include=["metadatas", "distances"],
    )

    if not results["ids"][0]:
        # No results at all — fall back
        return GENERIC_PROMPT_TEMPLATE.format(domain=domain), "generic_fallback", 0.0, True

    distance   = results["distances"][0][0]
    metadata   = results["metadatas"][0][0]
    role_id    = results["ids"][0][0]

    # ChromaDB cosine space: distance = 1 - cosine_similarity (range [0, 2])
    # Convert to similarity in [0, 1]
    similarity = 1 - (distance / 2)

    if similarity >= threshold:
        prompt_template = metadata.get("prompt_template", "")
        return prompt_template, role_id, round(similarity, 4), False
    else:
        return GENERIC_PROMPT_TEMPLATE.format(domain=domain), "generic_fallback", round(similarity, 4), True


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


def call_openai(client, question: str, system_prompt: str, model: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]
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

FIELDNAMES = [
    "question_id", "domain", "role", "question_type", "condition",
    "selected_role", "role_similarity", "used_fallback",
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
                completed.add(row["question_id"])
        print(f"Resuming: {len(completed)} results already recorded.")
    return completed, rows


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(
    questions: list[dict],
    collection,
    similarity_threshold: float,
    openai_client,
    claude_client,
    answerer_model: str,
    judge_model: str,
    output_path: Path,
    delay: float,
):
    completed, existing_rows = load_completed(output_path)

    remaining = [q for q in questions if q["id"] not in completed]
    if not remaining:
        print("All questions already evaluated.")
        return

    print(f"Condition          : {CONDITION}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Answerer           : {answerer_model}")
    print(f"Judge              : {judge_model}")
    print(f"Remaining          : {len(remaining)} questions\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fallback_count = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)

        for i, q in enumerate(remaining, 1):
            # 1. Select role via embedding top-1
            system_prompt, selected_role, similarity, used_fallback = select_role(
                collection, q["question"], q["domain"], similarity_threshold
            )
            if used_fallback:
                fallback_count += 1

            # 2. Get answer from GPT-4o mini
            try:
                response = call_openai(openai_client, q["question"], system_prompt, answerer_model)
            except Exception as e:
                response = f"ANSWER_ERROR: {e}"

            # 3. Judge with Claude Haiku
            score, justification = judge_response(
                claude_client, judge_model,
                q["domain"], q["role"], q["question"], response,
            )

            writer.writerow({
                "question_id":         q["id"],
                "domain":              q["domain"],
                "role":                q["role"],
                "question_type":       q.get("type", ""),
                "condition":           CONDITION,
                "selected_role":       selected_role,
                "role_similarity":     similarity,
                "used_fallback":       used_fallback,
                "question":            q["question"],
                "response":            response,
                "judge_score":         score if score is not None else "PARSE_FAIL",
                "judge_justification": justification,
            })
            f.flush()

            fallback_tag = " [fallback]" if used_fallback else f" [{selected_role}]"
            score_str = str(score) if score is not None else "?"
            print(f"  [{i:3d}/{len(remaining)}] {q['id'][:38]:38s} | sim={similarity:.3f}{fallback_tag:25s} | score={score_str}/5")
            time.sleep(delay)

    print(f"\nFallbacks used: {fallback_count}/{len(remaining)} ({100*fallback_count/max(len(remaining),1):.1f}%)")
    print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate embedding-selected role injection on open-ended domain questions"
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
        "--db-path", default=DEFAULT_DB_PATH,
        help=f"ChromaDB path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--similarity-threshold", type=float, default=SIMILARITY_THRESHOLD,
        help=f"Min similarity to use role prompt; below this falls back to generic (default: {SIMILARITY_THRESHOLD})",
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

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    questions    = load_questions(args.questions)
    collection   = init_chromadb(args.db_path)
    openai_cli   = make_openai_client(openai_key)
    claude_cli   = make_claude_client(anthropic_key)

    run_eval(
        questions=questions,
        collection=collection,
        similarity_threshold=args.similarity_threshold,
        openai_client=openai_cli,
        claude_client=claude_cli,
        answerer_model=args.answerer_model,
        judge_model=args.judge_model,
        output_path=Path(args.output),
        delay=args.delay,
    )
    print("Done.")
