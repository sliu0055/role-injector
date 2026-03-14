#!/usr/bin/env python3
"""
query_role.py — Query ChromaDB for the best-matching expert role.

Usage:
    python query_role.py "What are the risks of aortic stenosis?" --top-k 3
    python query_role.py "How do I structure a Series A term sheet?" --top-k 1

Output: JSON with top matching roles, scores, and prompt templates.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def query_role(question: str, top_k: int = 3, db_path: str = None, usage_boost: float = 0.05):
    if db_path is None:
        db_path = os.environ.get(
            "ROLE_INJECTOR_DB_PATH",
            os.path.expanduser("~/.role-injector/roledb")
        )
    
    if not Path(db_path).exists():
        print(json.dumps({"error": f"DB not found at {db_path}. Run init_roledb.py first."}))
        sys.exit(1)
    
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print(json.dumps({"error": "chromadb not installed. Run: pip install chromadb sentence-transformers --break-system-packages"}))
        sys.exit(1)
    
    client = chromadb.PersistentClient(path=db_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    try:
        collection = client.get_collection("roles", embedding_function=ef)
    except Exception as e:
        print(json.dumps({"error": f"Collection not found: {e}"}))
        sys.exit(1)
    
    results = collection.query(
        query_texts=[question],
        n_results=min(top_k * 3, collection.count()),  # fetch extra for reranking
        include=["metadatas", "distances"]
    )
    
    if not results["ids"][0]:
        print(json.dumps({"matches": []}))
        return
    
    # Rerank: boost by usage_count
    matches = []
    for role_id, metadata, distance in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite. Convert to similarity.
        similarity = 1 - (distance / 2)
        
        # Apply usage boost (capped so popular roles don't dominate cold queries)
        usage = metadata.get("usage_count", 0)
        boosted_score = similarity + min(usage * usage_boost, 0.3)
        
        matches.append({
            "id": role_id,
            "name": metadata.get("name"),
            "domain": metadata.get("domain"),
            "subdomain": metadata.get("subdomain"),
            "usage_count": usage,
            "similarity": round(similarity, 4),
            "score": round(boosted_score, 4),
            "prompt_template": metadata.get("prompt_template"),
        })
    
    # Sort by boosted score, take top_k
    matches.sort(key=lambda x: x["score"], reverse=True)
    matches = matches[:top_k]
    
    print(json.dumps({"matches": matches}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query role DB for best matching expert role")
    parser.add_argument("question", help="The user's question")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to return")
    parser.add_argument("--db-path", default=None, help="ChromaDB path (default: ~/.role-injector/roledb)")
    parser.add_argument("--usage-boost", type=float, default=0.05, help="Score boost per usage count")
    args = parser.parse_args()
    
    query_role(args.question, args.top_k, args.db_path, args.usage_boost)
