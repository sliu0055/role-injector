#!/usr/bin/env python3
"""
update_usage.py — Increment usage counter for a role in ChromaDB.

Usage:
    python update_usage.py cardiologist
    python update_usage.py patent-attorney --db-path /custom/path/roledb

Higher usage_count → stronger score boost in future retrievals.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def update_usage(role_id: str, db_path: str = None):
    if db_path is None:
        db_path = os.environ.get(
            "ROLE_INJECTOR_DB_PATH",
            os.path.expanduser("~/.role-injector/roledb")
        )
    
    if not Path(db_path).exists():
        print(f"ERROR: DB not found at {db_path}. Run init_roledb.py first.")
        sys.exit(1)
    
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print("ERROR: chromadb not installed.")
        sys.exit(1)
    
    client = chromadb.PersistentClient(path=db_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    try:
        collection = client.get_collection("roles", embedding_function=ef)
    except Exception as e:
        print(f"ERROR: Collection not found: {e}")
        sys.exit(1)
    
    # Fetch current metadata
    result = collection.get(ids=[role_id], include=["metadatas", "documents"])
    
    if not result["ids"]:
        print(f"ERROR: Role '{role_id}' not found in DB.")
        sys.exit(1)
    
    metadata = result["metadatas"][0]
    document = result["documents"][0]
    
    old_count = metadata.get("usage_count", 0)
    new_count = old_count + 1
    metadata["usage_count"] = new_count
    
    # Update in place
    collection.update(
        ids=[role_id],
        documents=[document],
        metadatas=[metadata]
    )
    
    # Also update the source markdown file if it exists
    filepath = metadata.get("filepath")
    if filepath and Path(filepath).exists():
        content = Path(filepath).read_text(encoding="utf-8")
        updated = content.replace(
            f"usage_count: {old_count}",
            f"usage_count: {new_count}"
        )
        if updated != content:
            Path(filepath).write_text(updated, encoding="utf-8")
            print(f"✓ Updated {role_id}: usage_count {old_count} → {new_count} (DB + source file)")
        else:
            print(f"✓ Updated {role_id}: usage_count {old_count} → {new_count} (DB only)")
    else:
        print(f"✓ Updated {role_id}: usage_count {old_count} → {new_count} (DB only)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Increment role usage count")
    parser.add_argument("role_id", help="Role ID to increment (e.g. cardiologist)")
    parser.add_argument("--db-path", default=None, help="ChromaDB path")
    args = parser.parse_args()
    
    update_usage(args.role_id, args.db_path)
