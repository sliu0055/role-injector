#!/usr/bin/env python3
"""
init_roledb.py — Index all role markdown files into ChromaDB.

Usage:
    python init_roledb.py --roles-dir ./roles --db-path ~/.role-injector/roledb

Run this once initially, and again whenever you add new roles.
"""

import argparse
import os
import re
import json
from pathlib import Path

def parse_role_file(filepath: Path) -> dict | None:
    """Parse a role markdown file into a structured dict."""
    content = filepath.read_text(encoding="utf-8")
    
    # Extract YAML frontmatter
    fm_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not fm_match:
        print(f"  [SKIP] No frontmatter: {filepath}")
        return None
    
    fm_text = fm_match.group(1)
    
    def fm_get(key):
        m = re.search(rf'^{key}:\s*(.+)$', fm_text, re.MULTILINE)
        return m.group(1).strip() if m else ""
    
    role_id = fm_get("id")
    name = fm_get("name")
    domain = fm_get("domain")
    subdomain = fm_get("subdomain")
    usage_count = int(fm_get("usage_count") or 0)
    
    # Parse aliases list
    aliases_match = re.search(r'^aliases:\s*\[(.+?)\]', fm_text, re.MULTILINE)
    aliases = []
    if aliases_match:
        aliases = [a.strip().strip('"\'') for a in aliases_match.group(1).split(',')]
    
    # Extract expertise bullets
    expertise_match = re.search(r'## Expertise\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
    expertise_text = expertise_match.group(1).strip() if expertise_match else ""
    
    # Extract approach bullets
    approach_match = re.search(r'## Approach\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
    approach_text = approach_match.group(1).strip() if approach_match else ""
    
    # Build embedding text: name + domain + subdomain + aliases + expertise + approach
    embed_parts = [name, domain, subdomain] + aliases
    embed_parts += [line.lstrip("- ") for line in expertise_text.splitlines() if line.strip().startswith("-")]
    embed_parts += [line.lstrip("- ") for line in approach_text.splitlines() if line.strip().startswith("-")]
    embed_text = " ".join(filter(None, embed_parts))
    
    # Extract prompt template
    template_match = re.search(r'## Prompt Template\n\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
    prompt_template = template_match.group(1).strip() if template_match else f"You are a {name}. Answer with the expertise of a seasoned professional in {subdomain}."
    
    return {
        "id": role_id,
        "name": name,
        "domain": domain,
        "subdomain": subdomain,
        "aliases": aliases,
        "usage_count": usage_count,
        "embed_text": embed_text,
        "prompt_template": prompt_template,
        "filepath": str(filepath),
    }


def init_db(roles_dir: str, db_path: str):
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print("ERROR: chromadb not installed. Run:")
        print("  pip install chromadb sentence-transformers --break-system-packages")
        return

    roles_path = Path(roles_dir)
    if not roles_path.exists():
        print(f"ERROR: roles directory not found: {roles_dir}")
        return

    print(f"Initializing ChromaDB at: {db_path}")
    os.makedirs(db_path, exist_ok=True)
    
    client = chromadb.PersistentClient(path=db_path)
    
    # Use sentence-transformers for embedding
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create or reset the collection
    try:
        client.delete_collection("roles")
        print("Dropped existing 'roles' collection.")
    except Exception:
        pass
    
    collection = client.create_collection(
        name="roles",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Index all role files
    role_files = list(roles_path.rglob("*.md"))
    print(f"Found {len(role_files)} role files.")
    
    indexed = 0
    skipped = 0
    
    for filepath in role_files:
        role = parse_role_file(filepath)
        if not role or not role["id"]:
            skipped += 1
            continue
        
        # Boost score by usage_count (stored as metadata for post-retrieval reranking)
        collection.add(
            ids=[role["id"]],
            documents=[role["embed_text"]],
            metadatas=[{
                "name": role["name"],
                "domain": role["domain"],
                "subdomain": role["subdomain"],
                "usage_count": role["usage_count"],
                "prompt_template": role["prompt_template"],
                "filepath": role["filepath"],
            }]
        )
        indexed += 1
        print(f"  ✓ {role['id']} ({role['domain']})")
    
    print(f"\nDone! Indexed {indexed} roles, skipped {skipped}.")
    print(f"DB ready at: {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index roles into ChromaDB")
    parser.add_argument("--roles-dir", default="./roles", help="Directory containing role .md files")
    parser.add_argument("--db-path", default=os.path.expanduser("~/.role-injector/roledb"), help="ChromaDB path")
    args = parser.parse_args()
    
    init_db(args.roles_dir, args.db_path)
