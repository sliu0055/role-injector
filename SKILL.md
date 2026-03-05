---
name: role-injector
description: Automatically detects when a user question requires domain expertise and injects the best-matching expert role prompt to improve response quality. Use this skill whenever a user asks a complex, technical, professional, or domain-specific question — including medical, legal, engineering, financial, scientific, psychological, or any question that would benefit from expert framing. Also triggers when a user explicitly asks for expert advice, analysis, diagnosis, or professional opinion. The more a role is used, the higher it ranks in future retrievals. If the user's question is simple or conversational, skip this skill.
---

# Role Injector

Automatically finds and injects the best expert role for any domain-specific question, drawing from a comprehensive role library stored as markdown files with ChromaDB for fast retrieval.

## When to Use This Skill

Activate when the user's question has **any** of these signals:
- Asks for analysis, diagnosis, advice, or professional judgment
- Uses technical or domain-specific vocabulary
- Is multi-step, has trade-offs, or requires specialized knowledge
- Mentions a professional domain (medical, legal, financial, engineering, etc.)
- Would clearly benefit from an expert framing vs. a generalist answer

Skip for simple factual lookups, casual conversation, or creative writing with no domain expertise needed.

---

## Workflow

### Step 1: Assess the Question

Quickly classify the question:
- **Domain**: What field does this belong to? (e.g., cardiology, contract law, structural engineering)
- **Complexity**: Simple (skip skill) / Moderate / Complex / Expert-level
- **Intent**: Is the user seeking advice, analysis, explanation, or diagnosis?

If complexity is Simple → respond normally without role injection.

### Step 2: Find the Best Role

**If ChromaDB is available** (preferred):
```bash
python "$ROLE_INJECTOR_HOME/scripts/query_role.py" "<user question>" --top-k 3
```
Pick the highest-scored role. If score < 0.5, fall back to keyword matching.

> Set `ROLE_INJECTOR_HOME` to the directory where you cloned/installed this skill,
> e.g. `export ROLE_INJECTOR_HOME=~/.claude/skills/role-injector` in your shell profile.

**If ChromaDB is NOT available** (fallback):
- Match keywords from the question against role domains in `references/role-catalog.md`
- Pick the closest match by domain + sub-specialty

### Step 3: Inject the Role

Before your answer, output:

```
🎭 **Role**: [Role Name]
*[One-sentence expertise description]*

---
```

Then answer the question fully in that role's voice, using the role's prompt template from the role's markdown file.

### Step 4: Update Usage Counter

```bash
python "$ROLE_INJECTOR_HOME/scripts/update_usage.py" "<role_id>"
```

This increments the role's usage count, which boosts its retrieval ranking over time.

---

## Role Library Structure

Each role is a markdown file at `roles/<domain>/<role-id>.md`:

```
roles/
├── medical/
│   ├── cardiologist.md
│   ├── neurologist.md
│   └── ...
├── legal/
│   ├── contract-lawyer.md
│   ├── patent-attorney.md
│   └── ...
├── engineering/
│   ├── software-architect.md
│   ├── structural-engineer.md
│   └── ...
└── ...
```

See `references/role-schema.md` for the exact markdown format each role file must follow.
See `references/role-catalog.md` for the full list of ~500 roles organized by domain.

---

## Database Setup

### First-time initialization
```bash
pip install chromadb sentence-transformers
python "$ROLE_INJECTOR_HOME/scripts/init_roledb.py" --roles-dir "$ROLE_INJECTOR_HOME/roles" --db-path ~/.role-injector/roledb
```

This indexes all role markdown files into ChromaDB using sentence-transformers embeddings. Re-run whenever you add new roles.

### Database location
Default: `~/.role-injector/roledb/`
Override with: `ROLE_INJECTOR_DB_PATH` env variable.

---

## Adding New Roles

1. Create a new `.md` file following `references/role-schema.md`
2. Place it in the correct domain subfolder under `roles/`
3. Re-run `init_roledb.py` to index it

---

## Reference Files

- `references/role-schema.md` — Template and field definitions for role markdown files
- `references/role-catalog.md` — Full catalog of ~500 roles by domain (source of truth)
- `scripts/init_roledb.py` — Initialize / re-index ChromaDB from role markdown files
- `scripts/query_role.py` — Query ChromaDB for best-matching role
- `scripts/update_usage.py` — Increment usage counter for a role
