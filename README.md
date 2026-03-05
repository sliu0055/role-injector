# role-injector

A Claude Code skill that automatically detects domain-specific questions and injects the best-matching expert role prompt to improve response quality.

When you ask Claude a medical, legal, financial, engineering, or other expert question, role-injector finds the right specialist role from a library of ~500 roles and frames the answer in that expert's voice.

---

## How It Works

1. Claude assesses your question for domain signals
2. Queries a ChromaDB vector database of ~500 expert roles using semantic similarity
3. Injects the best-matching role prompt before answering
4. Tracks usage so frequently-used roles rank higher over time

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/role-injector.git ~/.claude/skills/role-injector
```

### 2. Set the environment variable

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
export ROLE_INJECTOR_HOME=~/.claude/skills/role-injector
```

### 3. Install dependencies

```bash
pip install chromadb sentence-transformers
# or: pip install -r "$ROLE_INJECTOR_HOME/requirements.txt"
```

### 4. Initialize the database

```bash
python "$ROLE_INJECTOR_HOME/scripts/init_roledb.py"   --roles-dir "$ROLE_INJECTOR_HOME/roles"   --db-path ~/.role-injector/roledb
```

Re-run this step whenever you add new roles.

---

## Usage

Claude Code activates this skill automatically when you ask domain-specific questions.

**Example questions that trigger it:**
- "What are the risks of aortic stenosis?"
- "How should I structure a Series A term sheet?"
- "What design pattern should I use for an event-driven microservices system?"
- "What are the HKIAC arbitration rules for expedited proceedings?"

**Example output:**

```
Role: Cardiologist
Board-certified Cardiologist with 15 years of clinical experience

---
Aortic stenosis carries several significant risks...
```

---

## Role Library

Roles are organized by domain under `roles/`:

```
roles/
├── medical/          # 80+ roles: cardiologist, neurologist, surgeon...
├── legal/            # 50+ roles: patent attorney, arbitration specialist...
├── technology/       # 80+ roles: software architect, LLM researcher...
├── finance/          # 40+ roles: CFO, quant analyst, actuary...
├── science/          # 40+ roles: physicist, biochemist, epidemiologist...
└── ...               # business, education, government, arts, trades...
```

~500 total roles across 12 domains. See `references/role-catalog.md` for the full list.

---

## Adding New Roles

1. Copy a role file (e.g. `roles/medical/cardiologist.md`) as a template
2. Fill in the fields following `references/role-schema.md`
3. Save to the correct domain subfolder
4. Re-run `init_roledb.py` to index it

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/init_roledb.py` | Index all role `.md` files into ChromaDB |
| `scripts/query_role.py` | Query the DB for the best-matching role |
| `scripts/update_usage.py` | Increment a role's usage counter |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ROLE_INJECTOR_HOME` | *(required for scripts)* | Path to this skill's directory |
| `ROLE_INJECTOR_DB_PATH` | `~/.role-injector/roledb` | ChromaDB storage location |

---

## Requirements

- Python 3.9+
- `chromadb >= 0.4.0`
- `sentence-transformers >= 2.2.0`
- Claude Code with skills support
