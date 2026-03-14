# Role Injection Evaluation Experiment

## Research Question

Does injecting a domain-specific expert role prompt improve LLM accuracy on domain-specific questions, and does automated retrieval of the right role outperform simpler approaches?

---

## Claims

This research aims to test four claims. The current implementation covers Claims 1 and 2. Claims 3 and 4 require additional infrastructure and are scoped under Future Experiments.

| # | Claim | Status |
|---|-------|--------|
| 1 | Role injection improves accuracy over baseline | **Implemented** — baseline vs role_injected conditions |
| 2 | The specific role matters — a generic "be an expert" nudge is not equivalent | **Implemented** — generic_expert condition as control |
| 3 | Semantic retrieval (ChromaDB) picks better roles than keyword matching or random selection | Not yet implemented — requires multiple indexed roles and a retrieval comparison harness |
| 4 | Usage-boosted scoring converges to better role selection over time | Not yet implemented — requires usage simulation or longitudinal data collection |

### What each claim needs

**Claim 1** (current): Compare accuracy of baseline (no prompt) vs role_injected (cardiologist prompt) on MedQA. If role_injected > baseline, claim holds.

**Claim 2** (current): Compare role_injected vs generic_expert ("You are a medical expert.") on MedQA. If role_injected > generic_expert, the detailed role adds value beyond a simple nudge. A wrong-domain condition (e.g., software architect prompt on medical questions) would further strengthen this claim.

**Claim 3** (future): Index 15+ roles across multiple domains into ChromaDB. For each question, compare:
- Role selected by semantic retrieval (ChromaDB `query_role.py`)
- Role selected by keyword matching (string overlap with role catalog)
- Randomly selected role from the same domain
- Randomly selected role from any domain

Measure accuracy under each retrieval strategy. If semantic retrieval yields the highest accuracy, claim holds.

**Claim 4** (future): Simulate usage accumulation over a sequence of queries. Compare retrieval quality (top-1 accuracy of role selection) at the start (all usage_count = 0) vs after N queries with usage boosting. Alternatively, collect real usage data over time and measure whether frequently-used roles are retrieved more reliably for repeat query types.

---

## Dataset

**MedQA** — Free-form multiple-choice questions from USMLE (US Medical Licensing Exam).

| Property | Value |
|----------|-------|
| Source | [bigbio/med_qa](https://huggingface.co/datasets/bigbio/med_qa) |
| Config | `med_qa_en_4options_source` |
| Split | `test` (1,273 questions) |
| Sample size | 200 questions (initial run) |
| Format | 4-option MC (A–D) |
| Fields | `question`, `options` (list of `{key, value}`), `answer_idx` (letter), `answer` (text) |
| Requires | `trust_remote_code=True`, `datasets` library |

---

## Conditions (3)

### Condition 1: Baseline

No system prompt. The question is sent to the model as-is.

> **User**: "A 65-year-old male presents with chest pain... A) Pericarditis B) Inferior MI C) Aortic dissection D) Pulmonary embolism"

Measures the model's **default** performance with no steering.

### Condition 2: Role-injected (Cardiologist)

System prompt set to the cardiologist role from `roles/medical/cardiologist.md`:

> **System**: "You are a board-certified Cardiologist with 15 years of clinical experience. You have deep expertise in heart disease, cardiac imaging, arrhythmias, and cardiovascular risk management. When answering, systematically consider the clinical picture, reference ACC/AHA guidelines where applicable, and clearly flag any red flags or urgency signals. Note important caveats a non-specialist might miss."

Measures whether a **detailed expert role** improves accuracy.

### Condition 3: Generic expert

System prompt set to a minimal expert cue:

> **System**: "You are a medical expert."

This is the critical control. It answers: **does the detailed role matter, or does any "be an expert" nudge produce the same effect?**

### Interpretation guide

| Result | Interpretation |
|--------|----------------|
| Cond 2 > Cond 1 > Cond 3 ≈ Cond 1 | Role helps, generic nudge doesn't — strong evidence for role injection |
| Cond 2 ≈ Cond 3 > Cond 1 | Any expert nudge helps equally — role detail adds no value |
| Cond 2 ≈ Cond 3 ≈ Cond 1 | Role injection has no measurable effect |
| Cond 1 > Cond 2 | Role injection hurts — the extra context is noise |

---

## Model

| Property | Value |
|----------|-------|
| Provider | Google Gemini (free tier) |
| Model | `gemini-2.5-flash` |
| SDK | `google-genai` (new unified SDK, not legacy `google-generativeai`) |
| Rate limit (free) | ~15 RPM, ~1,500 RPD, ~1M TPM |
| Delay between calls | 4.5 seconds (fits within 15 RPM) |

### API call pattern

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=API_KEY)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction="<system prompt or None>"
    ),
    contents="<question + options + answer instruction>"
)
```

---

## Answer extraction

The prompt ends with:

> "Respond with ONLY the letter of the correct answer (A, B, C, or D). Do not include any explanation."

Extraction logic (in priority order):
1. Exact single letter match (A–D)
2. Leading letter with punctuation: `B.` or `B)`
3. "The answer is B" pattern
4. Last single letter on its own line
5. If none match → `PARSE_FAIL`

---

## File structure

```
eval/
  EXPERIMENT.md    — This file (experiment design)
  run_medqa.py     — Main script: load MedQA → call Gemini → save CSV
  analyze_medqa.py — Read CSV → compute accuracy + statistical tests
  results.csv      — Raw results (one row per question × condition)
```

### run_medqa.py behavior

1. Load MedQA test split via HuggingFace `datasets` library
2. Sample N questions (default 200)
3. For each question, run all 3 conditions sequentially
4. Write each result row to CSV immediately (crash-safe, resume-safe)
5. Resume support: skip question/condition pairs already in CSV
6. Rate limit: configurable delay (default 4.5s)

### analyze_medqa.py behavior

1. Read results CSV
2. Compute accuracy per condition
3. Print summary table with deltas between conditions

---

## How to Run

### Prerequisites

- Python 3.10+
- A Google Gemini API key (free) from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### Step 1: Install dependencies

```bash
# From the repo root
pip install -r requirements.txt
```

Or install just the eval dependencies:

```bash
pip install datasets google-genai
```

### Step 2: Set your API key

```bash
export GEMINI_API_KEY="your-key-here"
```

### Step 3: Run the evaluation

```bash
# Default: 200 questions, gemini-2.5-flash, outputs to eval/results.csv
python eval/run_medqa.py

# Custom options
python eval/run_medqa.py --num-questions 50 --model gemini-2.5-flash --output eval/results.csv
```

This takes ~45 minutes for 200 questions (600 API calls at 4.5s intervals).

**If the script is interrupted**, re-run the same command — it resumes from where it left off. Already-evaluated question/condition pairs are skipped.

### Step 4: Analyze results

```bash
python eval/analyze_medqa.py eval/results.csv
```

This prints:
- Accuracy per condition (baseline, role_injected, generic_expert)
- Parse failure counts
- Pairwise accuracy deltas
- McNemar's chi-squared test for statistical significance (p < 0.05 threshold)

### CLI reference

**run_medqa.py**

| Flag | Default | Description |
|------|---------|-------------|
| `--num-questions` | `200` | Number of MedQA questions to evaluate |
| `--model` | `gemini-2.5-flash` | Gemini model name |
| `--output` | `eval/results.csv` | Output CSV path |
| `--delay` | `4.5` | Seconds between API calls (free tier: 15 RPM) |
| `--split` | `test` | HuggingFace dataset split |

**analyze_medqa.py**

```bash
python eval/analyze_medqa.py <path-to-results.csv>
```

No flags — just pass the CSV path as the first argument.

---

## Time and cost estimate

| Parameter | Value |
|-----------|-------|
| Questions | 200 |
| Conditions | 3 |
| Total API calls | 600 |
| Delay per call | 4.5s |
| Estimated runtime | ~45 minutes |
| Cost | Free (Gemini free tier) |

To run faster, reduce `--delay` if your quota allows, or reduce `--num-questions` for a quick sanity check (e.g., `--num-questions 10` takes ~2 minutes).

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `datasets` | Load MedQA from HuggingFace |
| `google-genai` | Call Gemini API (new unified SDK) |

---

## Future experiments

### Strengthening Claims 1 & 2

- **Wrong-domain condition (Claim 2)**: Add a 4th condition — inject software architect role on medical questions. If it performs worse than cardiologist but similar to baseline, this proves domain-specificity matters.
- **Larger sample**: Scale from 200 to full 1,273 test questions once pilot results are promising.
- **Multiple models**: Run same experiment on Haiku, Sonnet, Llama to test whether the effect generalizes across model sizes and providers.
- **MMLU subsets**: Extend to `clinical_knowledge`, `medical_genetics`, `anatomy` for broader medical coverage.
- **Cross-domain**: Add legal (MMLU `professional_law`) with legal roles to test whether findings hold outside medicine.

### Claim 3: Retrieval quality

- Write 15+ role files across medical, legal, and tech domains.
- Index all roles in ChromaDB via `init_roledb.py`.
- Build a retrieval comparison harness that, for each question, selects a role using:
  1. Semantic retrieval (`query_role.py`)
  2. Keyword matching against `role-catalog.md`
  3. Random role from the correct domain
  4. Random role from any domain
- Run each retrieved role as the system prompt and compare accuracy.

### Claim 4: Usage boosting

- Simulate a sequence of 500+ queries where `update_usage.py` increments counts after each use.
- At intervals (0, 100, 200, 500 queries), measure top-1 retrieval precision: does the system select the "ideal" role more often as usage data accumulates?
- Alternatively, deploy the skill in real use and log retrieval choices over weeks, then analyze convergence.
