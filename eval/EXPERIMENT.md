# Role Injection Evaluation Experiment

## Research Question

Does injecting a domain-specific expert role prompt improve LLM accuracy on domain-specific questions, and does automated retrieval of the right role outperform simpler approaches?

---

## Claims

This research aims to test four claims. The current implementation covers Claims 1 and 2. Claims 3 and 4 require additional infrastructure and are scoped under Future Experiments.

| # | Claim | Status |
|---|-------|--------|
| 1 | Role injection improves accuracy over baseline | **Tested** — not supported (see results below) |
| 2 | The specific role matters — a generic "be an expert" nudge is not equivalent | **Tested** — reversed: generic outperformed specific role |
| 3 | Semantic retrieval (ChromaDB) picks better roles than keyword matching or random selection | Not yet implemented |
| 4 | Usage-boosted scoring converges to better role selection over time | Not yet implemented |

### What each claim needs

**Claim 1** (tested): Compare accuracy of baseline (no prompt) vs role_injected (cardiologist prompt) on MedQA. If role_injected > baseline, claim holds.

**Claim 2** (tested): Compare role_injected vs generic_expert ("You are a medical expert.") on MedQA. If role_injected > generic_expert, the detailed role adds value beyond a simple nudge. A wrong-domain condition (e.g., software architect prompt on medical questions) would further strengthen this claim.

**Claim 3** (future): Index 15+ roles across multiple domains into ChromaDB. For each question, compare:
- Role selected by semantic retrieval (ChromaDB `query_role.py`)
- Role selected by keyword matching (string overlap with role catalog)
- Randomly selected role from the same domain
- Randomly selected role from any domain

Measure accuracy under each retrieval strategy. If semantic retrieval yields the highest accuracy, claim holds.

**Claim 4** (future): Simulate usage accumulation over a sequence of queries. Compare retrieval quality (top-1 accuracy of role selection) at the start (all usage_count = 0) vs after N queries with usage boosting. Alternatively, collect real usage data over time and measure whether frequently-used roles are retrieved more reliably for repeat query types.

---

## Results (Run 1)

**Model**: Claude Haiku 4.5 (`claude-haiku-4-5-20251001`)
**Dataset**: MedQA English, 4-option, test split, first 200 questions
**Date**: 2026-03-15

### Accuracy

| Condition | Correct | Total | Accuracy | Parse Failures |
|-----------|---------|-------|----------|----------------|
| Baseline (no prompt) | 159 | 200 | 79.5% | 0 |
| Role-injected (cardiologist) | 161 | 200 | 80.5% | 0 |
| Generic expert | 166 | 200 | 83.0% | 0 |

### Comparisons

| Comparison | Delta | Significant? |
|------------|-------|--------------|
| Role-injected vs baseline | +1.0% | No (chi2=0.40) |
| Generic expert vs baseline | +3.5% | **Yes** (chi2=7.00, p<0.05) |
| Role-injected vs generic expert | -2.5% | No (chi2=2.27) |

### Paired analysis (McNemar contingency)

**Baseline vs role_injected** (n=200): Both correct: 155, only baseline: 4, only role_injected: 6, both wrong: 35. Not significant.

**Baseline vs generic_expert** (n=200): Both correct: 159, only baseline: 0, only generic_expert: 7, both wrong: 34. Statistically significant (p<0.05).

**Generic_expert vs role_injected** (n=200): Both correct: 158, only generic_expert: 8, only role_injected: 3, both wrong: 31. Not significant.

### Interpretation

The results match the interpretation guide row: **"Cond 2 ≈ Cond 3 > Cond 1"** — any expert nudge helps, but the detailed role adds no measurable value over a generic one.

Notably, the generic expert prompt ("You are a medical expert.") outperformed the specific cardiologist role. A likely explanation: MedQA covers all of medicine (anatomy, pharmacology, pathology, surgery, etc.), while the cardiologist prompt narrows the model's focus to cardiovascular topics. The specificity may hurt on non-cardiology questions.

**This suggests a follow-up experiment**: filter MedQA to cardiology-related questions only, where the cardiologist role's specificity should be an advantage rather than a constraint.

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

The script supports two providers via the `--provider` flag.

### Claude (recommended)

| Property | Value |
|----------|-------|
| Provider | Anthropic |
| Model | `claude-haiku-4-5-20251001` |
| SDK | `anthropic` |
| Cost | ~$0.25/1M input tokens (~$1 for full 1,273 questions) |
| Delay | 1 second (generous rate limits) |

### Gemini

| Property | Value |
|----------|-------|
| Provider | Google Gemini (free tier) |
| Model | `gemini-2.5-flash` |
| SDK | `google-genai` |
| Cost | Free |
| Rate limit | ~15 RPM, ~1,500 RPD |
| Delay | 4.5 seconds (to fit within RPM limit) |

**Note**: The Gemini free tier throttled aggressively in our initial run (582/600 calls rate-limited). Claude is more reliable.

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
  EXPERIMENT.md    — This file (experiment design + results)
  run_medqa.py     — Main script: load MedQA → call LLM → save CSV
  analyze_medqa.py — Read CSV → compute accuracy + statistical tests
  results.csv      — Raw results (one row per question × condition)
```

### run_medqa.py behavior

1. Load MedQA test split via HuggingFace `datasets` library
2. Sample N questions (default 200)
3. For each question, run all 3 conditions sequentially
4. Write each result row to CSV immediately (crash-safe, resume-safe)
5. Resume support: skip question/condition pairs already in CSV
6. Rate limit: configurable delay (default 1.0s for Claude, 4.5s for Gemini)

### analyze_medqa.py behavior

1. Read results CSV
2. Compute accuracy per condition
3. Print summary table with deltas between conditions
4. McNemar's chi-squared test for paired statistical significance

---

## How to Run

### Prerequisites

- Python 3.9+
- An API key: Anthropic (recommended) or Google Gemini (free)

### Step 1: Install dependencies

```bash
# From the repo root
pip install -r requirements.txt
```

Or install just the eval dependencies:

```bash
# For Claude
pip install datasets anthropic

# For Gemini
pip install datasets google-genai
```

### Step 2: Set your API key

```bash
# Claude
export ANTHROPIC_API_KEY="your-key-here"

# Or Gemini
export GEMINI_API_KEY="your-key-here"
```

### Step 3: Run the evaluation

```bash
# Claude (default, recommended)
python eval/run_medqa.py --provider claude --num-questions 200

# Gemini
python eval/run_medqa.py --provider gemini --num-questions 200 --delay 4.5

# Quick sanity check (any provider)
python eval/run_medqa.py --provider claude --num-questions 5
```

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
| `--provider` | `claude` | API provider: `claude` or `gemini` |
| `--num-questions` | `200` | Number of MedQA questions to evaluate |
| `--model` | Provider default | Model name (e.g., `claude-haiku-4-5-20251001`) |
| `--output` | `eval/results.csv` | Output CSV path |
| `--delay` | `1.0` | Seconds between API calls |
| `--split` | `test` | HuggingFace dataset split |

**analyze_medqa.py**

```bash
python eval/analyze_medqa.py <path-to-results.csv>
```

No flags — just pass the CSV path as the first argument.

---

## Time and cost estimate

### Claude (recommended)

| Parameter | Value |
|-----------|-------|
| Questions | 200 |
| Conditions | 3 |
| Total API calls | 600 |
| Delay per call | 1s |
| Estimated runtime | ~10 minutes |
| Cost | ~$0.10 |

### Gemini (free)

| Parameter | Value |
|-----------|-------|
| Questions | 200 |
| Conditions | 3 |
| Total API calls | 600 |
| Delay per call | 4.5s |
| Estimated runtime | ~45 minutes |
| Cost | Free (but may hit daily quota) |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `datasets` | Load MedQA from HuggingFace |
| `anthropic` | Call Claude API (recommended provider) |
| `google-genai` | Call Gemini API (free alternative) |

---

## Future experiments

### Follow-up on Claims 1 & 2

Based on Run 1 results, the cardiologist role may be too narrow for the broad MedQA dataset. Next steps:

- **Filter to cardiology questions**: Subset MedQA to questions containing cardiovascular keywords (e.g., "heart", "cardiac", "arrhythmia", "ECG", "myocardial"). The cardiologist role should show a larger advantage on domain-matched questions.
- **Wrong-domain condition (Claim 2)**: Add a 4th condition — inject software architect role on medical questions. If it performs worse than cardiologist and similar to or worse than baseline, this proves domain-specificity matters.
- **Multiple medical roles**: Add a general practitioner role and compare against the cardiologist on broad MedQA. The GP role may outperform the specialist on general questions.
- **Larger sample**: Scale from 200 to full 1,273 test questions.
- **Multiple models**: Run same experiment on Sonnet, Opus, Gemini, Llama to test whether the effect generalizes across model sizes and providers.
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
