# When Does Persona Prompting Actually Help? A Retrieval and Metric Analysis of Expert Role Injection in LLMs

## 1. Introduction
- LLM persona/role prompting is widely used but poorly understood
- Common assumption: "be an expert" improves response quality
- Our finding: role injection is a **bad average** — it reshapes responses rather than improving them, trading clarity for expertise depth
- The effect is domain-dependent: helps medical/psychology, hurts finance/legal/science/tech
- Contributions:
  1. Retrieval comparison: 4 strategies x 6 domains x 1140 questions
  2. Metric decomposition: reveals the clarity-expertise tradeoff invisible to aggregate scoring
  3. Hybrid retrieval method: embedding shortlist + LLM selection outperforms embedding-only

## 2. Related Work
- Prompt engineering and system instructions
- Persona/role prompting in LLMs
- LLM-as-judge evaluation methodology
- Retrieval-augmented generation (RAG) — our role retrieval is related

## 3. Method

### 3.1 Role Prompt Database
- 38 expert roles across 6 domains (medical, psychology, finance, legal, science, technology)
- Stored in ChromaDB with sentence-transformer embeddings (all-MiniLM-L6-v2)

### 3.2 Experimental Conditions
- **Baseline**: no system prompt
- **General expert**: one-line generic domain prompt ("You are a knowledgeable expert in {domain}...")
- **Embedding**: ChromaDB top-1 by cosine similarity
- **Hybrid**: ChromaDB top-5 shortlist -> Gemini Flash selects best role

### 3.3 Evaluation Setup
- 1,140 open-ended questions (30 per role x 38 roles, 3 types: advisory, conceptual, practical)
- Answerer: GPT-4o mini (same across all conditions)
- Selector: Gemini Flash (different provider to prevent confounding)
- Judge: Claude Haiku (blind, multi-metric)
- Metrics: accuracy, expertise_depth, relevance, safety, clarity, time_sensitive_correctness

### 3.4 Statistical Tests
- Friedman test (4 related groups)
- Pairwise Wilcoxon signed-rank with Bonferroni correction
- Cohen's d effect sizes

## 4. Results

### 4.1 Overall Scores
- No significant difference between baseline (4.390) and hybrid (4.382)
- Both significantly beat embedding (4.349)
- All effect sizes negligible on aggregate avg_score (Cohen's d < 0.12)

### 4.2 Metric Decomposition
- **Expertise depth**: role injection gains +0.28 (medium Cohen's d = 0.54, p < 0.001)
- **Safety**: hybrid best (4.81), general expert worst (4.63), p < 0.001
- **Clarity**: baseline dominates (4.90 vs ~4.55), p < 0.001
- **Accuracy**: baseline and general expert slightly higher (~4.05 vs ~4.00)
- Key insight: gains and losses cancel out in aggregate, masking the tradeoff

### 4.3 Domain Analysis
- **Hybrid wins**: medical (4.457 vs 4.394, p < 0.001), psychology (trend, p = 0.07)
- **Baseline wins**: finance (p < 0.001), legal (p < 0.01), science (p < 0.001), technology (p < 0.001)

### 4.4 Question Type
- **Advisory** (n=850): hybrid best (4.403) — role injection helps "what should I do" questions
- **Conceptual** (n=284): baseline best (4.435) — role injection hurts "explain this" questions

### 4.5 Retrieval Precision
- 85.8% exact role match, 11.4% different role, 2.8% generic fallback
- Mismatches are mostly sensible substitutions (doctor -> specialist)
- Wrong-role performance not significantly different from correct-role (p = 0.13)
- Medical has lowest precision (80.8%) but highest hybrid benefit

### 4.6 Response Length Confound
- Role injection produces 31% longer responses (3095 vs 2361 chars)
- Length correlates **negatively** with score for role-injected conditions (rho = -0.29)
- Length-normalized scores still favor hybrid (+0.015), confirming the effect is not verbosity bias

## 5. Judge Robustness
- GPT-4o re-judges 200-question stratified subset (800 judgments)
- GPT-4o scores systematically higher (mean diff +0.43) — more lenient grader, not contradicting
- Condition ranking differs due to small between-condition gaps (~0.04 range)
- Both judges agree: between-condition differences are consistently small
- Expertise-depth advantage of role injection is consistent across both judges
- Reinforces core finding: role injection produces marginal overall change with metric-specific tradeoffs

## 6. Discussion
- Role injection is not universally helpful — it is a tradeoff between expertise depth and clarity
- The clarity cost likely reflects role prompts encouraging domain jargon and verbose hedging
- Domain sensitivity: domains with established expert communication norms (medicine, psychology) benefit; domains where plain language is valued (finance, legal) do not
- Practical recommendation: use role injection for advisory medical/psychological queries; skip it for factual/conceptual questions
- The small effect sizes approach the resolution limit of LLM-based evaluation; future work should complement automated judging with human expert evaluation to validate fine-grained quality differences

## 7. Limitations
- Single answerer model (GPT-4o mini) — generalizability to other models unknown
- Synthetic questions — real user queries may differ
- No human evaluation
- No wrong-domain ablation (e.g., software architect prompt on medical questions)
- 38-role corpus is small relative to real-world role diversity

## 8. Conclusion
- Role injection reshapes LLM responses rather than uniformly improving them
- Multi-metric evaluation is essential — single-score metrics mask the expertise-clarity tradeoff
- When to use it: advisory questions in specialized domains (medical, psychology)
- When to skip it: conceptual questions and domains valuing plain language
- Hybrid retrieval (embedding + LLM) significantly outperforms embedding-only selection
