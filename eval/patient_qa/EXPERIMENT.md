# Patient Q&A Experiment: Role Injection Evaluation

## Overview

This experiment tests whether injecting an expert role prompt improves an LLM's responses to open-ended patient questions — questions a real patient might ask during a doctor's visit.

Unlike the prior MedQA experiment (multiple-choice USMLE questions), this dataset uses free-form, conversational questions and evaluates response quality holistically via an LLM-as-judge.

---

## Research Question

Does injecting a detailed General Practitioner role prompt produce better patient-facing responses than a generic expert prompt or no prompt at all?

---

## Methods

### Dataset

30 hand-crafted patient questions spanning 10 medical domains:

| Domain | Questions |
|---|---|
| Cardiology | 5 |
| General medicine | 4 |
| Neurology | 3 |
| Gastroenterology | 3 |
| Endocrinology | 3 |
| Respiratory | 3 |
| Musculoskeletal | 2 |
| Infectious disease | 2 |
| Mental health | 2 |
| Dermatology / Nephrology / Hematology | 1 each |

Questions are open-ended and conversational (e.g., *"Doctor, I've been having chest pain for three days that gets worse when I climb stairs. I'm 55 with high blood pressure. Should I be worried?"*). No multiple choice — full free-text answers are evaluated.

### Conditions

| Condition | System Prompt |
|---|---|
| **baseline** | None |
| **role_injected** | Full GP role from `roles/medical/doctor.md` (board-certified GP, 20 years experience, systematic reasoning, red-flag guidance, evidence-based guidelines) |
| **generic_expert** | `"You are a medical expert."` |

### Answer Model

- **Provider**: OpenAI
- **Model**: `gpt-4o-mini`
- **Max tokens**: 600

### Judge

- **Model**: `claude-haiku-4-5-20251001`
- **Blind labeling**: Answers are presented as A/B/C with label order rotated deterministically per question (all 6 permutations used) so the judge cannot infer condition from position.

Each answer is scored on four criteria (1–5 scale):

| Criterion | Description |
|---|---|
| `medical_accuracy` | Factually correct, consistent with current evidence |
| `patient_helpfulness` | Helps the patient understand their situation and next steps |
| `safety_guidance` | Flags red flags, urgency levels, and when to seek in-person care |
| `clarity` | Written in plain language a non-medical patient can understand |

The judge also picks an `overall_winner` — the single response it would most want a real patient to receive.

---

## Results

### Overall Winner Count (n=30)

| Condition | Wins | % |
|---|---|---|
| **role_injected** | **27** | **90%** |
| generic_expert | 2 | 7% |
| baseline | 1 | 3% |

### Average Scores by Condition

| Condition | Overall Avg | medical_accuracy | patient_helpfulness | safety_guidance | clarity |
|---|---|---|---|---|---|
| **role_injected** | **4.61** | **4.80** | **4.73** | **4.83** | 4.03 |
| generic_expert | 3.98 | 4.27 | 4.03 | 3.47 | 4.13 |
| baseline | 3.88 | 4.13 | 3.67 | 3.37 | **4.33** |

### Key Observations

1. **Role injection wins decisively on open-ended questions.** The GP role prompt won 27 of 30 questions, compared to 2 for generic expert and 1 for baseline. This is the inverse of the MedQA result, where generic expert outperformed role injection.

2. **Safety guidance is where role injection dominates.** The gap is largest on `safety_guidance`: role_injected (4.83) vs. generic_expert (3.47) vs. baseline (3.37). The role prompt explicitly instructs the model to flag red flags and urgency — and this carries over directly into the output.

3. **Baseline scores highest on clarity.** Without a system prompt, `gpt-4o-mini` produces shorter, simpler responses. These are easier to read but lack clinical depth and safety guidance.

4. **Generic expert underperforms expectations.** Despite a 3.5% boost over baseline in average score, it only wins 2 questions. The minimal prompt improves medical accuracy slightly but does not reliably produce the systematic reasoning and safety flagging of the full role.

### Exceptions (Non-role_injected Wins)

| Q# | Domain | Winner | Reason |
|---|---|---|---|
| Q14 | Endocrinology | baseline | Role-injected answer was truncated (hit token limit mid-response); baseline gave a complete, well-organized answer |
| Q25 | Nephrology | generic_expert | Generic and role-injected were close (4.75 each); judge preferred generic_expert's slightly more actionable organization |
| Q30 | General (weight loss) | generic_expert | Generic_expert gave a more patient-centered, sustainability-focused answer matching the patient's stated concern |

### Scores by Domain

| Domain | role_injected avg | baseline avg | generic_expert avg | Winner |
|---|---|---|---|---|
| Cardiology | 4.75 | 3.85 | 3.75 | role_injected (5/5) |
| Neurology | 4.75 | 3.58 | 3.92 | role_injected (3/3) |
| Gastroenterology | 4.58 | 3.42 | 3.83 | role_injected (3/3) |
| Endocrinology | 4.25 | 3.83 | 4.08 | role_injected (2/3) |
| Respiratory | 4.75 | 3.42 | 3.75 | role_injected (3/3) |
| Musculoskeletal | 4.75 | 4.50 | 4.25 | role_injected (2/2) |
| Infectious disease | 4.50 | 3.75 | 3.88 | role_injected (2/2) |
| Mental health | 4.63 | 4.00 | 3.75 | role_injected (2/2) |
| General | 4.25 | 3.94 | 4.25 | role_injected (2/4), generic (2/4) |

---

## Comparison with Prior MedQA Results

| Experiment | Task type | Best condition | Notes |
|---|---|---|---|
| MedQA Run 1 (broad) | Multiple choice (USMLE) | generic_expert (+3.5% acc) | Role injection +1%, not significant |
| MedQA Run 2 (cardiology) | Multiple choice (USMLE) | baseline (79.4%) | Role injection **hurt** by 5% |
| **Patient Q&A (this run)** | **Open-ended patient questions** | **role_injected (90% win rate)** | **Role injection wins decisively** |

**Hypothesis**: Role injection provides most value when the task requires structured clinical reasoning, safety judgment, and nuanced communication — not when it requires selecting a single correct answer from constrained options. On multiple-choice, the role prompt's verbose reasoning style may conflict with the extraction format; on open-ended questions, that same structure becomes a clear advantage.

---

## Limitations

- **Single judge model**: Using `claude-haiku-4-5-20251001` as judge may favor responses that match Claude's training style (structured, safety-conscious), potentially biasing toward role_injected answers.
- **No ground truth**: Unlike MedQA, there is no objectively correct answer. Judge scores reflect perceived quality, not verified accuracy.
- **Small sample**: 30 questions; domain-level results (1–5 questions each) are not statistically reliable.
- **Single answer model**: Only `gpt-4o-mini` was tested. Results may differ for other models.

---

## Files

| File | Description |
|---|---|
| `questions.json` | 30 patient questions with domain labels |
| `run_patient_qa.py` | Generates answers for all 3 conditions via OpenAI API |
| `judge_patient_qa.py` | LLM-as-judge scoring and winner selection |
| `results.csv` | Raw answers (90 rows: 30 questions × 3 conditions) |
| `judge_results.csv` | Per-question scores and winners from the judge |
