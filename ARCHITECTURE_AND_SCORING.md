# Drug Repurposing Assistant — Architecture & Scoring

> Generated on 2026-02-04. This document describes how the system works end‑to‑end and how the scoring mechanism is computed, based on the current codebase.

## 1) High‑Level System Architecture

### 1.1 Components (Major Layers)
1. **API Layer (FastAPI)**
   - Entry point for clients.
   - Endpoints accept drug + indication queries and return results.
   - See `src/api.py`.

2. **Coordinator / Orchestrator**
   - `MasterAgent` coordinates the full workflow.
   - Normalizes inputs, plans tasks, dispatches agent work, aggregates results.
   - See `src/agents/master_agent.py`.

3. **Specialized Evidence Agents (Worker Agents)**
   - Literature: PubMed/Europe PMC + summarization.
   - Clinical: ClinicalTrials.gov and other registries.
   - Safety: AE profiles, labels, FAERS signals.
   - Molecular: Mechanistic plausibility.
   - Patent: FTO/IP landscape.
   - Market: TAM/competitors.
   - Each returns structured evidence with metadata.

4. **Reasoning & Scoring**
   - `ReasoningAgent` aggregates evidence, applies constraints, computes scores, detects contradictions, and ranks hypotheses.
   - See `src/agents/reasoning_agent.py`.

5. **UI (React / Static)**
   - Renders analysis results returned by the API.
   - See `ui-react/src` or `ui/`.

---

## 2) Detailed Execution Flow (End‑to‑End)

### Step 1 — API Request
- Client calls `/analyze` with:
  - `drug_name`, `indication`, optional `drug_synonyms`, `indication_synonyms`.
  - Options: `include_patent`, `use_internal_data`.
- `FastAPI` parses and validates the request.

### Step 2 — Job Creation & Task Planning
- `MasterAgent.start_job(...)` is invoked.
- **Query normalization**
  - Canonicalizes drug/indication.
  - Expands synonyms for broader search.
- **Task planning**
  - Builds a list of agent tasks based on options.
  - Core agents always run (literature/clinical/safety/molecular).
  - Patent/market/internal are conditional.

### Step 3 — Agent Execution
- Each agent executes its pipeline and returns results:
  - **Literature agent** → papers, claims, evidence items.
  - **Clinical agent** → trials, outcomes, evidence summaries.
  - **Safety agent** → safety score, flags, label data.
  - **Molecular agent** → mechanism and target plausibility.
  - **Patent agent** → FTO score, blocking patents.
  - **Market agent** → TAM, competitors, market signals.

### Step 4 — Result Aggregation
- `ResultAggregator` merges all outputs into:
  - `by_dimension` (structured by evidence dimension)
  - `raw_evidence` (all agent results)
  - `validation_issues` (any failures)

### Step 5 — Reasoning & Scoring
- `ReasoningAgent` runs the main synthesis:
  1. **Evidence aggregation** (normalize into `Evidence` objects)
  2. **Constraint checks** (hard vetoes)
  3. **Dimension scoring** (clinical/safety/patent/market/molecular/regulatory)
  4. **Composite score** (weighted average)
  5. **Contradiction detection**
  6. **Explainability generation**
  7. **Ranking**

### Step 6 — Response
- API returns a JSON response containing:
  - Job metadata
  - Task outputs
  - `reasoning_result` with scores and explanation

---

## 3) Evidence Model

Each piece of evidence is normalized into a unified schema:

- **Evidence**
  - `evidence_id`
  - `source_agent`
  - `dimension`
  - `content`
  - `confidence` (0–1)
  - `polarity` (positive / neutral / negative)
  - `metadata`

These are used for scoring and reasoning.

---

## 4) Scoring Mechanism (Detailed)

### 4.1 Dimension Types
- Clinical Evidence
- Safety Profile
- Patent Freedom
- Market Potential
- Molecular Rationale
- Regulatory Path

### 4.2 Dimension Score (per dimension)
For each dimension:

1. Collect all evidence items for that dimension.
2. Map polarity to numeric values:
   - Positive → **1.0**
   - Neutral → **0.5**
   - Negative → **0.0**
3. Compute **confidence‑weighted average**:

$$
\text{dimension_score} = \frac{\sum_i (c_i \times p_i)}{\sum_i c_i}
$$

Where:
- $c_i$ = evidence confidence (0–1)
- $p_i$ = polarity numeric value

If **no evidence** exists for that dimension:
- `score = 0.5` (neutral default)
- `confidence = 0.0`

### 4.3 Dimension Confidence
Average confidence across evidence items:

$$
\text{dimension_confidence} = \frac{1}{n} \sum_i c_i
$$

### 4.4 Composite Score
Each dimension score is weighted and averaged:

**Weights** (from `DIMENSION_WEIGHTS`):
- Clinical: **0.25**
- Safety: **0.25**
- Patent: **0.15**
- Market: **0.15**
- Molecular: **0.10**
- Regulatory: **0.10**

Composite:

$$
\text{composite} = \frac{\sum_d (w_d \times s_d)}{\sum_d w_d}
$$

Where:
- $w_d$ = dimension weight
- $s_d$ = dimension score

### 4.5 Constraint / Veto Logic
Before computing composite:
- Safety, patent, and regulatory constraints are checked.
- If **any constraint is violated**, composite is forced to **0.0**.

### 4.6 Decision Level Mapping
Decision is determined from composite score (unless vetoed):
- **≥ 0.80** → Highly Recommended
- **≥ 0.65** → Recommended
- **≥ 0.45** → Review Required
- **≥ 0.25** → Not Recommended
- **< 0.25** → Reject

---

## 5) Contradiction Detection

- Evidence is grouped per dimension.
- High‑confidence positive + negative pairs are flagged as contradictions.
- Contradictions are included in the final reasoning output for human review.

---

## 6) Explainability

The explanation summarizes:
- Composite score + decision
- Dimension breakdown
- Constraints triggered (if any)
- Contradiction count

---

## 7) Files of Interest

- API: `src/api.py`
- Orchestration: `src/agents/master_agent.py`
- Reasoning/Scoring: `src/agents/reasoning_agent.py`
- Evidence Agents: `src/agents/*.py`
- UI (React): `ui-react/src/*`

---

## 8) Notes & Assumptions

- Some connectors use mock data where real APIs are not configured.
- LLM usage depends on environment variables (e.g., GROQ/OpenAI keys).
- UI currently renders URLs as plain text (not clickable). This does **not** affect API correctness, but affects UX.
