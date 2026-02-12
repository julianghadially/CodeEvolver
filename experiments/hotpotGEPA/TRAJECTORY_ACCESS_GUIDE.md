# Accessing GEPA Trajectories and Full Program State

## Overview

Your GEPA optimization ran successfully! This guide explains how to access the complete optimization history, including all candidates, trajectories, and evolved prompts.

## Key Changes Made

### 1. **Extended Database Schema** (`src/schemas/db_schemas.py`)
Added fields to `JobRecord` to store:
- `gepa_state`: Complete GEPA state as JSON
- `program_candidates`: All program candidates evaluated
- `candidate_scores`: Validation scores for each candidate
- `parent_programs`: Parent relationships (genealogy)

### 2. **Updated Callback System** (`src/optimizer/callback.py`)
Modified `CallbackJobUpdater.set_completed()` to accept and serialize the GEPA state with:
- All program candidates
- Validation scores
- Parent program relationships
- Iteration metadata

### 3. **Updated Optimizer** (`src/optimizer/optimizer.py`)
Modified `run_gepa_optimization()` to:
- Extract full state from `GEPAResult`
- Create a `FinalGEPAState` object with all optimization data
- Pass it to the callback for persistence

### 4. **New API Endpoint** (`src/main.py`)
Created `GET /job/{job_id}/state` endpoint that returns:
- All program candidates from the optimization
- Validation scores for each candidate
- Parent relationships between candidates
- Best candidate identification
- Complete iteration history

### 5. **New Response Schema** (`src/schemas/responses.py`)
Added `JobDetailedStateResponse` with fields for accessing the complete optimization state.

## Understanding Your Results

### Why the Best Candidate Has Simple Prompts

Looking at your best candidate:
```json
{
  "program.generate_answer": "Answer questions with a short factoid answer.",
  "program.create_query_hop2": "Given the fields `question`, `summary_1`, produce the fields `query`.",
  ...
}
```

This is likely from an **early iteration** (possibly even the seed candidate). GEPA maintains a **Pareto frontier** of candidates and selects based on validation set performance. Your logs show:

- **Iteration 25**: Detailed prompt for `program.generate_answer` (6 paragraphs)
- **Iteration 26**: Comprehensive prompt for `program.create_query_hop2`
- **13 code mutations**: Major structural changes

The best overall performance may have come from:
1. **Code changes** providing the biggest improvements
2. An **early prompt configuration** that worked well with the evolved code
3. The **combination** of simple prompts + sophisticated code performing better than complex prompts

## How to Access the Data

### Option 1: Using the New API Endpoint

```python
import httpx

job_id = "your-job-id"
response = httpx.get(f"https://your-api.modal.run/job/{job_id}/state")
state = response.json()

# Access all candidates
all_candidates = state["program_candidates"]
print(f"Total candidates evaluated: {len(all_candidates)}")

# See each candidate's prompts
for idx, candidate in enumerate(all_candidates):
    score = state["candidate_scores"][idx]
    print(f"\n=== Candidate {idx} (score: {score:.3f}) ===")

    # Show prompts (exclude _code)
    for key, value in candidate.items():
        if key != "_code":
            print(f"{key}: {value[:100]}...")

    # Show code changes if present
    if "_code" in candidate:
        import json
        code_data = json.loads(candidate["_code"])
        print(f"Branch: {code_data.get('git_branch')}")
        print(f"Change: {code_data.get('change_request', 'N/A')[:100]}...")

# Find best candidate
best_idx = state["best_idx"]
best_candidate = state["best_candidate"]
best_score = state["best_score"]
print(f"\n=== Best Candidate (idx: {best_idx}, score: {best_score:.3f}) ===")
for key, value in best_candidate.items():
    if key != "_code":
        print(f"{key}: {value}")

# Analyze parent relationships
parent_programs = state["parent_programs"]
print(f"\n=== Genealogy ===")
for idx, parents in enumerate(parent_programs):
    if parents:
        print(f"Candidate {idx} <- Parents: {parents}")
```

### Option 2: Querying MongoDB Directly

```python
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio

async def get_job_state(job_id: str):
    client = AsyncIOMotorClient("your-mongodb-url")
    db = client.codeevolver

    job = await db.jobs.find_one({"job_id": job_id})

    # Access GEPA state
    gepa_state = job.get("gepa_state", {})
    program_candidates = job.get("program_candidates", [])
    candidate_scores = job.get("candidate_scores", [])

    print(f"Total candidates: {len(program_candidates)}")
    print(f"Best score: {job.get('best_score')}")

    return gepa_state

# Run it
asyncio.run(get_job_state("your-job-id"))
```

### Option 3: Check Git Branches

Each candidate created a git branch. To see the evolved code:

```bash
# List all branches from your run
git branch -r | grep "codeevolver-20260212001600"

# Checkout a specific candidate's branch
git checkout codeevolver-20260212001600-4c7d53

# View the program changes
git diff main langProPlus/hotpotGEPA/hotpot_program.py

# Check the stored program JSON
cat codeevolver/results/best_program_20260212001600.json
```

## Analyzing the Optimization Run

### 1. **Find the Iteration Where Best Prompts Appeared**

```python
# Look for candidates with detailed prompts
for idx, candidate in enumerate(all_candidates):
    generate_answer_prompt = candidate.get("program.generate_answer", "")
    if len(generate_answer_prompt) > 200:  # Detailed prompt
        print(f"Candidate {idx}: Detailed prompt (score: {candidate_scores[idx]:.3f})")
        print(generate_answer_prompt[:200])
```

### 2. **Compare Code vs Prompt Mutations**

```python
import json

code_mutations = []
prompt_mutations = []

for idx, candidate in enumerate(all_candidates):
    if "_code" in candidate:
        code_data = json.loads(candidate["_code"])
        if code_data.get("change_request"):
            code_mutations.append({
                "idx": idx,
                "score": candidate_scores[idx],
                "change": code_data["change_request"]
            })
        else:
            prompt_mutations.append({
                "idx": idx,
                "score": candidate_scores[idx]
            })

print(f"Code mutations: {len(code_mutations)}")
print(f"Prompt mutations: {len(prompt_mutations)}")
print(f"Best score from code mutation: {max(m['score'] for m in code_mutations):.3f}")
print(f"Best score from prompt mutation: {max(m['score'] for m in prompt_mutations):.3f}")
```

### 3. **Build the Evolution Tree**

```python
def print_evolution_tree(candidates, scores, parents, idx=0, indent=0):
    """Recursively print the evolution tree."""
    score = scores[idx]
    print("  " * indent + f"Candidate {idx} (score: {score:.3f})")

    # Find children
    children = [i for i, p in enumerate(parents) if p and idx in p]
    for child_idx in children:
        print_evolution_tree(candidates, scores, parents, child_idx, indent + 1)

# Find root candidates (no parents)
roots = [i for i, p in enumerate(parent_programs) if not p]
print("Evolution Trees:")
for root in roots:
    print_evolution_tree(all_candidates, candidate_scores, parent_programs, root)
```

## Next Steps

1. **Deploy the changes**:
   ```bash
   modal deploy modal_app.py
   ```

2. **Run a new optimization** to verify the changes work

3. **Query your completed job**:
   ```python
   response = httpx.get(f"https://your-api.modal.run/job/{job_id}/state")
   state = response.json()
   ```

4. **Analyze the results** using the scripts above

## Trajectory Data (Future Enhancement)

Currently, we're storing:
- ✅ All program candidates
- ✅ Validation scores
- ✅ Parent relationships
- ❌ Individual execution traces (trajectories)

To add execution traces, you would need to:
1. Store `eval_batch.trajectories` in the GEPA state
2. Add a `trajectories` field to the database schema
3. Create an endpoint to retrieve specific trajectory data (can be large!)

## Understanding Your Specific Run

Based on your logs and results:

1. **Code Changes Dominated**: 13 code mutations vs many prompt mutations
2. **Best Performance**: Simple prompts + evolved code architecture
3. **Key Innovation**: Direct passage-to-answer extraction (removed summarization)
4. **Effective Changes**:
   - Removed `summarize1` and `summarize2` modules
   - Changed from `dspy.Predict` to `dspy.ChainOfThought`
   - Modified answer generation to extraction

The winning candidate likely combined:
- **Early simple prompts** (seed or near-seed)
- **Late-stage code architecture** from the 13 mutations

This is actually a **success story** for GEPA - it found that structural code changes mattered more than prompt engineering for this task!
