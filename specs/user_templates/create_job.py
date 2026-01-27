#!/usr/bin/env python3
"""Job creation script for CodeEvolver GEPA optimization.

Submits an optimization job to the CodeEvolver API and polls for results.

SETUP:
    1. Update JOB_CONFIG below with your repository details
    2. Set CODEEVOLVER_API_KEY environment variable
    3. Run: python scripts/create_job.py

WHAT YOU NEED IN YOUR REPO:
    - A DSPy module class (program)
    - A metric function (metric) that scores predictions
    - A training dataset (trainset_path)
    - A program.json from dspy program.save() (optional)

CodeEvolver handles the rest: loading your module, applying prompt
mutations, running evaluation, and optimizing via GEPA.
"""

import os
import sys
import json
import time
import requests

# =============================================================================
# CONFIGURATION - Update these for your project
# =============================================================================

CODEEVOLVER_API_URL = "https://julianghadially--codeevolver-fastapi-app-dev.modal.run"
REPO_URL = "https://github.com/your-org/your-repo"

JOB_CONFIG = {
    # Required - Repository
    "repo_url": REPO_URL,

    # Required - DSPy program class (dotted import path)
    "program": "src.your_module.YourPipeline",

    # Required - Metric function (dotted import path, last component is function name)
    "metric": "eval.evaluate.metric",

    # Required - Training data (path to file in your repo)
    "trainset_path": "data/train.jsonl",

    # Optional - Validation set (defaults to trainset if not provided)
    # "valset_path": "data/val.jsonl",

    # Optional - Saved DSPy program state (from program.save())
    # "saved_program_json_path": "program.json",

    # Optional - Field names that are inputs (vs. labels)
    # If not set, all fields are treated as both inputs and labels
    # "input_keys": ["statement"],

    # Optimization configuration
    "reflection_lm": "openai/gpt-5-mini",
    "max_metric_calls": 1000,
    "num_threads": 1,
    "seed": 0,
}


# =============================================================================
# JOB CREATION AND MANAGEMENT
# =============================================================================

def create_job(api_key: str = None, config_override: dict = None) -> dict:
    """Create optimization job via CodeEvolver API.

    Args:
        api_key: CodeEvolver API key (or set CODEEVOLVER_API_KEY env var)
        config_override: Optional dict to override JOB_CONFIG values

    Returns:
        API response with job_id, status, etc.
    """
    api_key = api_key or os.environ.get("CODEEVOLVER_API_KEY")
    if not api_key:
        print("Error: CODEEVOLVER_API_KEY not set")
        sys.exit(1)

    # Merge config with overrides
    config = {**JOB_CONFIG}
    if config_override:
        config.update(config_override)

    print(f"Creating optimization job...")
    print(f"  Repository: {config['repo_url']}")
    print(f"  Program: {config['program']}")
    print(f"  Metric: {config['metric']}")
    print(f"  Train data: {config.get('trainset_path', 'inline')}")
    print(f"  Max metric calls: {config.get('max_metric_calls', 1000)}")

    response = requests.post(
        f"{CODEEVOLVER_API_URL}/optimize",
        json=config,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def poll_job(job_id: str, api_key: str = None, interval: int = 30) -> dict:
    """Poll job status until completion.

    Args:
        job_id: Job ID from create_job response
        api_key: CodeEvolver API key
        interval: Seconds between polls

    Returns:
        Final job status dict
    """
    api_key = api_key or os.environ.get("CODEEVOLVER_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    print(f"\nPolling job {job_id}...")
    while True:
        response = requests.get(
            f"{CODEEVOLVER_API_URL}/job/{job_id}",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        status = response.json()

        state = status["status"]
        iteration = status.get("current_iteration", "?")
        best_score = status.get("best_score")

        score_str = f"  Best score: {best_score:.4f}" if best_score is not None else ""
        print(f"  [{state}] Iteration {iteration}{score_str}")

        if state in ("completed", "failed", "cancelled"):
            return status

        time.sleep(interval)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    result = create_job()
    print(f"\nJob created: {result['job_id']}")

    final = poll_job(result["job_id"])
    print(f"\nFinal status: {final['status']}")

    if final["status"] == "completed":
        print(f"Best score: {final.get('best_score')}")
        print(f"Best candidate: {json.dumps(final.get('best_candidate'), indent=2)}")
    elif final.get("error"):
        print(f"Error: {final['error']}")
