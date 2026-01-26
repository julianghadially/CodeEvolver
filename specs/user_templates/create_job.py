#!/usr/bin/env python3
"""Create a CodeEvolver GEPA optimization job.

USAGE:
    python scripts/create_job.py

REQUIRED FILES IN YOUR REPO:
    - eval/evaluate.py  (evaluation script, see template)
    - data/train.json   (training dataset)

ENVIRONMENT:
    CODEEVOLVER_API_KEY - Your API key (or pass directly)
"""

import os
import requests

# =============================================================================
# CONFIGURATION - Customize these for your project
# =============================================================================

CODEEVOLVER_API_URL = "https://codeevolver.modal.run"

JOB_CONFIG = {
    # Required
    "repo_url": "https://github.com/YOUR_ORG/YOUR_REPO",
    "trainset_path": "data/train.json",
    "eval_script": "eval/evaluate.py",
    
    # Optional
    "valset_path": "data/val.json",  # or None
    "config": {
        "max_iterations": 100,
        "reflection_lm": "openai/gpt-5-mini",
    }
}

# =============================================================================
# JOB CREATION
# =============================================================================

def create_job(api_key: str = None) -> dict:
    """Create optimization job via CodeEvolver API."""
    api_key = api_key or os.environ.get("CODEEVOLVER_API_KEY")
    if not api_key:
        raise ValueError("CODEEVOLVER_API_KEY not set")
    
    response = requests.post(
        f"{CODEEVOLVER_API_URL}/optimize",
        json=JOB_CONFIG,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def get_job_status(job_id: str, api_key: str = None) -> dict:
    """Check job status."""
    api_key = api_key or os.environ.get("CODEEVOLVER_API_KEY")
    
    response = requests.get(
        f"{CODEEVOLVER_API_URL}/job/{job_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    job = create_job()
    print(f"Job created: {job['job_id']}")
    print(f"Status: {job['status']}")
