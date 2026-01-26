#!/usr/bin/env python3
"""Evaluation script for CodeEvolver GEPA optimization.

CONTRACT:
    python eval/evaluate.py \
        --candidate /tmp/candidate.json \
        --batch /tmp/batch.json \
        --output /tmp/results.json

INPUT FILES (written by CodeEvolver):
    candidate.json: {"predictor.predict": "instruction text", ...}
    batch.json: [{"statement": "...", "label": "..."}, ...]

OUTPUT FILE (written by your script):
    results.json: {"scores": [1.0, 0.0, ...], "outputs": [...]}
"""

import argparse
import json

# =============================================================================
# YOUR IMPORTS - Add your project imports here
# =============================================================================

# from src.your_module import YourPipeline
# from src.your_metrics import compute_accuracy


# =============================================================================
# IMPLEMENT THESE FUNCTIONS
# =============================================================================

def load_program(candidate: dict):
    """Load your program and apply candidate prompt configurations.
    
    Args:
        candidate: {"predictor.predict": "instruction text", ...}
    
    Returns:
        Your initialized program ready to run.
    """
    # TODO: Load your DSPy pipeline
    # TODO: Apply candidate prompts to predictors
    raise NotImplementedError("Implement load_program()")


def run_and_score(program, example: dict) -> tuple[dict, float]:
    """Run program on one example and compute score.
    
    Args:
        program: Your initialized program
        example: Single example from batch (e.g., {"statement": "...", "label": "..."})
    
    Returns:
        (output_dict, score) where score is 0.0-1.0
    """
    # TODO: Run your program on the example
    # TODO: Compute score against ground truth
    raise NotImplementedError("Implement run_and_score()")


# =============================================================================
# MAIN - No changes needed below
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    with open(args.candidate) as f:
        candidate = json.load(f)
    with open(args.batch) as f:
        batch = json.load(f)
    
    program = load_program(candidate)
    
    scores, outputs = [], []
    for example in batch:
        try:
            output, score = run_and_score(program, example)
            outputs.append(output)
            scores.append(score)
        except Exception as e:
            outputs.append({"error": str(e)})
            scores.append(0.0)
    
    with open(args.output, "w") as f:
        json.dump({"scores": scores, "outputs": outputs}, f)


if __name__ == "__main__":
    main()
