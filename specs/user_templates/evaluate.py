"""Example metric function for CodeEvolver GEPA optimization.

CONTRACT:
    Your metric function receives a dspy.Example (ground truth) and a
    dspy.Prediction (model output), and returns a float score (0.0–1.0).

    CodeEvolver handles everything else: loading your DSPy module,
    applying candidate prompts, running examples, and collecting scores.

USAGE:
    In your create_job.py, reference this function with a dotted import path:

        "metric": "eval.evaluate.metric"

    This means: from eval.evaluate, import the function named 'metric'.
"""

import dspy


def metric(example: dspy.Example, prediction: dspy.Prediction) -> float:
    """Score a single prediction against ground truth.

    Args:
        example: Ground truth with all fields (inputs + labels).
        prediction: Model output from your DSPy program's forward().

    Returns:
        Score between 0.0 and 1.0.
    """
    # TODO: Implement your scoring logic
    # Example: exact match on a 'label' field
    # return float(prediction.label == example.label)
    raise NotImplementedError("Implement your metric function")
