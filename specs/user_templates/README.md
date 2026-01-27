# CodeEvolver User Templates

Templates for integrating your repository with CodeEvolver GEPA optimization.

## What You Need

```
your-repo/
├── src/
│   └── your_module.py     # DSPy module with forward() method
├── eval/
│   └── evaluate.py        # Metric function (from template)
├── data/
│   └── train.jsonl         # Training dataset (.json, .jsonl, or .csv)
├── program.json            # Optional: saved DSPy program state
└── scripts/
    └── create_job.py       # Job creation script (from template)
```

## Setup

1. Copy `evaluate.py` to `eval/evaluate.py` and implement your metric function
2. Copy `create_job.py` to `scripts/create_job.py`
3. Update `JOB_CONFIG` with your repo details:
   - `program`: dotted path to your DSPy module class
   - `metric`: dotted path to your metric function
   - `trainset_path`: path to training data in your repo
4. Set `CODEEVOLVER_API_KEY` environment variable
5. Run `python scripts/create_job.py`

## How It Works

CodeEvolver handles everything automatically:
1. Clones your repo
2. Imports your DSPy module from `program`
3. Loads your metric from `metric`
4. Reads training data from `trainset_path`
5. Runs GEPA optimization: mutates prompts, evaluates, and selects the best

**You only write two things:** your DSPy module and a metric function.

## Metric Function

Your metric scores a single prediction against ground truth:

```python
def metric(example: dspy.Example, prediction: dspy.Prediction) -> float:
    return float(prediction.label == example.label)
```

Reference it in `create_job.py` using a dotted import path:
- `"eval.evaluate.metric"` → imports `metric` from `eval.evaluate`

This uses the same format as `program`.

## Dataset Format

Any of `.json`, `.jsonl`, or `.csv`:

```json
[
    {"statement": "Your input text", "label": "expected_output"},
    ...
]
```
