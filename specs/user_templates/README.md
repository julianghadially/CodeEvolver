# CodeEvolver User Templates

Templates for integrating your repository with CodeEvolver GEPA optimization.

## Required Files

```
your-repo/
├── eval/
│   └── evaluate.py     # Evaluation script (from template)
├── data/
│   └── train.json      # Training dataset
│   └── test.json      # Testing dataset (optional)
└── scripts/
    └── create_job.py   # Job creation script (from template)
```

## Setup

1. Copy `evaluate.py` to `eval/evaluate.py`
2. Implement `load_program()` and `run_and_score()`
3. Copy `create_job.py` to `scripts/create_job.py`
4. Update `JOB_CONFIG` with your repo details
5. Set `CODEEVOLVER_API_KEY` environment variable
6. Run `python scripts/create_job.py`

## Dataset Format

```json
[
    {"statement": "Your input text", "label": "expected_output"},
    ...
]
```

Adjust field names in `evaluate.py` to match your dataset.
