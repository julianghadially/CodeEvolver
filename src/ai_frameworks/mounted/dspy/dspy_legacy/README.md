# DSPy Legacy Compatibility

Compatibility with older DSPy versions is not tested and is maintained by the community. This folder contains compatibility modules for DSPy versions prior to 3.0.0.

## Version Threshold

- **DSPy < 3.0.0** (2.6.x and earlier): Uses legacy trace capture via `dspy.context(trace=[])`
- **DSPy >= 3.0.0**: Uses modern trace capture via `bootstrap_trace_data` from `dspy.teleprompt.bootstrap_trace`

## Version Check

The main `evaluate.py` automatically detects the DSPy version and routes to the appropriate implementation:

1. **Version Detection**: Checks `dspy.__version__` at module load time
2. **Routing**: If version < 3.0.0, imports and uses `dspy_legacy/evaluate.py`
3. **Fallback**: If legacy import fails, falls back to simple evaluation without traces

## Legacy Trace Capture (DSPy 2.6.x)

The legacy implementation in `evaluate.py` uses DSPy's native trace capture pattern:

```python
with dspy.context(trace=[]):
    prediction = program(**example.inputs())
    trace = dspy.settings.trace  # Captured traces
```

This is based on the pattern used in [DSPy's bootstrap.py](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/bootstrap.py).

## Modern Trace Capture (DSPy >= 3.0.0)

The modern implementation uses `bootstrap_trace_data` from `dspy.teleprompt.bootstrap_trace` (added in DSPy 3.0.0, August 2025), which provides a higher-level API for trace capture during evaluation.

## Compatibility Matrix

| DSPy Version | Trace Capture Method | Implementation File |
|-------------|---------------------|---------------------|
| 2.5.x | `dspy.context()` | `dspy_legacy/evaluate.py` |
| 2.6.x (including 2.6.13) | `dspy.context()` | `dspy_legacy/evaluate.py` |
| 3.0.0+ | `bootstrap_trace_data` | `evaluate.py` |

DSPy 3.0.0 was released in August 2025.

