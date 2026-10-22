# Copyright © 2026 440 Labs LLC
"""Render every predictor's formatted messages for a repo checkout (seed-parity check)."""
import importlib, json, sys
repo, modpath, clsname = sys.argv[1:4]
sys.path.insert(0, repo)
import dspy
mod = importlib.import_module(modpath)
prog = getattr(mod, clsname)()
adapter = dspy.ChatAdapter()
out = {}
for name, pred in prog.named_predictors():
    sig = pred.signature
    inputs = {k: f"<{k} value>" for k in sig.input_fields}
    msgs = adapter.format(signature=sig, demos=[], inputs=inputs)
    out[name] = msgs
print(json.dumps(out, indent=1, sort_keys=True))
