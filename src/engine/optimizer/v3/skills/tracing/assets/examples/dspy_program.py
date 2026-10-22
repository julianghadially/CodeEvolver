"""Example: a DSPy program with auto-instrumented LM calls plus
manually instrumented deterministic glue.

Setup
-----

    requirements.txt:
        dspy
        openinference-instrumentation-dspy
        opentelemetry-api

    your_pkg/__init__.py:
        from openinference.instrumentation.dspy import DSPyInstrumentor

        DSPyInstrumentor().instrument()  # call once, at import time

After this, every DSPy `Predict` / `ChainOfThought` / etc. call emits
OTel spans automatically. You only need `@traceable` on the surrounding
deterministic code.
"""

import dspy

from your_pkg._tracing import traceable


class Classify(dspy.Signature):
    """Classify whether a claim is true or false."""

    statement: str = dspy.InputField()
    verdict: str = dspy.OutputField(desc="'true' or 'false'")


class FactChecker(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(Classify)

    def forward(self, statement: str) -> dspy.Prediction:
        # Pre-processing: deterministic, worth a span
        normalized = self._normalize(statement)
        # LM call: auto-instrumented by DSPyInstrumentor — no @traceable needed
        result = self.classify(statement=normalized)
        # Post-processing: deterministic, worth a span
        verdict = self._post_process(result.verdict)
        return dspy.Prediction(verdict=verdict)

    @traceable("tool")
    def _normalize(self, statement: str) -> str:
        return statement.strip().lower()

    @traceable("tool")
    def _post_process(self, raw_verdict: str) -> str:
        return "true" if "true" in raw_verdict.lower() else "false"


# Trace shape per row (after DSPy filter drops the structural wrappers):
#
# trace = [
#   {"signature_key": "pipeline", "span_kind": "chain", ...},                 # eval root
#   {"signature_key": "FactChecker._normalize", "span_kind": "tool", ...},
#   {"signature_key": "LM.__call__", "span_kind": "llm", ...},                # auto from OpenInference
#   {"signature_key": "FactChecker._post_process", "span_kind": "tool", ...},
# ]
#
# Compare with what you'd get WITHOUT the DSPy filter — every
# `Predict.forward`, `ChainOfThought.forward`, `ChatAdapter.__call__`
# duplicates the same prompt up the stack. The filter at
# `src/engine/telemetry/filters/ai_frameworks/dspy.py` keeps only the
# `LM.__call__` leaf, so the architect sees one prompt, not five.
#
# Tip: instrument `forward()` itself with `@traceable("chain")` only if
# you want a wrapper span around the whole module. Most DSPy users skip
# this — the eval's root "pipeline" span already covers it.
