"""Example: instrumenting a plain Python function.

Before
------

    def parse_claim(text: str) -> list[str]:
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        return sentences

After (drop `traceable.py` into your_pkg/_tracing.py)
-----------------------------------------------------
"""

from your_pkg._tracing import traceable


@traceable("tool")
def parse_claim(text: str) -> list[str]:
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    return sentences


# What the trace entry looks like for `parse_claim("A. B.")`:
#
# {
#   "signature_key": "parse_claim",
#   "span_kind": "tool",
#   "parent_key": "<whatever called this>",  # e.g. "pipeline"
#   "inputs": {"text": "A. B."},
#   "output": "['A', 'B']"
# }
#
# Notes:
# - `inputs` keys come from the function signature, not the call site —
#   positional args and kwargs both get bound to their parameter names.
# - `output` is `str(result)`, truncated to 2000 chars by default. If you
#   need structured output, `json.dumps()` it inside the function or pass
#   `max_attr_chars=...` to give it more room.
# - If the function raises, `ce.error` gets `"<ExcType>: <message>"` and
#   the exception still propagates to the caller.
