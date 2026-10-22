"""Example: instrumenting an async LM call.

Before
------

    import httpx

    async def call_judge(prompt: str, model: str = "gpt-4o") -> str:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.example.com/v1/chat",
                json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            )
            return r.json()["choices"][0]["message"]["content"]

After
-----
"""

import httpx

from your_pkg._tracing import traceable


@traceable("llm")
async def call_judge(prompt: str, model: str = "gpt-4o") -> str:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.example.com/v1/chat",
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        )
        return r.json()["choices"][0]["message"]["content"]


# Trace entry shape:
#
# {
#   "signature_key": "call_judge",
#   "span_kind": "llm",
#   "parent_key": "pipeline",
#   "inputs": {"prompt": "Classify whether this claim is true...", "model": "gpt-4o"},
#   "output": "False"
# }
#
# Notes:
# - `@traceable` detects async functions via `inspect.iscoroutinefunction`
#   and uses an async wrapper automatically. No separate decorator.
# - If the LM client library is already auto-instrumented by an
#   OpenInference instrumentor, you may double up: a decorated wrapper
#   span PLUS the instrumentor's inner span. That is fine — the architect
#   reads both, and the wrapper span gives you a clean place to record
#   prompt / model as `ce.inputs.*`.
# - The full prompt is stored by default — the architect is expected to
#   grep / parse it itself. Pass `max_attr_chars=N` to opt into truncation
#   at a specific call site (e.g. when the function handles secrets):
#
#       @traceable("llm", max_attr_chars=2000)
#       async def call_with_credentials(prompt: str, api_key: str) -> str: ...
