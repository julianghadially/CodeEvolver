"""Copyright © 2026 440 Labs LLC

Per-framework span filters.

Each submodule targets one AI framework or telemetry vendor (DSPy,
LangChain, LangFuse, ...). A framework module exposes at minimum a
`should_drop(span) -> bool` predicate that returns True for spans the
framework emits as internal plumbing (e.g., DSPy's
`ChainOfThought.forward` / `Predict.forward` wrappers around the actual
LM call). The top-level `filters.make_keep_predicate` composes these
into a single `should_keep` passed to `setup_tracer`.
"""
