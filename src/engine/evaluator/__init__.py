"""Copyright © 2026 440 Labs LLC

Generic evaluator package.

This package owns evaluation of a client program against a dataset and a set
of metrics. It is deliberately framework-agnostic: the client supplies a
program factory via a dotted path and a list of metric functions, and the
evaluator runs each row through the program and scores it.

Importantly, nothing in this package imports from src.engine.optimizer. The
optimizer calls into `run_eval()` as a tool; the evaluator never calls back
into the optimizer.
"""

from .config import EvaluatorConfig, parse_factory_path
from .sandbox_client import run_eval

__all__ = ["EvaluatorConfig", "parse_factory_path", "run_eval"]
