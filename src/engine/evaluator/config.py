"""Copyright © 2026 440 Labs LLC

Evaluator configuration types and parsing helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvaluatorConfig:
    """Configuration for running the generic evaluator.

    The program is resolved from `program_factory`, a dotted path of the form
    `pkg.mod:factory_fn`. The factory is called once per sandbox start and
    must return a callable — plain function, class instance with __call__, or
    framework module. Plain functions (`pkg.mod:func`) are accepted as a
    shorthand and auto-wrapped in a trivial factory at resolution time.

    Each metric is called as:
        metric_fn(output=program_output, example=Example(row))

    `output` is the program's return value (untouched); `example` is the full
    dataset row wrapped as an `Example` (attribute + item access), so the metric
    reads gold fields via `example.<field>` / `example["<field>"]`.
    """

    program_factory: str
    metric_fns: list[str]
    metric_names: list[str]
    primary_metric: str

    # Row-evaluation concurrency. Defaults to 1 (sequential) because client
    # programs are not assumed thread-safe — most are not DSPy. Opt in per call
    # by passing a higher `num_threads`. When > 1 the mounted evaluator runs
    # rows on a ThreadPoolExecutor (clamped to the row count and a hard ceiling
    # of 128); the single resolved program instance and the metric callables are
    # then invoked concurrently, so the client's `__call__`/metrics must be
    # thread-safe — keep per-request state local, not on `self`. Threads help
    # I/O-bound programs (LLM calls release the GIL); CPU-bound clients should
    # leave it at 1.
    num_threads: int = 1
    # The mounted evaluator picks the dataset based on the `suffix` the host
    # sends (`"test"` → testset, otherwise trainset — the comparison set is a
    # fixed subsample of the trainset, so `"comparison"` reads the trainset
    # file with comparison row ids).
    trainset_path: str | None = None
    # Held-out test set. Scored only at finalization (seed + final primary,
    # `test_samples` draws each) and picked by the mounted evaluator on
    # `suffix == "holdout"` / `"holdout_<k>"`. Locked from every
    # agent for the whole run via deny-path hooks. None when no test set was
    # provided.
    testset_path: str | None = None
    # Authoritative list of row fields to pass to the program. When set, the
    # mounted evaluator passes only these row keys to the program call; non-
    # input columns (id, answer, label, …) never leak. Falls back to
    # `inspect.signature` introspection when None.
    input_keys: list[str] | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.metric_fns) != len(self.metric_names):
            raise ValueError(
                f"metric_fns ({len(self.metric_fns)}) and metric_names "
                f"({len(self.metric_names)}) must have the same length"
            )
        if self.primary_metric not in self.metric_names:
            raise ValueError(
                f"primary_metric {self.primary_metric!r} is not in metric_names "
                f"{self.metric_names!r}"
            )
        # Validate factory paths look vaguely sensible — deeper resolution happens
        # in the sandbox where the client's packages are importable.
        parse_factory_path(self.program_factory)
        for fn in self.metric_fns:
            parse_factory_path(fn)

    def to_payload(
        self,
        *,
        row_ids: list[int] | None,
        iteration_idx: int,
        suffix: str,
        capture_traces: bool,
    ) -> dict:
        """Serialize to a JSON payload for the execution layer's `run_evaluation`."""
        # The mounted evaluator receives both dataset paths and picks one
        # based on the suffix. Keeping them in `extra` preserves the
        # existing passthrough contract with the mounted script.
        extra = dict(self.extra or {})
        extra.setdefault("trainset_path", self.trainset_path)
        extra.setdefault("testset_path", self.testset_path)
        return {
            "command": "evaluate",
            "program_factory": self.program_factory,
            "metric_fns": self.metric_fns,
            "metric_names": self.metric_names,
            "primary_metric": self.primary_metric,
            "num_threads": self.num_threads,
            "input_keys": self.input_keys,
            "row_ids": row_ids,
            "iteration_idx": iteration_idx,
            "suffix": suffix,
            "capture_traces": capture_traces,
            "extra": extra,
        }


def parse_factory_path(path: str) -> tuple[str, str]:
    """Parse a dotted factory path.

    Accepts either `pkg.mod:factory_fn` (the canonical form) or the shorthand
    `pkg.mod.factory_fn` (last dot segment interpreted as the attribute).

    Raises ValueError for malformed input.
    """
    if not isinstance(path, str) or not path:
        raise ValueError(f"factory path must be a non-empty string, got {path!r}")

    if ":" in path:
        module, _, attr = path.partition(":")
    else:
        if "." not in path:
            raise ValueError(
                f"factory path {path!r} must be `pkg.mod:fn` or `pkg.mod.fn`"
            )
        module, _, attr = path.rpartition(".")

    if not module or not attr:
        raise ValueError(f"factory path {path!r} is missing module or attribute")
    if not attr.isidentifier():
        raise ValueError(f"factory path {path!r}: {attr!r} is not a valid identifier")
    for segment in module.split("."):
        if not segment.isidentifier():
            raise ValueError(
                f"factory path {path!r}: {segment!r} is not a valid module segment"
            )
    return module, attr
