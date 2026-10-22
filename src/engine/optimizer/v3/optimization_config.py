"""Copyright © 2026 440 Labs LLC

Canonical schema for a single optimization run's configuration.

This dataclass is the one definition of the optimization config. The config JSON written to the
sandbox (`<ws>/.codeevolver/optimization_config.json`) *is* this object —
`to_dict()` / `from_dict()` are the wire (de)serialization.

The wire format is a **flat** JSON dict (the six job/auth fields live at the
top level, not nested) so existing readers and inline test configs keep
working; the nested `JobContext` is reconstructed on `from_dict`.

Naming: the eval-row budget knob is `max_metric_calls` end-to-end. `from_dict`
also reads the legacy `max_eval_rows` key as a fallback so a config never
silently loses its budget.

Environment-agnostic — no DB or cloud SDK imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.engine.services.job_context import JobContext

DEFAULT_REFLECTION_LM = "anthropic/claude-opus-4-7"


@dataclass
class OptimizationConfig:
    """All inputs for one optimization run (the config-JSON contract)."""

    # ---- required ----
    program: str
    metric: str

    # ---- workspace ----
    workspace: str = "/workspace"

    # ---- datasets ----
    trainset_json: list | None = None
    trainset_path: str | None = None
    testset_json: list | None = None
    testset_path: str | None = None
    # Fixed comparison set (engine 3): drawn from the trainset with seed 42,
    # used by the architect for apples-to-apples branch comparison. Counted
    # against the eval-row budget.
    comparison_set_size: int | None = None
    # Comparison-credit ledger (specs/archive-public/comparison_credits.md): comparison-set
    # runs are paced by credits — `comparison_credits_initial` granted before
    # the seed hook, `comparison_credits_per_iteration` more at each iteration
    # start. Agent-initiated comparison evals are refused at balance <= 0;
    # loop-initiated runs debit but may overdraw. `None` per-iteration rate
    # disables the ledger (and its acknowledgement step) entirely.
    comparison_credits_initial: int = 1
    comparison_credits_per_iteration: int | None = 1

    # ---- metrics / evaluation ----
    metric_names: list[str] | None = None
    primary_metric: str | None = None
    input_keys: list[str] | None = None
    num_threads: int = 1
    max_score: float = 1.0

    # ---- models / harness ----
    reflection_lm: str = DEFAULT_REFLECTION_LM
    coding_harness: str | None = None
    coding_harness_model: str | None = None

    # ---- budget / timeouts / determinism ----
    seed: int = 0
    subsample_size: int = 20
    max_metric_calls: int | None = None
    job_timeout_seconds: int | None = None
    # Backstop (seconds) for one comparison-set or finalization test-set eval.
    comparison_eval_timeout: int = 14400
    # Finalization draws the held-out test set this many times for the FINAL
    # primary (uncounted against the eval budget). Every reading is reported
    # (`final_test_scores`); `final_test_score` is the mean.
    test_samples: int = 3
    # Same for the SEED. Defaults to 1: every arm in the research matrix
    # re-measures the same seed, so its readings average across runs and
    # extra draws per run buy nothing.
    seed_test_samples: int = 1
    # `run_variance_estimate` (specs/archive-public/sampling_variance.md): how many times the
    # architect's once-per-run noise estimate evaluates the program on its
    # fixed sample. Internal knob — the agent never chooses it. Counted
    # against the budget (5 x 50 = 250 rows = 5% of a 5000-row research
    # budget); e2e tests set 2 to stay affordable.
    variance_estimate_samples: int = 5
    # Hard backstop (seconds) on a single architect subsample / single-row eval
    # subprocess. Generous by design (2h) — it exists only to kill a genuinely
    # hung client program, not to bound a legitimately slow eval (programs can
    # start ~1h and grow to several hours as reasoning steps are added).
    subsample_eval_timeout: int = 7200
    debug_max_iterations: int | None = None

    # ---- asa (specs/archive-public/v3_asa.md) ----
    # The constant multiplied against the current gap to the maximum expected
    # score; sets how sensitive ASA is to declaring a non-primary lineage
    # ELIGIBLE (worth forced investment). Max expected 1.0, primary at 0.90
    # -> gap 0.1 -> threshold 0.1 * 0.1 = 0.01 (mean per-row positive
    # contribution the lineage must have over the primary).
    asa_behavioral_contribution_constant: float = 0.10
    # Upper bound on that threshold (score units, per row).
    asa_behavioral_contribution_threshold_cap: float = 0.05
    # The primary's comparison record is refreshed by the loop once it is
    # older than this many LANDED iterations on the primary branch.
    asa_primary_stale_landed_iterations: int = 4

    # ---- behavior flags ----
    additional_instructions: str | None = None
    enable_meta_reflection: bool = False
    # Post-run run-integrity inspector (advisory; verdict stamped on the
    # latest iteration record). Never affects run success.
    enable_run_integrity_agent: bool = False
    # Start-of-run clean-slate inspector (seed step). Opt-in: a "contaminated"
    # verdict ABORTS the run, so research configs enable it explicitly.
    enable_clean_slate_inspector: bool = False
    # Extra path fragments (substring-matched) the coding / inspection /
    # reflection agents must never read or edit. Appended to CodeEvolver's
    # built-in holdout fences (test set, etc.); never a replacement.
    additional_deny_paths: list[str] | None = None

    # ---- git / callback auth (flattened on the wire) ----
    job: JobContext = field(default_factory=JobContext)

    def __post_init__(self) -> None:
        for name in ("test_samples", "seed_test_samples"):
            value = getattr(self, name)
            if int(value) < 1:
                raise ValueError(f"{name} must be >= 1, got {value!r}")
            setattr(self, name, int(value))
        if int(self.variance_estimate_samples) < 2:
            raise ValueError(
                "variance_estimate_samples must be >= 2 (a variance needs two "
                f"readings), got {self.variance_estimate_samples!r}"
            )
        self.variance_estimate_samples = int(self.variance_estimate_samples)
        if int(self.comparison_credits_initial) < 0:
            raise ValueError(
                f"comparison_credits_initial must be >= 0, got {self.comparison_credits_initial!r}"
            )
        self.comparison_credits_initial = int(self.comparison_credits_initial)
        if self.comparison_credits_per_iteration is not None:
            if int(self.comparison_credits_per_iteration) < 0:
                raise ValueError(
                    "comparison_credits_per_iteration must be >= 0 or None, got "
                    f"{self.comparison_credits_per_iteration!r}"
                )
            self.comparison_credits_per_iteration = int(self.comparison_credits_per_iteration)

    @classmethod
    def from_dict(cls, d: dict) -> "OptimizationConfig":
        """Build from the flat config-JSON dict. Tolerant of missing keys
        (applies the entry point's defaults) and ignores
        unknown keys — except the removed valset keys, which are rejected
        loudly so a stale caller cannot silently run without its valset."""
        removed = [k for k in ("valset_json", "valset_path", "valset_bar", "max_valset_size") if d.get(k) is not None]
        if removed:
            raise ValueError(
                f"Engine 3 removed the validation set; config keys {removed} are no longer "
                "supported. Fold valset rows into the trainset and use `comparison_set_size` "
                "(replaces max_valset_size) for the fixed comparison set."
            )
        return cls(
            program=d["program"],
            metric=d["metric"],
            workspace=d.get("workspace", "/workspace"),
            trainset_json=d.get("trainset_json"),
            trainset_path=d.get("trainset_path"),
            testset_json=d.get("testset_json"),
            testset_path=d.get("testset_path"),
            comparison_set_size=d.get("comparison_set_size"),
            comparison_credits_initial=d.get("comparison_credits_initial", 1),
            comparison_credits_per_iteration=d.get("comparison_credits_per_iteration", 1),
            metric_names=d.get("metric_names"),
            primary_metric=d.get("primary_metric"),
            input_keys=d.get("input_keys"),
            num_threads=d.get("num_threads", 1),
            max_score=d.get("max_score", 1.0),
            reflection_lm=d.get("reflection_lm", DEFAULT_REFLECTION_LM),
            coding_harness=d.get("coding_harness"),
            coding_harness_model=d.get("coding_harness_model"),
            seed=d.get("seed", 0),
            subsample_size=d.get("subsample_size", 20),
            # Standardized key is `max_metric_calls`; fall back to the legacy
            # `max_eval_rows` so an in-flight config keeps its budget.
            max_metric_calls=d.get("max_metric_calls", d.get("max_eval_rows")),
            job_timeout_seconds=d.get("job_timeout_seconds"),
            comparison_eval_timeout=d.get("comparison_eval_timeout", 14400),
            test_samples=d.get("test_samples", 3),
            seed_test_samples=d.get("seed_test_samples", 1),
            variance_estimate_samples=d.get("variance_estimate_samples", 5),
            subsample_eval_timeout=d.get("subsample_eval_timeout", 7200),
            debug_max_iterations=d.get("debug_max_iterations"),
            asa_behavioral_contribution_constant=d.get("asa_behavioral_contribution_constant", 0.10),
            asa_behavioral_contribution_threshold_cap=d.get("asa_behavioral_contribution_threshold_cap", 0.05),
            asa_primary_stale_landed_iterations=d.get("asa_primary_stale_landed_iterations", 4),
            additional_instructions=d.get("additional_instructions"),
            enable_meta_reflection=d.get("enable_meta_reflection", False),
            enable_run_integrity_agent=d.get("enable_run_integrity_agent", False),
            enable_clean_slate_inspector=d.get("enable_clean_slate_inspector", False),
            additional_deny_paths=d.get("additional_deny_paths"),
            job=JobContext.from_config(d),
        )

    def to_dict(self) -> dict:
        """Serialize to the flat config-JSON dict (job fields hoisted to top
        level)."""
        return {
            "workspace": self.workspace,
            "program": self.program,
            "metric": self.metric,
            "trainset_json": self.trainset_json,
            "trainset_path": self.trainset_path,
            "testset_json": self.testset_json,
            "testset_path": self.testset_path,
            "reflection_lm": self.reflection_lm,
            "coding_harness": self.coding_harness,
            "coding_harness_model": self.coding_harness_model,
            "num_threads": self.num_threads,
            "seed": self.seed,
            "additional_instructions": self.additional_instructions,
            "comparison_set_size": self.comparison_set_size,
            "comparison_credits_initial": self.comparison_credits_initial,
            "comparison_credits_per_iteration": self.comparison_credits_per_iteration,
            "debug_max_iterations": self.debug_max_iterations,
            "asa_behavioral_contribution_constant": self.asa_behavioral_contribution_constant,
            "asa_behavioral_contribution_threshold_cap": self.asa_behavioral_contribution_threshold_cap,
            "asa_primary_stale_landed_iterations": self.asa_primary_stale_landed_iterations,
            "max_metric_calls": self.max_metric_calls,
            "job_timeout_seconds": self.job_timeout_seconds,
            "comparison_eval_timeout": self.comparison_eval_timeout,
            "test_samples": self.test_samples,
            "seed_test_samples": self.seed_test_samples,
            "variance_estimate_samples": self.variance_estimate_samples,
            "subsample_eval_timeout": self.subsample_eval_timeout,
            "subsample_size": self.subsample_size,
            "metric_names": self.metric_names,
            "primary_metric": self.primary_metric,
            "max_score": self.max_score,
            "input_keys": self.input_keys,
            "enable_meta_reflection": self.enable_meta_reflection,
            "enable_run_integrity_agent": self.enable_run_integrity_agent,
            "enable_clean_slate_inspector": self.enable_clean_slate_inspector,
            "additional_deny_paths": self.additional_deny_paths,
            # Job/auth fields, flattened to the top level.
            "repo_url": self.job.repo_url,
            "github_token": self.job.github_token,
            "callback_url": self.job.callback_url,
            "jwt_token": self.job.jwt_token,
            "job_id": self.job.job_id,
            "initial_branch": self.job.initial_branch,
        }


__all__ = ["OptimizationConfig", "DEFAULT_REFLECTION_LM"]
