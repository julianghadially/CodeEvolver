"""Program execution and mutation related schemas."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src.config import validate_python_sandbox_version


class OptimizeRequest(BaseModel):
    """Request payload for POST /optimize.

    Users must provide either inline `trainset` data or a `trainset_path`
    pointing to a data file in their repository.
    """

    repo_url: str = Field(..., description="Git repository URL")
    program: str = Field(
        ...,
        description="Dotted import path of dspy.module program that will undergo optimization (e.g., 'src.fire.AgentPipeline'). Specify from project root and include the class itself (not just the file)",
    )
    metric: str = Field(
        ...,
        description=(
            "Dotted import path of dspy metric function (e.g., 'eval.metric.accuracy). Specify from project root and include the class itself (not just the file)"
        ),
    )
    # Dataset — provide inline data OR a path to a file in the repo
    trainset: list[dict[str, Any]] | None = Field(
        default=None,
        description="Training dataset as inline list of dicts",
    )
    trainset_path: str | None = Field(
        default=None,
        description="Path to training data file in repo (json/jsonl/csv)",
    )
    testset: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Held-out test dataset as inline list of dicts. Optional. Never "
            "seen by any agent during the run (it is locked); only the seed "
            "and the best candidate are scored on it at finalization."
        ),
    )
    testset_path: str | None = Field(
        default=None,
        description="Path to held-out test data file in repo (json/jsonl/csv). Optional.",
    )
    input_keys: list[str] | None = Field(
        default=None,
        description="Explicit input field names for dspy.Example.with_inputs()",
    )

    # LM configuration
    # NOTE: there is no `program_lm` knob. The LM that runs the client's
    # program is configured inside the client repo (e.g. a `dspy.LM` bound via
    # `dspy.context(lm=...)` in the program's `__init__`/`forward`). If the run
    # needs to constrain the program's model, express it via
    # `additional_instructions`. CodeEvolver only owns `reflection_lm`.
    reflection_lm: str = Field(
        default="anthropic/claude-opus-4-7",
        description=(
            "Anthropic litellm model (anthropic/...) that drives the seed-step "
            "codeevolver.md summary AND selects the model for every Claude Agent "
            "SDK agent (architect, coding, inspection). Must be an Anthropic "
            "model — the Agent SDK is Anthropic-only; non-Anthropic values are "
            "rejected at startup."
        ),
    )
    coding_harness: Literal["opencode", "claude"] | None = Field(
        default=None,
        description="Agentic coding harness for optimizer runs. Defaults to opencode.",
    )
    coding_harness_model: str | None = Field(
        default=None,
        description="Harness-specific coding model, separate from reflection_lm. Ignored when coding_harness is claude.",
    )

    # Optimization config
    max_metric_calls: int = Field(
        default=1000,
        description="Max metric evaluations (optimization budget)",
    )
    num_threads: int = Field(
        default=1,
        description="Threads for parallel row evaluation (opt-in; clamped to 128 and to the row count)",
    )
    seed: int = Field(
        default=0,
        description="Random seed for reproducibility",
    )

    # Additional guidance for CodeEvolver optimization
    additional_instructions: str | None = Field(
        default=None,
        description=(
            "Client-provided guidance for CodeEvolver optimization. "
            "May include: constraints (changes that are off-limits), "
            "services (external APIs available with keys in environment), "
            "and ideas for optimization approaches."
        ),
    )

    # Extra file fences for the autonomous agents
    additional_deny_paths: list[str] | None = Field(
        default=None,
        description=(
            "Extra path fragments the coding/inspection/reflection agents must "
            "never read or edit. Substring-matched (absolute or relative), "
            "appended to CodeEvolver's built-in fences (valset/test set, etc.)."
        ),
    )

    # Code mutation frequency control with exponential decay
    initial: int = Field(
        default=1,
        description=(
            "Starting ratio of prompts per code change. "
            "0=code only, 1=alternating (1:1), 2=2 prompts per code, etc. "
        ),
    )
    decay_rate: int = Field(
        default=25,
        description="Iterations between each multiplier step for prompts-per-code ratio",
    )
    decay_factor: int = Field(
        default=2,
        description="Multiplier applied at each decay step (e.g., 2 means ratio doubles)",
    )
    code_cutoff_step: int | None = Field(
        default=None,
        description="Stop code mutations after this iteration (None=no cutoff)",
    )

    # Evaluation batch size
    subsample_size: int = Field(
        default=20,
        description="Number of examples per evaluation batch (subsample size)",
    )

    # Per-evaluation timeouts
    subsample_eval_timeout: int = Field(
        default=7200,
        description=(
            "Hard backstop in seconds for a single architect subsample (default 7200s / 2 hours)."
        ),
    )
    comparison_eval_timeout: int = Field(
        default=14400,
        description="Timeout in seconds for one comparison-set or finalization test-set evaluation (default 14400s / 4 hours)",
    )
    test_samples: int = Field(
        default=3,
        ge=1,
        description=(
            "How many times finalization draws the held-out test set for the FINAL "
            "program (never budget-counted). Every reading is reported "
            "(final_test_scores); final_test_score is the mean."
        ),
    )
    seed_test_samples: int = Field(
        default=1,
        ge=1,
        description=(
            "Held-out draws for the SEED program at finalization (seed_test_scores / "
            "seed_test_score). Defaults to 1 — across a research matrix every arm "
            "re-measures the same seed."
        ),
    )

    variance_estimate_samples: int = Field(
        default=5,
        ge=2,
        description=(
            "How many times the architect's once-per-run `run_variance_estimate` "
            "tool evaluates the program on its fixed training sample (counted "
            "against max_metric_calls: N x subsample_size rows). Internal knob; "
            "the agent never chooses it."
        ),
    )

    # Comparison set size (engine 3)
    comparison_set_size: int | None = Field(
        default=None,
        description=(
            "Size of the fixed comparison set drawn from the trainset (seed 42), "
            "used by the architect for apples-to-apples branch comparison. "
            "Counted against max_metric_calls. Defaults to the server setting."
        ),
    )

    # Comparison-credit ledger (specs/archive-public/comparison_credits.md)
    comparison_credits_initial: int = Field(
        default=1,
        ge=0,
        description=(
            "Comparison-set credits granted up front at the start of the seed "
            "round. Arms whose seed hook runs a comparison eval (pareto_evolution, "
            "asa) spend this credit on it."
        ),
    )
    comparison_credits_per_iteration: int | None = Field(
        default=1,
        ge=0,
        description=(
            "Comparison-set credits earned at each iteration start. Agent-initiated "
            "comparison evals are refused at balance <= 0; loop-initiated runs debit "
            "but may overdraw. None disables the credit ledger entirely."
        ),
    )

    # Git branch configuration
    initial_branch: str = Field(
        default="main",
        description="Git branch to use as the starting point for optimization",
    )

    # Sandbox Python version (must match the client repo's required Python)
    python_version: str = Field(
        default="3.11",
        description="Python version for the workspace venv (e.g. '3.11', '3.12').",
    )

    # Post-run meta-reflection (advisory; strictly non-fatal). On by default.
    # When True, the IterationArchitect's own session traces are captured
    # during the run and a single advisory meta-reflection agent runs once at
    # the very end, writing proposals to /memory/meta_reflection.md. Set False
    # to skip the extra agent run and trace capture.
    enable_meta_reflection: bool = Field(
        default=True,
        description=(
            "Run the post-run meta-reflection agent (advisory, non-fatal). "
            "Captures architect session traces during the run and emits "
            "improvement proposals at finalization. Default True."
        ),
    )

    # Post-run run-integrity inspection (advisory; strictly non-fatal). The
    # verdict (cheating category + functioning-iteration %) is stamped onto
    # the latest iteration history record — it never stops or fails the run.
    enable_run_integrity_agent: bool = Field(
        default=True,
        description=(
            "Run the post-run run-integrity inspector (advisory, non-fatal). "
            "Classifies cheating (test-set leakage / hard-coding) and the % "
            "of functioning iterations; stamps the verdict on the latest "
            "iteration history record. Default True."
        ),
    )

    # Start-of-run clean-slate inspection. OPT-IN: a 'contaminated' verdict
    # ABORTS the run before any eval spend, so this defaults to False and is
    # enabled explicitly by research configs.
    enable_clean_slate_inspector: bool = Field(
        default=False,
        description=(
            "Run the start-of-run clean-slate inspector in the seed step. "
            "Checks git history, deny-path coverage of holdout data, and "
            "container leftovers; a 'contaminated' verdict aborts the run. "
            "Default False (opt-in)."
        ),
    )

    # ASA (specs/archive-public/v3_asa.md) — used only by method="asa"
    asa_behavioral_contribution_constant: float = Field(
        default=0.10,
        description=(
            "ASA: constant multiplied against the current gap to the maximum expected "
            "score; sets how sensitive ASA is to declaring a non-primary lineage "
            "eligible. Max 1.0, primary at 0.90 -> gap 0.1 -> threshold 0.01."
        ),
    )
    asa_behavioral_contribution_threshold_cap: float = Field(
        default=0.05,
        description="ASA: upper bound on the eligibility threshold (per-row score units).",
    )
    asa_primary_stale_landed_iterations: int = Field(
        default=4,
        description=(
            "ASA: the loop refreshes the primary's comparison record once it is older "
            "than this many landed iterations on the primary branch."
        ),
    )

    # Debug mode
    debug_max_iterations: int | None = Field(
        default=None,
        description=(
            "Stop optimization after this many iterations (for testing). "
            "Recommended: 5 for quick tests. None=run until max_metric_calls."
        ),
    )

    @field_validator("python_version")
    @classmethod
    def _check_python_version(cls, v: str) -> str:
        return validate_python_sandbox_version(v)

    @model_validator(mode="after")
    def check_trainset_provided(self) -> "OptimizeRequest":
        if self.trainset is None and self.trainset_path is None:
            raise ValueError("Provide either 'trainset' (inline data) or 'trainset_path' (file in repo)")
        return self

class JobStatusUpdateRequest(BaseModel):
    """PUT /internal/job/{job_id}/status — update job status from optimizer sandbox."""

    status: str
    final_candidate: dict[str, Any] | None = None
    # Comparison-set (trainset subsample) scores, measured once at
    # finalization for metrics only (never budget-counted).
    seed_train_score: float | None = None
    final_train_score: float | None = None
    seed_test_score: float | None = None  # seed scored on the held-out test set (mean over draws)
    final_test_score: float | None = None  # final primary on the test set (mean over draws)
    seed_test_scores: list[float] | None = None  # every test-set draw of the seed
    final_test_scores: list[float] | None = None  # every test-set draw of the final primary
    total_metric_calls: int | None = None
    num_branches: int | None = None
    error: str | None = None
    ce_state: dict[str, Any] | None = None  # Full CE state on completion


class JobProgressUpdateRequest(BaseModel):
    """PUT /internal/job/{job_id}/progress — iteration progress from the optimizer."""

    current_iteration: int
    final_candidate: dict[str, Any]
    total_metric_calls: int
    num_branches: int
    ce_state: dict[str, Any] | None = None
    memory_md: str | None = None  # Latest /memory/memory.md snapshot from the sandbox


class ImportJobHistoryRequest(BaseModel):
    """POST /import-job-history — upload a local run's iteration history.

    API-key authenticated. Lets a researcher push the history of a job that
    ran locally (and was only mirrored to a local Mongo / JSONL) into the
    cloud database, without handling the raw Mongo credentials. Upserts the
    job under the caller's clerk_id; creates the job record if it is absent.
    """

    job_id: str
    iteration_history: dict[str, Any]  # iteration idx (str) -> IterationHistoryRecord
    status: str | None = None
    repo_url: str | None = None
    final_candidate: dict[str, Any] | None = None
    num_branches: int | None = None
    memory_md: str | None = None
    name: str | None = None  # Display name shown in the portal instead of job_id
    seed_test_score: float | None = None
    final_test_score: float | None = None
    meta_reflection_md: str | None = None  # meta_reflection.md report
    integrity_audit_md: str | None = None  # run_integrity.md report


class GitHubTokenResponse(BaseModel):
    """GET /internal/job/{job_id}/github-token — refresh GitHub token."""

    token: str | None = None
    error: str | None = None


class CleanupBranchesRequest(BaseModel):
    """Request payload for POST /cleanup-branches endpoint.

    Deletes all remote git branches matching 'codeevolver-{timestamp}*' pattern,
    except those explicitly listed.
    """

    repo_url: str = Field(
        ...,
        description="GitHub repository URL (e.g., 'https://github.com/user/repo')",
    )
    date: str = Field(
        ...,
        description="Date-time scope for branch cleanup in YYYYMMDDHHmmss format (e.g., '20260101120000'). Pass 'none' (case-insensitive) to match all codeevolver branches regardless of date.",
    )
    except_branches: list[str] = Field(
        ...,
        description="List of branch names to keep (exceptions). These branches will NOT be deleted even if they match the date pattern.",
    )

    @model_validator(mode="after")
    def validate_date_format(self) -> "CleanupBranchesRequest":
        # Normalize "none"/"None"/"NONE" etc. to lowercase
        if self.date.lower() == "none":
            self.date = "none"
            return self
        if len(self.date) != 14 or not self.date.isdigit():
            raise ValueError("date must be in YYYYMMDDHHmmss format (e.g., '20260101120000') or 'none' for all dates")
        return self
