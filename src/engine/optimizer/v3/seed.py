"""Copyright © 2026 440 Labs LLC

Engine 3 seed step — runs before the architect loop starts.

Owned responsibilities:
  1. Create the run-main branch (`codeevolver-{ts}-main`) from the configured
     initial branch — this is the run's INITIAL PRIMARY.
  2. Generate `codeevolver.md` via a read-only repo-analysis agent session
     (Read/Grep/Glob, through the selected harness; with fallback).
  3. Initialize `/memory/memory.md` from the starter template.
  4. Run the environment sanity eval (a few training rows — NOT counted
     against the eval budget).
  5. Construct `CEState` (budget counter starts at 0).
  6. Pre-populate `IterationHistory[0]` as the seed record so architect
     iterations can start at index 1.

There is no seed valset eval — engine 3 has no validation set.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.engine.evaluator.config import EvaluatorConfig
from src.engine.evaluator.dataset_config import DATASETS_MANIFEST_FILENAME, DatasetConfig
from src.engine.paths import history_dir
from src.engine.sandbox.verify_environment import validate_sandbox_environment
from src.engine.services.job_context import JobContext

from .codeevolver_md import (
    build_codeevolver_md_fallback_body,
    build_codeevolver_md_prompt,
    extract_architecture_summary,
    extract_architecture_title,
)
from src.engine.services.subprocess_service import (
    git_add_and_commit,
    git_checkout,
    git_create_branch_simple,
    git_fetch,
    git_get_sha,
    git_push_authenticated,
    run_inspection_agent,
    write_file,
)
from .ce_state import CEState
from .clean_slate import CLEAN_SLATE_REPORT_DENY_FRAGMENT
from src.engine.schemas.iteration_history import IterationHistory
from .prompts import build_memory_md_template
from .select_parent import utc_run_timestamp

logger = logging.getLogger(__name__)


def run_seed_step(
    *,
    workspace: str,
    datasets: DatasetConfig,
    evaluator_config: EvaluatorConfig,
    reflection_lm: str,
    program: str,
    metric: str,
    job: JobContext,
    environment_check_timeout: int = 7200,
    method: str = "ee",
    agent_model: str | None = None,
    coding_harness: str = "claude",
    coding_harness_model: str | None = None,
    additional_deny_paths: list[str] | None = None,
    enable_clean_slate_inspector: bool = False,
) -> tuple[Any, IterationHistory, dict]:
    """Run the pre-loop seed step.

    Returns `(state, history, seed_meta)` where `seed_meta` carries the
    values the top-level runner needs for its result dict
    (`seed_branch`/`seed_sha` plus codeevolver.md analysis token usage).
    """
    run_timestamp = utc_run_timestamp()
    seed_branch = f"codeevolver-{run_timestamp}-main"

    # Mirrors v1 (`main:src/engine/optimizer/adapter.py::_create_ce_main_branch`):
    # fetch origin and check out the configured initial branch before
    # branching so the run starts from the latest remote state. Best-effort —
    # fetch failures (e.g. local tests with no remote) log and continue.
    try:
        git_fetch(workspace)
    except RuntimeError as exc:
        logger.warning("seed: git fetch failed (continuing): %s", exc)
    try:
        git_checkout(workspace, job.initial_branch)
    except RuntimeError as exc:
        logger.warning(
            "seed: checkout of initial_branch=%s failed (continuing from HEAD): %s",
            job.initial_branch, exc,
        )

    # Keep the analysis agent blind to the held-out test set, mirroring the
    # holdout fence used by the architect and its sub-agents.
    holdout_deny_paths = [p for p in (datasets.testset_rel_path,) if p]
    # Append client-supplied extra fences (substring fragments) so the seed
    # analysis agent is blind to them too.
    if additional_deny_paths:
        holdout_deny_paths.extend(p for p in additional_deny_paths if p)
    # The clean-slate inspector's own report lands in the memory dir, next to
    # memory.md, which the architect is told to read. Fence it by name so a
    # finding that quotes a filename can never become architect context.
    holdout_deny_paths.append(CLEAN_SLATE_REPORT_DENY_FRAGMENT)

    # Start-of-run clean-slate inspection (opt-in). Runs BEFORE the run branch
    # exists and before any agent spend, for three reasons: the inspector is
    # told that `codeevolver-` branches mean contamination (so it must not be
    # standing on this run's own one); it is told the repo should hold only
    # the seed program (so `codeevolver.md` must not be committed yet); and an
    # abort here costs nothing, whereas one after `_generate_codeevolver_md`
    # has already paid for an agent session. This run's own memory dir is
    # still empty at this point, while OTHER jobs' dirs — the actual
    # contamination it is looking for — are visible.
    #
    # An explicit "contaminated" verdict raises `CleanSlateContaminationError`
    # and a mutating inspector raises `CleanSlateMutationError`; everything
    # else fails open and the status is stamped on the seed record below.
    clean_slate_status: str | None = None
    if enable_clean_slate_inspector:
        from .clean_slate import run_clean_slate_inspection

        clean_slate_status = run_clean_slate_inspection(
            workspace=workspace,
            initial_branch=job.initial_branch,
            deny_paths=holdout_deny_paths,
            testset_rel_path=datasets.testset_rel_path,
            agent_model=agent_model,
            coding_harness=coding_harness,
            coding_harness_model=coding_harness_model,
        )

    logger.info(
        "seed: creating run-main branch %s from %s", seed_branch, job.initial_branch
    )
    git_create_branch_simple(workspace, seed_branch)
    _exclude_engine_artifacts_from_git(workspace)

    codeevolver_md_body, codeevolver_md_usage = _generate_codeevolver_md(
        workspace=workspace,
        program=program,
        metric=metric,
        agent_model=agent_model,
        coding_harness=coding_harness,
        coding_harness_model=coding_harness_model,
        deny_paths=holdout_deny_paths,
    )
    architecture_title = (
        extract_architecture_title(codeevolver_md_body) or f"Seed ({program})"
    )
    architecture_summary = extract_architecture_summary(codeevolver_md_body)

    write_file(os.path.join(workspace, "codeevolver.md"), codeevolver_md_body)
    try:
        git_add_and_commit(workspace, "seed: add codeevolver.md", paths=["codeevolver.md"])
    except RuntimeError as exc:
        logger.warning("seed: codeevolver.md commit failed (continuing): %s", exc)

    # Push the run-main branch so downstream tooling (and users watching the
    # GitHub UI) can see the run has started. Non-fatal on failure — matches
    # v1 behavior of letting the loop progress even without a remote.
    if job.repo_url:
        push_result = git_push_authenticated(
            workspace=workspace,
            branch=seed_branch,
            **job.git_creds(),
            set_upstream=True,
        )
        if not push_result["success"]:
            logger.warning(
                "seed: push of %s failed (continuing): %s",
                seed_branch, push_result["stderr"],
            )

    memory_md = build_memory_md_template(
        seed_branch=seed_branch,
        architecture_title=architecture_title,
        method=method,
    )
    from src.engine.paths import memory_dir
    write_file(f"{memory_dir()}/memory.md", memory_md)

    # Seed the combined change-summaries file with the seed/Candidate 0 entry
    # so iteration 1's architect (which reads it in Phase 1) finds it present —
    # the in-loop regeneration only runs after an iteration completes.
    from .change_summary import seed_change_summary_section
    write_file(
        f"{memory_dir()}/iteration_change_summaries.md",
        seed_change_summary_section(architecture_title) + "\n",
    )

    # Environment smoke test: run the generic evaluator on the first 5
    # training rows. Raises `SandboxValidationError` if the environment is
    # broken, which propagates out so the run aborts before the architect
    # loop starts. NOT counted against the eval budget (spec §24).
    validate_sandbox_environment(
        workspace=workspace,
        evaluator_config=evaluator_config,
        datasets=datasets,
        n_rows=5,
        timeout=environment_check_timeout,
    )

    seed_head_sha = git_get_sha(workspace)
    state = CEState(
        seed_branch=seed_branch,
        seed_sha=seed_head_sha,
        comparison_history_path=f"{history_dir()}/comparison_set_history.jsonl",
    )
    # Dataset identity for the offline frontier analysis: file hashes (row
    # ids are positional) and the comparison ids nothing else persists.
    datasets.write_manifest(
        workspace, os.path.join(history_dir(), DATASETS_MANIFEST_FILENAME)
    )

    # Iteration 0 == seed; architect iterations start at 1. The seed's
    # held-out `test_score` is stamped later by the finalization step.
    history = IterationHistory()
    seed_record = history.create_entry(0)
    seed_record.update(
        {
            "git_branch": seed_branch,
            "close_sha": seed_head_sha,
            "architecture_title": architecture_title,
            "architecture_summary": architecture_summary,
            "primary_branch_after": seed_branch,
            "skipped": False,
        }
    )
    if clean_slate_status is not None:
        seed_record["clean_slate_status"] = clean_slate_status

    return (
        state,
        history,
        {
            "seed_branch": seed_branch,
            "seed_sha": seed_head_sha,
            # Token usage of the codeevolver.md repo-analysis agent (by-model
            # accounting). `{}` when the fallback body was used.
            "codeevolver_md_usage": codeevolver_md_usage,
            "codeevolver_md_model": agent_model or coding_harness_model,
        },
    )


def _truncate_repr(value: Any, limit: int = 240) -> str:
    s = repr(value)
    return s if len(s) <= limit else s[:limit] + "…"


# Engine-owned paths that live INSIDE the client workspace and must never
# count as program content: the staged skill dirs, the workspace lock, and
# the in-workspace artifact dir some launchers create. Content identity is a
# `git add -A` tree hash (`git_tree_signature`), so an untracked file here
# would make every close look like a program change — and, on the evolution
# arm, force a comparison-set re-run every iteration.
_ENGINE_EXCLUDE_PATTERNS = (
    ".codeevolver/",
    ".codeevolver.lock",
    ".claude/skills/",
    ".opencode/skills/",
    "__pycache__/",
    "*.pyc",
)


def _exclude_engine_artifacts_from_git(workspace: str) -> None:
    """Append engine-owned patterns to `.git/info/exclude` (never to the
    client's .gitignore, which is program content). Idempotent, best-effort."""
    exclude_path = os.path.join(workspace, ".git", "info", "exclude")
    try:
        existing = ""
        if os.path.exists(exclude_path):
            with open(exclude_path) as f:
                existing = f.read()
        present = set(existing.splitlines())
        missing = [p for p in _ENGINE_EXCLUDE_PATTERNS if p not in present]
        if not missing:
            return
        os.makedirs(os.path.dirname(exclude_path), exist_ok=True)
        with open(exclude_path, "a") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write("# codeevolver: engine-owned artifacts (not program content)\n")
            f.write("\n".join(missing) + "\n")
    except OSError as exc:
        logger.warning("seed: could not update .git/info/exclude (non-fatal): %s", exc)


def _generate_codeevolver_md(
    *,
    workspace: str,
    program: str,
    metric: str,
    agent_model: str | None,
    coding_harness: str,
    coding_harness_model: str | None,
    deny_paths: list[str] | None = None,
) -> tuple[str, dict]:
    """Generate a codeevolver.md body via a read-only repo-analysis agent.

    Runs a Read/Grep/Glob agent session (through the selected harness) so the
    model inspects the program source. Returns ``(body, usage)`` where ``body``
    is the agent's final synthesis (the architecture summary) and ``usage`` is
    the agent session's token-usage dict (``{}`` when the agent did not run /
    reported nothing) for run-level token accounting.
    """
    prompt = build_codeevolver_md_prompt(program_path=program, metric_path=metric)
    usage: dict = {}
    try:
        logger.info(
            "seed: requesting architecture summary via %s agent session (model=%s)",
            coding_harness, agent_model or coding_harness_model,
        )
        result = run_inspection_agent(
            workspace=workspace,
            prompt=prompt,
            max_turns=25,
            iteration_num=None,  # skip trace-inspection-only diagnostics
            writable=False,  # read-only: Read/Grep/Glob
            deny_paths=deny_paths,
            model=agent_model,
            coding_harness=coding_harness,
            coding_harness_model=coding_harness_model,
        )
        usage = (result or {}).get("usage") or {}
        body = (result or {}).get("proposed_change") or ""
        if body.strip():
            return body.strip() + "\n", usage
        logger.warning(
            "seed: analysis agent produced no usable summary (%s); using fallback",
            (result or {}).get("error"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "seed: analysis agent failed (%s); using fallback codeevolver.md",
            exc,
        )
    return (
        build_codeevolver_md_fallback_body(
            program_path=program,
            metric_path=metric,
        ),
        usage,
    )


__all__ = ["run_seed_step"]
