"""Shared architect tool metadata for provider harnesses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ArchitectToolSpec:
    name: str
    description: str
    # Full JSON Schema (with an explicit `required` list) — single source of
    # truth for both the OpenCode/MCP tool definitions and the Claude Agent
    # SDK `@tool` decorator. Do NOT collapse this back to a bare name->type
    # dict: the SDK marks every key of a bare dict as required, which is what
    # made the optional `filter` arg fail validation on run_eval.
    input_schema: dict[str, Any]

    def as_mcp_tool(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


RUN_EVAL_DESCRIPTION = (
    "Evaluate the current program. eval_sample='train' runs the "
    "training subsample (first call locks the subsample ids as this "
    "iteration's baseline; later calls reuse them so before/after "
    "deltas are fair). eval_sample='single' runs a single training "
    "row (a 1-row smoke check: spends 1 budget row, never locks the "
    "subsample, not a score to act on). eval_sample='comparison' runs the FIXED comparison set (a "
    "trainset subsample drawn once per run with seed 42, identical "
    "across branches and iterations) -- use it sparingly, for "
    "apples-to-apples branch comparison; it consumes real eval "
    "budget and is logged to comparison_set_history.jsonl with the "
    "branch and sha evaluated. Optional `filter` narrows the training "
    "subsample to rows matching all criteria: map column -> value "
    "(exact match) or column -> "
    "{\"eq\"|\"neq\"|\"in\"|\"contains\": value} for explicit ops. "
    "Example: {\"category\": \"metals\"} or "
    "{\"description\": {\"contains\": \"steel\"}}. Filter is locked "
    "on the first train call and ignored for 'single'/'comparison'. "
    "Subsample size is set by the run config -- you do not choose it. "
    "A filter may yield fewer rows than the configured size when the "
    "matching pool is smaller. Comparison runs are additionally limited "
    "by a credit system that replenishes at a fixed rate per iteration "
    "-- use them prudently! The first comparison call of an iteration "
    "returns an acknowledgement note with a verification code; re-call "
    "with `acknowledge_message` set to that code to proceed. Returns "
    "per-metric scores, primary_score, n_rows, failure counts, and "
    "trace/log file paths (per-row feedback: index_pretty_print.txt "
    "in trace_dir). This is the ONLY permitted way to evaluate -- do not "
    "run the metric or program from Bash or Python."
)

CODING_AGENT_DESCRIPTION = (
    "Invoke the coding sub-agent on the current branch. "
    "mode='execute' is the only supported mode; mode='tracing' is "
    "deferred. Must be called after at least one 'train' run_eval "
    "(to establish the baseline)."
)

LOAD_CLOSE_SKILL_FIRST = (
    "Load the `close_iteration` skill BEFORE calling this tool and follow "
    "its steps. "
)

CLOSE_ITERATION_DESCRIPTION = (
    LOAD_CLOSE_SKILL_FIRST +
    "Finalize this iteration. Call exactly once, at the end, after "
    "your work is committed and memory.md is updated (re-calling "
    "amends the registration; the last call wins). `register_branch` "
    "is a JSON OBJECT -- never a list -- recording the branch you are "
    "submitting: "
    "{\"branch\": \"codeevolver-...\", "
    "\"was_primary\": bool, \"is_primary\": bool, "
    "\"promotion_evidence\": \"...\"}. `was_primary` = the branch "
    "was the primary at the start of this iteration; `is_primary` = it "
    "is the primary after your changes (promoting a new primary "
    "requires promotion_evidence, e.g. comparison-set scores). "
    "`improvement_mode` declares whether this iteration explored a new "
    "macro approach ('exploration') or improved the best program "
    "('exploitation'). `memory_updated` = whether you updated "
    "memory.md this iteration. This tool is the sole source of truth "
    "for the primary branch."
)

# pareto_evolution: the optimizer forked the branch and owns retention +
# primary, so the tool carries only the memory claim. No branch, no primary,
# no improvement mode (the arm carries no explore/exploit guidance).
CLOSE_ITERATION_LOOP_BRANCH_DESCRIPTION = (
    LOAD_CLOSE_SKILL_FIRST +
    "Finalize this iteration. Call exactly once, at the end, after your "
    "work is committed on the branch you were given and memory.md is "
    "updated (re-calling amends the claim; the last call wins). The "
    "optimizer already knows your branch -- do not switch branches. "
    "`memory_updated` = whether you updated memory.md this iteration."
)

# freeform: the loop registers whatever HEAD is at close, so the tool carries
# only the memory claim. No branch, no primary, no improvement mode.
CLOSE_ITERATION_HEAD_DESCRIPTION = (
    LOAD_CLOSE_SKILL_FIRST +
    "Finalize this iteration. Call exactly once, at the end, after your "
    "work is committed and memory.md is updated (re-calling amends the "
    "claim; the last call wins). The commit checked out at HEAD when the "
    "iteration closes is your submitted state. `memory_updated` = whether "
    "you updated memory.md this iteration."
)

BRANCH_COMPARISON_DESCRIPTION = (
    "Read-only; costs no eval rows. For every registered lineage, this tool tells you how many rows a branch has an advantage over a reference branch on the fixed comparison set. It also computes the 'positive behavioral contribution' of the branch relative to a reference branch on the fixed comparison set. 'positive behavioral contribution' is the summation across all rows, i, of the difference between branch_score_i - reference_score_i, subject to branch_score_i > reference_score_i. In other words, max(0, branch_score_i - reference_score_i). A normalized score is also calculated (i.e., by dividing by the number of rows in the comparison set). The pruned Pareto score is calculated and tells you the number of rows where the lineage ties the max for that row across all candidates; "
    "'n/a - fully dominated' is when that branch has no rows with any advantage over the other candidates. "
    "This tool also reveals the ELIGIBLE non-primary parents. Not all non-primary parents are eligible. An eligible non-primary parent is a parent branch that has a positive behavioral contribution against the primary which exceeds a minimum threshold. "
    "`reference` defaults to the current primary. A lineage without a comparison record "
    "is listed in the footer."
)

RUN_VARIANCE_ESTIMATE_DESCRIPTION = (
    "Estimate run-to-run evaluation noise (specs: sampling variance). "
    "Auto-commits any uncommitted changes, draws ONE fresh random training "
    "sample of subsample_size rows, and evaluates the CURRENT program on "
    "those same rows N times (N is fixed by the run config, typically 5) -- "
    "costing N x subsample_size eval rows from your budget. Returns the "
    "score of each run, the within-item variance (per-row variance across "
    "the repeated runs, pooled) and the run-to-run standard deviation of the "
    "subsample metric (sqrt(within_item_variance / rows)). The result is "
    "stored for the rest of the run and shown in every later iteration's "
    "system prompt. May be called ONCE per run -- later calls are refused "
    "without spending rows. These runs are NOT training evals: they do not "
    "set or replace this iteration's baseline. Best used after a few "
    "improvement cycles (end of iterations 1-3), not on the seed program. "
    "Takes no arguments."
)

INSPECT_TRACES_DESCRIPTION = (
    "Spawn a read-only reflection sub-agent to analyze one of this "
    "iteration's trace JSONL files. `trace=\"parent\"` is the BASELINE "
    "file -- the first run_eval(train) of the iteration, before your "
    "changes. `trace=\"child\"` is the file from your latest "
    "run_eval(train) after a change (it does not exist until you have "
    "run one). `trace=\"comparison\"` is this iteration's latest "
    "comparison-set eval (the directory is rewritten by each comparison "
    "eval -- check provenance.json for which version it describes). Pass "
    "a specific, substantive natural-language query -- "
    "the inspector is your eyes into the trace file."
)


ARCHITECT_TOOL_SPECS: dict[str, ArchitectToolSpec] = {
    "run_eval": ArchitectToolSpec(
        name="run_eval",
        description=RUN_EVAL_DESCRIPTION,
        input_schema={
            "type": "object",
            "properties": {
                "eval_sample": {
                    "type": "string",
                    "enum": ["single", "train", "comparison"],
                },
                "filter": {
                    "type": "object",
                    "additionalProperties": True,
                },
                "acknowledge_message": {
                    "type": "string",
                    "description": (
                        "Verification code from the acknowledgement note "
                        "returned by the first comparison call of this "
                        "iteration. Required to run a comparison eval."
                    ),
                },
            },
            "required": ["eval_sample"],
            "additionalProperties": False,
        },
    ),
    "coding_agent": ArchitectToolSpec(
        name="coding_agent",
        description=CODING_AGENT_DESCRIPTION,
        input_schema={
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["execute", "tracing"]},
                "change_request": {"type": "string"},
            },
            "required": ["mode", "change_request"],
            "additionalProperties": False,
        },
    ),
    "inspect_traces": ArchitectToolSpec(
        name="inspect_traces",
        description=INSPECT_TRACES_DESCRIPTION,
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "trace": {"type": "string", "enum": ["parent", "child", "comparison"]},
            },
            "required": ["query", "trace"],
            "additionalProperties": False,
        },
    ),
    "run_variance_estimate": ArchitectToolSpec(
        name="run_variance_estimate",
        description=RUN_VARIANCE_ESTIMATE_DESCRIPTION,
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    ),
    "branch_comparison": ArchitectToolSpec(
        name="branch_comparison",
        description=BRANCH_COMPARISON_DESCRIPTION,
        input_schema={
            "type": "object",
            "properties": {"reference": {"type": ["string", "null"]}},
            "required": [],
            "additionalProperties": False,
        },
    ),
    "close_iteration": ArchitectToolSpec(
        name="close_iteration",
        description=CLOSE_ITERATION_DESCRIPTION,
        input_schema={
            "type": "object",
            "properties": {
                # The single branch this iteration submits. Deliberately a
                # flat object (never a list, never keyed by repo name): one
                # entry makes duplicate primaries structurally impossible.
                "register_branch": {
                    "type": "object",
                    "properties": {
                        "branch": {"type": "string"},
                        "was_primary": {"type": ["boolean", "string"]},
                        "is_primary": {"type": ["boolean", "string"]},
                        "promotion_evidence": {"type": "string"},
                    },
                    "required": ["branch", "was_primary", "is_primary"],
                    "additionalProperties": False,
                },
                "improvement_mode": {
                    "type": "string",
                    "enum": ["exploration", "exploitation"],
                },
                "memory_updated": {"type": "boolean"},
            },
            "required": ["register_branch", "improvement_mode", "memory_updated"],
            "additionalProperties": False,
        },
    ),
}


CloseIterationMode = Literal["register_branch", "loop_branch", "head"]

_CLOSE_ITERATION_HEAD_SPEC = ArchitectToolSpec(
    name="close_iteration",
    description=CLOSE_ITERATION_HEAD_DESCRIPTION,
    input_schema={
        "type": "object",
        "properties": {"memory_updated": {"type": "boolean"}},
        "required": ["memory_updated"],
        "additionalProperties": False,
    },
)

_CLOSE_ITERATION_LOOP_BRANCH_SPEC = ArchitectToolSpec(
    name="close_iteration",
    description=CLOSE_ITERATION_LOOP_BRANCH_DESCRIPTION,
    input_schema={
        "type": "object",
        "properties": {"memory_updated": {"type": "boolean"}},
        "required": ["memory_updated"],
        "additionalProperties": False,
    },
)


def architect_tool_spec(
    name: str,
    *,
    close_iteration_mode: CloseIterationMode = "register_branch",
) -> ArchitectToolSpec:
    """The spec for `name`, honouring the strategy's close_iteration variant.

    `ARCHITECT_TOOL_SPECS` stays the EE default; only `close_iteration` has
    other shapes: the loop-owned-branch variant and the HEAD-registering
    variant (both `memory_updated` only; they differ in what the loop does
    with HEAD, not in the schema).
    """
    if name == "close_iteration" and close_iteration_mode == "loop_branch":
        return _CLOSE_ITERATION_LOOP_BRANCH_SPEC
    if name == "close_iteration" and close_iteration_mode == "head":
        return _CLOSE_ITERATION_HEAD_SPEC
    return ARCHITECT_TOOL_SPECS[name]


def ordered_architect_tool_names(*, agent_managed_git: bool = True) -> list[str]:
    """Canonical architect tool order.

    `agent_managed_git=True` adds `close_iteration`;
    Simple loop sets its tool list itself and does not expose coding_agent, inspect_traces, close_iteration, etc. It's decoupled.
    """
    names = ["run_eval", "coding_agent", "inspect_traces"]
    if agent_managed_git:
        names.append("close_iteration")
    # Once-per-run noise estimate (specs/archive-public/sampling_variance.md). Exposed by
    # every engine-3 registry until the run has taken its estimate — the
    # registry drops it afterwards, and both harnesses filter by membership.
    names.append("run_variance_estimate")
    # Strategy-owned (ASA). Listed here so both harnesses know the spec; it
    # is only EXPOSED when the strategy's `extra_tools` put it in the
    # registry (both harnesses filter by registry membership).
    names.append("branch_comparison")
    return names


def mcp_architect_tool_definitions(
    *,
    agent_managed_git: bool = True,
    registered_names: set[str] | None = None,
    close_iteration_mode: CloseIterationMode = "register_branch",
) -> list[dict[str, Any]]:
    names = ordered_architect_tool_names(agent_managed_git=agent_managed_git)
    return [
        architect_tool_spec(
            name,
            close_iteration_mode=close_iteration_mode,
        ).as_mcp_tool()
        for name in names
        if registered_names is None or name in registered_names
    ]


__all__ = [
    "ARCHITECT_TOOL_SPECS",
    "ArchitectToolSpec",
    "CloseIterationMode",
    "architect_tool_spec",
    "mcp_architect_tool_definitions",
    "ordered_architect_tool_names",
]
