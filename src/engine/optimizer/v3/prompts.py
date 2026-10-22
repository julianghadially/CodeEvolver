"""Copyright © 2026 440 Labs LLC

Prompt builders for the engine 3 optimizers.

- `ARCHITECT_SYSTEM_PROMPT_BASE` — the explore/exploit architect prompt
  skeleton used by EE (and ASA, which appends its rule sections): a lean
  prompt for the agent-managed loop (exploit guidance inline; exploration
  and iteration-closing guidance live in the `exploration` /
  `close_iteration` skills; tools: run_eval / coding_agent / inspect_traces
  / close_iteration). Strategy-owned sections (`PromptSections`) are
  spliced into named slots by `make_architect_prompt_builder`.
  `ARCHITECT_SYSTEM_PROMPT` is the EE-rendered template (kept under its
  historical name for importers/tests).
- `FREEFORM_ARCHITECT_SYSTEM_PROMPT_BASE` — research Baseline 1's prompt:
  the EE prompt with every explore/exploit and git/candidate-pool passage
  removed (`specs/v3_freeform.md`). Kept as its own literal (not spliced
  from the EE base) so it reads as one prompt; drift against EE is checked
  by hand. Two arms render it: `freeform` with its two slots empty
  (`FREEFORM_ARCHITECT_SYSTEM_PROMPT` / `build_freeform_architect_system_prompt`)
  and `pareto_evolution` with the `## Git Branch Background` section and
  the exploit-phase merge note spliced in (`build_evo_architect_system_prompt`).
  The evolution arm thus differs from freeform ONLY by the
  loop-owned population mechanics, never by search guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping
from src.engine.paths import history_dir, memory_dir, traces_dir
from src.engine.telemetry.trace_layout import eval_trace_dir

from .variance_estimate import variance_note

if TYPE_CHECKING:
    from .iteration_context import IterationContext

SystemPromptBuilder = Callable[..., str]


ARCHITECT_SYSTEM_PROMPT_BASE = """
Your task is to improve the program in the repository at `{workspace_path}`. 

To maximize improvements across the iterations you are allotted, you must balance your time between exploration and exploitation. We recommend at least 30% of the iterations be spent on exploration. During exploration, you primarily test out new approaches, which sometimes differ from the current approach (see macro approaches). Meanwhile, during exploitation, you should focus on improving the program as you normally would (chasing the most promising opportunity). 

## Exploitation Guidance (aka normal improvement)
1. Study {memory_dir}/memory.md and {memory_dir}/iteration_change_summaries.md to learn about past architectures attempted.
2. {exploit_branch_line}
3. Run a baseline evaluation on the training dataset using the `run_eval` tool. `run_eval` scores the program on a random {subsample_size}-row sample of the training set. Note, the run_eval tool fixes the training subsample of rows to use for this iteration. (using eval_sample="train")
4. Make your improvement(s) as you see fit. (Do not violate any constraints from the additional instructions below).
5. **Call `run_eval(eval_sample="train")` again** to measure the result. Any call after the baseline automatically re-uses the same sample, so the before/after numbers are directly comparable (a new seed is drawn for each iteration - your work spans one iteration).

You must run the baseline eval through the run_eval tool, and it is recommended to do this before making edits. Never run the training or comparison sets outside the run_eval tool or your work will be rejected. The system will force an eval run if you do not run one before your run ends. Each run writes a trace directory you can inspect with the inspect_traces tool (`trace="parent"` = the baseline eval's traces, `trace="child"` = your latest post-change eval's traces) or with Bash / Read / Grep. Traces often contain valuable examples to learn from on how a program executes. Pass a specific, substantive query — the inspector is your eyes into the traces.

    {traces_dir}/iteration_<N>_parent/   (baseline eval traces)
    {traces_dir}/iteration_<N>_child/    (post-change eval traces)

Each directory holds `index.jsonl` (one line per example: score, failed, feedback, `row_file`), `rows/<example_id>/row.json` (input, output, feedback, and a `spans` table), `rows/<example_id>/span_NNN.json` (one module call each: inputs, output, metadata) and `provenance.json` (the `eval_id`, `sha_at_eval`, `tree`, `committed`, `score` and `ts` of the eval that produced the directory). Every file fits a single Read. A directory is REWRITTEN by every eval with the same suffix, so before analysing a directory from an earlier iteration check its `provenance.json` against the code you are on — it may hold the traces of a version that was later reverted.

## Exploration Guidance
Spend at least 30% of the iterations experimenting with a variety of "macro approaches" (Exploration).
See the exploration skill for guidance on best practices.

### Explore Multiple Macro Approaches during Exploration
Macro approaches are non-incremental changes which provide a fundamentally new solution, often distinct from or divergent from existing approaches. This often includes a different approach to the program's architecture. 

## Making changes

Use `coding_agent(mode="execute", change_request=...)` for substantive changes, or edit files directly and commit with Bash for small edits. Either way, the baseline eval must come first.

Measure before and after - at least once per iteration.

## Directories
You have access to a few directories:
Workspace path: {workspace_path}
Memory directory: {memory_dir}
Traces directory: {traces_dir}
History directory: {history_dir}

{git_branching_section}
## Finishing an iteration

Once your planned improvement(s) are complete along with any follow-on changes you want to make, it is time to finalize the iteration. Call the close_iteration skill and follow those instructions.

{additional_section}

{program_section}
{subsample_note}
{budget_note}

"""


# ----- Strategy-owned prompt sections -----------------------------------
#
# EE's population-management instructions live in the prompt (the agent
# manages branches and the primary). The base keeps named slots so the
# sections stay whole; only EE renders it today (ASA appends to EE's).

EE_EXPLOIT_BRANCH_LINE = (
    "If you've chosen to exploit, select the branch marked in the memory file "
    "as the primary candidate (current best)."
)

EE_GIT_BRANCHING_SECTION = """## Git Branching Guidance / Candidate Pool
All work should be performed on branches with the prefix `codeevolver-`. DO NOT TOUCH BRANCHES THAT DON'T HAVE THE PREFIX `codeevolver-`. 

If you chose to exploit this iteration, you should use the "primary" branch and commit your changes to that same branch. At any given time, there is one branch that is the primary branch, which is currently set to: {primary_branch}. Other branches are primarily used for exploration. There is no need to create new branches or change the primary branch when choosing to exploit.

For exploration, you will create new branches and/or promote to primary according to guidance in the exploration skill. 
"""

# pareto_evolution's git section — spliced into the FREEFORM base (below),
# not the EE base: the loop forked this branch, the agent just works on it.
EVO_GIT_BRANCHING_SECTION = """## Git Branch Background
You are making changes on the branch {child_branch} as part of an evolutionary improvement process. This branch was selected by the optimizer because it scores uniquely well on certain rows (i.e., it is on the pareto frontier for several examples). Do not switch branches.
"""


@dataclass(frozen=True)
class PromptSections:
    """The strategy-owned pieces of the EE architect prompt."""

    git_branching_section: str
    exploit_branch_line: str


EE_PROMPT_SECTIONS = PromptSections(
    git_branching_section=EE_GIT_BRANCHING_SECTION,
    exploit_branch_line=EE_EXPLOIT_BRANCH_LINE,
)


def render_architect_prompt_template(sections: PromptSections) -> str:
    """Splice the strategy's sections into the base template.

    Plain `str.replace` (not `.format`) so a section may itself carry
    `{primary_branch}` / `{child_branch}` / `{memory_dir}` slots that the
    builder's single `.format()` pass renders together with the base.
    """
    return (
        ARCHITECT_SYSTEM_PROMPT_BASE
        .replace("{git_branching_section}", sections.git_branching_section)
        .replace("{exploit_branch_line}", sections.exploit_branch_line)
    )


# The EE-rendered template under its historical name (tests read it).
ARCHITECT_SYSTEM_PROMPT = render_architect_prompt_template(EE_PROMPT_SECTIONS)


# Whether this note appears is decided by the EVO loop's exploit-phase parent
# preselect (pareto_evolution/select_candidate.py::choose_parent_iteration),
# which draws a per-iteration probabilistic exploit-and-merge decision and
# stamps `IterationContext.exploit_phase`. Reading that single flag here keeps
# the merge guidance and the switch to best-candidate selection flipped
# together. EVO-only: merging is the agent's job (GEPA's merge half) and part
# of Treatment B's "population management with merging"
# (specs/research_plan.md); it is not explore/exploit guidance, so it is
# compatible with the freeform prompt the arm uses.
EVO_MERGE_NOTE = """
### Combine Approaches (end of run)
As you approach the end of the iteration budget, consider combining the best approaches from your memory of other branches (see memory.md). You can port over functionality from other branches to this branch by viewing files from other git branches. Make sure to keep your changes on the current branch.
"""


def _evo_merge_note(exploit_phase: bool) -> str:
    """Merge guidance for the EVO architect prompt.

    Present exactly when `IterationContext.exploit_phase` is True — the boolean
    the exploit-phase preselect drew this iteration — so the note and the
    best-candidate selection toggle together off one shared draw.
    """
    return EVO_MERGE_NOTE if exploit_phase else ""


def _additional_section(additional_instructions: str | None) -> str:
    if not additional_instructions:
        return ""
    return (
        "## Additional Instructions from Client\n"
        f"{additional_instructions}\n"
    )


def _parent_module_section(program_path: str | None) -> str:
    if not program_path:
        return ""
    return (
        "## Critical Constraint: Parent Module\n"
        f"All code changes MUST be made within the top-most parent module "
        f"class: `{program_path}`.\n"
        "- You may create new helper modules and sub-components, but they "
        "must be used by the parent module's entry point.\n"
        "- Do NOT create new wrapper classes that bypass the parent module.\n"
        "- Do NOT rename or replace the parent module class.\n"
        f"- The evaluation system calls `{program_path}` directly — any "
        "code not reachable from it will never execute.\n"
    )


def _program_section(program_path: str | None) -> str:
    """Spec §28: a very simple program pointer."""
    if not program_path:
        return ""
    return f"The program entry point is `{program_path}`.\n"


_MUST_EXPLORE_SECTION = """
## THIS ITERATION MUST EXPLORE
The share of iterations that have reported exploration has fallen below the 30% quota. In at least 30% of your iterations you are required to attempt changes that are of an exploratory nature - e.g., experimenting with a new macro approach (see the `exploration` skill) rather than incrementally improving the current program. 
"""


def make_architect_prompt_builder(sections: PromptSections) -> SystemPromptBuilder:
    """Produce a `SystemPromptBuilder` for one strategy's sections."""
    template = render_architect_prompt_template(sections)

    def _build(
        *,
        additional_instructions: str | None,
        program_path: str | None,
        workspace_path: str,
        context: IterationContext,
    ) -> str:
        prompt = template.format(
            memory_dir=memory_dir(),
            traces_dir=traces_dir(),
            history_dir=history_dir(),
            subsample_size=context.subsample_size,
            primary_branch=context.primary_branch_at_start or "(see memory.md)",
            child_branch=(
                context.child_branch or context.start_branch or "(the current branch)"
            ),
            subsample_note=_subsample_note(
                context.subsample_size,
                variance_estimate=context.variance_estimate,
                variance_estimate_samples=context.variance_estimate_samples,
            ),
            budget_note=_budget_note(
                context.max_eval_rows,
                context.evals_used,
                subsample_size=context.subsample_size,
                comparison_set_size=context.comparison_set_size,
            ),
            workspace_path=workspace_path,
            additional_section=_additional_section(additional_instructions),
            program_section=_program_section(program_path),
        )
        if context.must_explore:
            prompt += _MUST_EXPLORE_SECTION
        # ASA rule sections (specs/archive-public/v3_asa.md). No-op for every other arm —
        # only ASA's iteration-start hook sets `asa_requirements`.
        if getattr(context, "asa_requirements", None):
            from .asa.prompt_sections import system_prompt_sections

            prompt += system_prompt_sections(context)
        return prompt

    return _build


# EE (engine 3) system prompt — the agent manages branches and the primary.
build_architect_system_prompt = make_architect_prompt_builder(EE_PROMPT_SECTIONS)


# Freeform base (specs/v3_freeform.md §Prompt). Two slots, both empty for the
# freeform arm: `{git_branching_section}` (evo's `## Git Branch Background`)
# and `{merge_note}` (evo's exploit-phase merge note). They sit back-to-back
# with no separating newline so the freeform rendering is unchanged.
FREEFORM_ARCHITECT_SYSTEM_PROMPT_BASE = """
Your task is to improve the program in the repository at `{workspace_path}`. 

## Guidance
1. Study {memory_dir}/memory.md and {memory_dir}/iteration_change_summaries.md to learn about past architectures attempted.
2. Run a baseline evaluation on the training dataset using the `run_eval` tool. `run_eval` scores the program on a random {subsample_size}-row sample of the training set. Note, the run_eval tool fixes the training subsample of rows to use for this iteration. (using eval_sample="train")
3. Make your improvement(s) as you see fit. (Do not violate any constraints from the additional instructions below).
4. **Call `run_eval(eval_sample="train")` again** to measure the result. Any call after the baseline automatically re-uses the same sample, so the before/after numbers are directly comparable (a new seed is drawn for each iteration - your work spans one iteration).

You must run the baseline eval through the run_eval tool, and it is recommended to do this before making edits. Never run the training or comparison sets outside the run_eval tool or your work will be rejected. The system will force an eval run if you do not run one before your run ends. Each run writes a trace directory you can inspect with the inspect_traces tool (`trace="parent"` = the baseline eval's traces, `trace="child"` = your latest post-change eval's traces) or with Bash / Read / Grep. Traces often contain valuable examples to learn from on how a program executes. Pass a specific, substantive query — the inspector is your eyes into the traces.

    {traces_dir}/iteration_<N>_parent/   (baseline eval traces)
    {traces_dir}/iteration_<N>_child/    (post-change eval traces)

Each directory holds `index.jsonl` (one line per example: score, failed, feedback, `row_file`), `rows/<example_id>/row.json` (input, output, feedback, and a `spans` table), `rows/<example_id>/span_NNN.json` (one module call each: inputs, output, metadata) and `provenance.json` (the `eval_id`, `sha_at_eval`, `tree`, `committed`, `score` and `ts` of the eval that produced the directory). Every file fits a single Read. A directory is REWRITTEN by every eval with the same suffix, so before analysing a directory from an earlier iteration check its `provenance.json` against the code you are on — it may hold the traces of a version that was later reverted.

## Making changes

Use `coding_agent(mode="execute", change_request=...)` for substantive changes, or edit files directly and commit with Bash for small edits. Either way, the baseline eval must come first.

Measure before and after - at least once per iteration.

## Directories
You have access to a few directories:
Workspace path: {workspace_path}
Memory directory: {memory_dir}
Traces directory: {traces_dir}
History directory: {history_dir}

{git_branching_section}{merge_note}## Finishing an iteration

Once your planned improvement(s) are complete along with any follow-on changes you want to make, it is time to finalize the iteration. Call the close_iteration skill and follow those instructions.

{additional_section}

{program_section}
{subsample_note}
{budget_note}

"""

# The freeform-rendered template (slots spliced, `.format` fields intact —
# same convention as `ARCHITECT_SYSTEM_PROMPT`).
FREEFORM_ARCHITECT_SYSTEM_PROMPT = (
    FREEFORM_ARCHITECT_SYSTEM_PROMPT_BASE
    .replace("{git_branching_section}", "")
    .replace("{merge_note}", "")
)


def _freeform_format_kwargs(
    *,
    additional_instructions: str | None,
    program_path: str | None,
    workspace_path: str,
    context: IterationContext,
) -> dict[str, Any]:
    """The `.format` values shared by every freeform-based prompt: no
    comparison-set sentence in the sampling note, no explore guard."""
    return dict(
        memory_dir=memory_dir(),
        traces_dir=traces_dir(),
        history_dir=history_dir(),
        subsample_size=context.subsample_size,
        subsample_note=_subsample_note(
            context.subsample_size,
            include_comparison_note=False,
            variance_estimate=context.variance_estimate,
            variance_estimate_samples=context.variance_estimate_samples,
        ),
        budget_note=_budget_note(
            context.max_eval_rows,
            context.evals_used,
            subsample_size=context.subsample_size,
            comparison_set_size=context.comparison_set_size,
        ),
        workspace_path=workspace_path,
        additional_section=_additional_section(additional_instructions),
        program_section=_program_section(program_path),
    )


def build_freeform_architect_system_prompt(
    *,
    additional_instructions: str | None,
    program_path: str | None,
    workspace_path: str,
    context: IterationContext,
) -> str:
    """Freeform (Baseline 1) system prompt.

    Never appends `_MUST_EXPLORE_SECTION` — the arm has no explore guard —
    and drops the comparison-set sentence from the sampling note.
    """
    return FREEFORM_ARCHITECT_SYSTEM_PROMPT.format(
        **_freeform_format_kwargs(
            additional_instructions=additional_instructions,
            program_path=program_path,
            workspace_path=workspace_path,
            context=context,
        )
    )


# pareto_evolution: the freeform prompt plus the loop-owned-branch background
# and, on exploit-phase draws, the merge note. The git section is spliced with
# `str.replace` (not passed to `.format`) so its own `{child_branch}` slot is
# rendered by the single `.format` pass below.
_EVO_ARCHITECT_SYSTEM_PROMPT = FREEFORM_ARCHITECT_SYSTEM_PROMPT_BASE.replace(
    "{git_branching_section}", EVO_GIT_BRANCHING_SECTION + "\n"
)


def build_evo_architect_system_prompt(
    *,
    additional_instructions: str | None,
    program_path: str | None,
    workspace_path: str,
    context: IterationContext,
) -> str:
    """pareto_evolution system prompt (freeform-based).

    Identical to the freeform prompt except for `## Git Branch Background`
    (the loop forked `{child_branch}`; do not switch) and `EVO_MERGE_NOTE`
    when this iteration's exploit-and-merge draw fired. No explore/exploit
    guidance, no `_MUST_EXPLORE_SECTION`.
    """
    return _EVO_ARCHITECT_SYSTEM_PROMPT.format(
        **_freeform_format_kwargs(
            additional_instructions=additional_instructions,
            program_path=program_path,
            workspace_path=workspace_path,
            context=context,
        ),
        child_branch=(
            context.child_branch or context.start_branch or "(the current branch)"
        ),
        merge_note=_evo_merge_note(context.exploit_phase),
    )


def _subsample_note(
    n: int,
    agent_consumer: Literal["inspect_traces", "run_eval"] = "run_eval",
    *,
    include_comparison_note: bool = True,
    variance_estimate: Mapping[str, Any] | None = None,
    variance_estimate_samples: int = 5,
) -> str:
    """The sampling-variance guidance shared by the architect prompts.

    `include_comparison_note=False` drops the closing sentence about the
    comparison set (freeform gives no guidance on comparison-set usage).
    `variance_estimate` is the run's `run_variance_estimate` result when it
    has been taken (specs/archive-public/sampling_variance.md): the middle paragraph then
    reports the measured noise instead of recommending the tool.
    """
    comparison_note = (
        'Note: Running `run_eval(eval_sample="comparison")` runs the program '
        "on a larger, fixed comparison set which is a subsample of the "
        "training set. This is reserved for comparing branches "
        "apples-to-apples.\n"
        if include_comparison_note
        else ""
    )
    variance_paragraph = variance_note(
        variance_estimate, n, variance_estimate_samples
    )
    if agent_consumer == "inspect_traces":
        return f"""
The traces you are inspecting come from a data sample. Subsamples when small (e.g., < 100 rows) can be quite noisy depending on the application. Ultimately, the application is judged by its ability to generalize beyond the training set.

Provide key information on individual data points, but be careful not to make conclusions that over-rotate on noise.

"""
    else:
        return f"""\
### Sampling Guidance
When comparing eval results, be wary of variance between samples! The training data score is measured on a subsample of {n} rows randomly drawn from the training set. A random number seed ensures that the same sample is drawn within iterations (i.e., before/after delta is fair). However, across iterations and across branches, the seed is changed so you can learn from more examples across many iterations - but that introduces added variance. WARNING: do NOT compare across diferent training samples. While you can compare if the sample is the same, be careful NOT to over-rotate on scores, since the program itself may have some variance.

{variance_paragraph}

One common mistake is to design an improvement that focuses on fixing individual errors in the training subsample. That improvement of course easily performs well in one subsample, but then those improvements don't always generalize.

A better approach is to test generalizable approaches on the training subsample. That hypothesis can be informed by the training data but does not overrotate on one-off examples.

{comparison_note}"""


def _budget_note(
    max_eval_rows: int | None,
    evals_used: int,
    *,
    subsample_size: int = 0,
    comparison_set_size: int = 0,
) -> str:
    """Evaluation-budget awareness for the architect system prompt.

    Returns an empty string when no budget is configured (`max_eval_rows`
    falsy). The remaining count is a start-of-iteration snapshot — the prompt
    is built once at iteration start, so it reflects the budget before this
    iteration's evals run; every `run_eval` response carries the live
    `eval_rows_remaining`. States the per-eval-type row cost so the agent can
    plan the endgame instead of discovering a hard refusal mid-experiment.
    """
    if not max_eval_rows:
        return ""
    remaining = max(0, max_eval_rows - evals_used)
    pct = (evals_used / max_eval_rows) * 100
    note = (
        "### Eval-row budget (run-wide)\n"
        f"You have used {pct:.2f}% of the run's eval-row budget; "
        f"{remaining} of {max_eval_rows} rows remain (a start-of-iteration "
        "snapshot — every run_eval response carries the live "
        "`eval_rows_remaining`).\n"
    )
    if subsample_size > 0:
        cost = f"A train eval costs {subsample_size} rows"
        if comparison_set_size > 0:
            ratio = comparison_set_size / subsample_size
            cost += (
                f"; a comparison eval costs {comparison_set_size} rows "
                f"(\u2248{ratio:.0f}\u00d7 a train eval)"
            )
        cost += (
            "; `single` costs 1. That is room for "
            f"~{remaining // subsample_size} more train eval(s).\n"
        )
        note += cost
        if remaining < 2 * subsample_size:
            note += (
                "You do not have room for a before/after train pair. "
                "Make your change and validate it directly"
                + (
                    " on the comparison set"
                    if comparison_set_size > 0
                    else " with the rows you have left"
                )
                + " instead of spending the baseline first.\n"
            )
    return note + "\n"


# Shared description of an eval trace directory. Every prompt that points an
# agent at `iteration_{N}_{suffix}/` embeds this so the navigation advice
# cannot drift between the architect, the inspector, the analysis agent and
# the post-run auditors.
TRACE_DIR_LAYOUT_NOTE = """\
Layout of a trace directory (`iteration_<N>_<suffix>/`):
- `index.jsonl` — one short line per example: `example_id`, `score`,
  `failed`, `feedback`, `error_summary`, `row_file`.
- `index_pretty_print.txt` — the same per-row feedback, human-readable.
- `rows/<example_id>/row.json` — the example: `input`, `output`,
  `scores_by_metric`, `feedback`, `error_summary`, and a `spans` table
  (`k`, `signature_key`, `span_kind`, `parent_key`, `error`, `file`).
- `rows/<example_id>/span_NNN.json` — one module call (span) each:
  `signature_key`, `span_kind` (`"llm"`, `"tool"`, `"chain"`, `"retriever"`,
  `"function"`), `parent_key` (the enclosing span's `signature_key`),
  `inputs` (what the module received — for LLM spans usually `messages`),
  `output`, `error`, `metadata`.
- `provenance.json` — which eval produced this directory: `eval_id`,
  `sha_at_eval`, `tree`, `committed`, `branch`, `eval_sample`, `suffix`,
  `score`, `ts`. A directory is REWRITTEN by every eval with the same
  suffix, so a directory from an earlier iteration may describe a version
  that was later reverted — check the stamp before drawing conclusions.
- sibling file `../iteration_<N>_evals.jsonl` — one line per eval; quick
  score re-read.

Every file is pretty-printed JSON that fits in a single Read. A value shaped `{"__chunked_string__": ["...", "..."]}` is one long string split into pieces to fit line limits; read it as their concatenation."""


def build_trace_inspection_prompt(
    *,
    query: str,
    iteration_num: int,
    suffix: str = "child",
    subsample_size: int = 20,
) -> str:
    """Prompt for the `inspect_traces` sub-agent.

    `subsample_size` feeds the shared sampling note. Sampling note is customized for this inspect_trace agent, so the inspector knows the
    scores it reasons about could be noisy.
    """
    trace_dir = eval_trace_dir(traces_dir(), iteration_num, suffix)
    which = {
        "comparison": "the **comparison** trace: this iteration's latest "
        "comparison-set eval (check provenance.json for which version it "
        "describes)",
        "parent": "the **parent** trace: the iteration's BASELINE train eval, "
                  "before any change was made",
        "child": "the **child** trace: the latest train eval AFTER the "
                 "architect's change",
    }.get(suffix, f"the `{suffix}` trace")
    return f"""\
Analyze the full trace data based on the query provided and provide
detailed summaries based on the query, including any problems or issues
found. Preserve relevant excerpts from the logs so the architect that
consumes your summary has maximum context.

query: {query}

## Full Trace Data
Per-example execution traces are in the directory:
`{trace_dir}/`
This is {which}. State which trace you read at the top of your summary.

{TRACE_DIR_LAYOUT_NOTE}

### How to query trace data
1. Read `{trace_dir}/index.jsonl` (or `index_pretty_print.txt`, the
   readable version) to find failures (`"failed": true`) and low scores.
2. Open an example: `Read("{trace_dir}/rows/<example_id>/row.json")` —
   input, output, feedback, and the `spans` table.
3. Open the span that matters: `Read("{trace_dir}/rows/<example_id>/span_NNN.json")`
   (the `file` column of the `spans` table).
4. To search across examples: `Grep(pattern, path="{trace_dir}")` with
   `glob="*.json"`, then Read the files it names.
5. Compare the `inputs` / `output` of each span against what the metric's
   `feedback` says went wrong, to identify WHICH module produces poor
   intermediate outputs.

{_subsample_note(subsample_size, agent_consumer="inspect_traces")}
"""


MEMORY_MD_TEMPLATE = """\
# CodeEvolver Memory

Seed branch: {seed_branch}
Seed architecture: {architecture_title}

{pool_section}## Overall Progress
- Iteration 0 (Seed Program): (fill in)
- Iteration 1: (fill in)

## Macro Approaches
This section documents distinct, non-incremental strategies tried - i.e., macro approaches are fundamentally new solutions, usually different program architectures. Incremental edits (prompt tweaks, parameter changes, small feature additions) are intentionally excluded. 

### (fill in approach title here)
- Approach: (Fill in)
- Result: (Fill in)
- Observation: (Fill in)
- Remaining opportunity: (Fill in)

## Observations
(Fill in observations, reflections, memories, etc. here. No required format.)
No observations yet.

"""


_EE_CANDIDATE_POOL_SECTION = """\
## Candidate Pool
The branches still worth working from. One line per branch: name, what it is,
latest known scores. Maintain 2-5 branches; Stop tracking branches here once you no longer
plan to use them. 

- {seed_branch}: seed program

"""

# Which methods seed the macro-bearing template, and what fills its pool
# slot. `ee` and `asa` explore, so they keep `## Macro Approaches` and the
# agent-managed `## Candidate Pool`. `freeform` and
# `pareto_evolution` carry no exploration guidance and use the macro-free
# template — evo's population is loop-owned (`population.json`), so its
# memory has no pool section either.
_POOL_SECTION_BY_METHOD: dict[str, str] = {
    "ee": _EE_CANDIDATE_POOL_SECTION,
    "asa": _EE_CANDIDATE_POOL_SECTION,
}

MEMORY_MD_TEMPLATE_NO_MACRO = """\
# CodeEvolver Memory

Seed branch: {seed_branch}
Seed architecture: {architecture_title}

## Overall Progress
- Iteration 0 (Seed Program): (fill in)
- Iteration 1: (fill in)

## Observations
(Fill in observations, reflections, memories, etc. here. No required format.)
No observations yet.

"""


def build_memory_md_template(
    *,
    seed_branch: str,
    architecture_title: str,
    method: str = "ee",
) -> str:
    """Starter `/memory/memory.md`.

    Engine 3: the architect writes memory.md directly every iteration (no
    reflection agent digests this file). `method` selects the template: `ee` / `asa` get the
    macro-bearing MEMORY_MD_TEMPLATE with the Candidate Pool section; every
    other method (`freeform`, `pareto_evolution`) gets the macro-free one.
    """
    if method in _POOL_SECTION_BY_METHOD:
        pool = _POOL_SECTION_BY_METHOD[method].replace("{seed_branch}", seed_branch)
        template = MEMORY_MD_TEMPLATE.replace("{pool_section}", pool)
    else:
        template = MEMORY_MD_TEMPLATE_NO_MACRO
    return template.format(
        seed_branch=seed_branch,
        architecture_title=architecture_title,
    )


def build_analysis_agent_prompt(
    *,
    iteration_num: int,
    workspace_path: str,
    branch: str | None,
    parent_sha: str | None = None,
    close_sha: str | None = None,
    change_request: str,
    subsample_baseline: float | None,
    child_trace_dir: str,
    parent_trace_dir: str,
    subsample_size: int = 20,
    changes_landed: bool = True,
    subsample_best: float | None = None,
    eval_series: list[tuple[str | None, float | None]] | None = None,
    attempted_diff_path: str | None = None,
) -> str:
    """Prompt for the per-iteration analysis agent (engine 3).

    The architect writes its own outcome reflection directly into memory.md,
    so this agent does not touch memory.md at all — its single job is the concise
    per-iteration change summary used to build
    `iteration_change_summaries.md`.

    Runs for landing AND non-landing iterations. When nothing landed, the
    summary is a negative result: what was built, what it measured, why it was
    backed out. `attempted_diff_path` carries the actual code that was tried —
    the agent has no Bash tool, so it cannot produce that diff itself, and for
    a reverted change there is no commit to diff against anyway.

    This agent also OWNS `codeevolver.md`: when a change landed
    it rewrites the architecture description in place so the file describes
    the program as it now exists on the branch. The architect is told not to
    touch the file; the loop commits whatever this agent changed.
    """
    _fmt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"
    mem = memory_dir()

    if attempted_diff_path:
        diff_line = (
            f"- Diff of the change this iteration built and measured "
            f"(parent -> best-scoring version tried): `{attempted_diff_path}` "
            f"— Read it; this is the primary evidence of what changed."
        )
    else:
        diff_line = (
            "- No attempted-change diff is available for this iteration "
            "(nothing was edited, or the diff could not be produced)."
        )

    if changes_landed:
        codeevolver_md_step = (
            f"Update `{workspace_path}/codeevolver.md` IN PLACE so it describes the "
            "program as it now exists on this branch. Edit the architecture "
            "description to reflect the change that landed; keep the file's "
            "`## ARCHITECTURE TITLE:` / `## ARCHITECTURE SUMMARY:` / "
            "`## ARCHITECTURE DESCRIPTION:` headers exactly (they are parsed). "
            "It is a description of the current program, NOT a changelog: no "
            "iteration history, no 'change summary' sections, no notes to future "
            "agents — and delete any stale `## Change summary (iteration N)` "
            "blocks you find. Do not edit any other file in the repository."
        )
    else:
        codeevolver_md_step = (
            f"Leave `{workspace_path}/codeevolver.md` unchanged — the program on "
            "this branch is identical to its parent, so the description is "
            "still accurate."
        )

    if eval_series:
        series = ", ".join(
            f"{sample or '?'}={_fmt(score)}" for sample, score in eval_series
        )
    else:
        series = "(none recorded)"

    if changes_landed:
        outcome_block = f"""This iteration LANDED a change: the program on `{branch or "(unknown)"}` differs from its parent.
Write a summary of the core change that shipped, then update `{workspace_path}/codeevolver.md` (step 4)."""
        title_rule = ""
    else:
        outcome_block = f"""This iteration LANDED NOTHING: the architect built a change, measured it, and left the program identical to its parent (or made no change at all). This is a NEGATIVE RESULT, and it is just as important to record as a success — future iterations read these summaries to avoid re-trying exhausted levers.
Write a summary of what was TRIED, what it MEASURED, and why it was backed out. Be specific about the mechanism of failure if the traces show one."""
        title_rule = (
            "- Begin the file with the single line `[REVERTED]` before the summary text.\n"
        )

    return f"""Instructions:
Read {workspace_path}/codeevolver.md to understand the system architecture, then write a concise change summary for iteration {iteration_num}.

{outcome_block}

Steps:
1. The objective iteration record is in the Context section below (branch, shas, score series, change request).
2. Read the attempted-change diff named in the Context — that is what actually changed. For behavioural detail, this iteration's training-subsample traces are in `{child_trace_dir}/` and the baseline's in `{parent_trace_dir}/`: `index.jsonl` lists every example (score, feedback, `row_file`); `rows/<example_id>/row.json` holds the example's input, output and a `spans` table; each `span_NNN.json` next to it is one module call (inputs, output, metadata). Every file fits a single Read; a string that appears as `{{"__chunked_string__": [...]}}` is one long string split into pieces — read it as their concatenation.
3. Write your summary of changes to `{mem}/iteration_{iteration_num}_change_summary.md`.
4. {codeevolver_md_step}

What to save to iteration_{iteration_num}_change_summary.md:
{title_rule}- A concise executive summary of the core change made in this iteration. e.g., "Converted to ReAct agent with retrieval tool and added examples to the prompt." 10-200 words. No more.
- Nothing else. Super minimal file.
- These change summary files are used to automatically generate iteration_change_summaries.md, which is provided to future architect iterations alongside memory.md.

Do NOT edit `{mem}/memory.md` — the architect maintains that file itself.

Context:
- Iteration number: {iteration_num}
- Branch: {branch or "(unknown)"}
- Parent sha: {parent_sha or "(unknown)"} -> closing sha: {close_sha or "(unknown)"}
- Program changed this iteration: {"yes" if changes_landed else "NO — reverted or never changed"}
- Training subsample: baseline {_fmt(subsample_baseline)} -> best version tried this iteration {_fmt(subsample_best)} (same {subsample_size} rows; noisy — do not over-interpret). The closing commit itself may not have been re-measured; the score describes the iteration's progress, not the exact HEAD.
- All evals this iteration, in order: {series} (`variance=` entries, if any, are the run_variance_estimate tool's repeated runs on a separate fixed sample — noise readings, not before/after measurements)
{diff_line}
- Change request(s) recorded this iteration:
```
{change_request or "(none recorded — the architect edited directly; read the attempted-change diff)"}
```
- Post-change trace directory: `{child_trace_dir}`
- Baseline trace directory: `{parent_trace_dir}`
"""


__all__ = [
    "ARCHITECT_SYSTEM_PROMPT",
    "ARCHITECT_SYSTEM_PROMPT_BASE",
    "EE_GIT_BRANCHING_SECTION",
    "EE_PROMPT_SECTIONS",
    "EVO_GIT_BRANCHING_SECTION",
    "EVO_MERGE_NOTE",
    "PromptSections",
    "SystemPromptBuilder",
    "make_architect_prompt_builder",
    "render_architect_prompt_template",
    "FREEFORM_ARCHITECT_SYSTEM_PROMPT",
    "FREEFORM_ARCHITECT_SYSTEM_PROMPT_BASE",
    "MEMORY_MD_TEMPLATE",
    "MEMORY_MD_TEMPLATE_NO_MACRO",
    "build_architect_system_prompt",
    "build_evo_architect_system_prompt",
    "build_freeform_architect_system_prompt",
    "build_memory_md_template",
    "build_trace_inspection_prompt",
    "build_analysis_agent_prompt",
]
