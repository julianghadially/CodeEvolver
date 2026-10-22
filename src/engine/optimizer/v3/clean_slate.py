"""Copyright © 2026 440 Labs LLC

Start-of-run clean-slate inspection.

A read-only "scientist" agent that runs in the SEED step — after the run
branch is created and the holdout deny paths are known, but before the
environment smoke test — and verifies the filesystem provided for this run
is uncontaminated. It layers an agentic judgment on top of the Docker runner's
deterministic controls (fresh container per run, history-depth-1 assert,
preflight egress canary): those catch known channels; this agent looks for
the ones nobody predicted.

It checks three things (spec: specs/engine3.md):
  1. Git history — no old solutions or run artifacts beyond the main program
     provided (prior ``codeevolver-*`` work, graded result files, old run
     outputs committed into history).
  2. Data folders — every holdout/test data file is covered by the deny
     paths; flags test-looking data files that are not fenced.
  3. Container leftovers — if running in a Docker container, folders left by
     prior agents that could contaminate this run (old-run data, memory,
     notes, ideas, branches).

OPT-IN and ABORT-CAPABLE: gated by ``enable_clean_slate_inspector`` (default
False everywhere). An explicit ``"contaminated"`` verdict raises
``CleanSlateContaminationError``, aborting the run before any eval spend —
by user decision, a contaminated 6-day research run is worse than a
false-positive abort. Everything else fails OPEN: agent errors, missing
output, an ``"uncertain"`` verdict (agent found something it could not
confidently classify — findings recorded, warning logged), or an unparseable
verdict all log a warning and let the run proceed.

The agent itself runs with the holdout deny paths ARMED — unlike the
run-integrity inspector, it has no reason to read test-set contents; it only
reasons about which files exist and whether they are fenced.

It is the only MID-RUN inspection caller granted a Bash tool
(``allow_bash=True``; the post-run meta-reflection and run-integrity auditors
also get one, without a mutation guard, because the run is over by then).
Checks 1 and 3 are unanswerable without one: ``git log`` / ``branch -a`` /
``reflog`` / ``fsck`` have no Read/Grep/Glob equivalent, and a deleted
``codeevolver-*`` branch leaves its commits reachable only through the reflog.
A verdict from an agent that cannot read git history is not worth acting on.
The shell is read-only *by contract*, and the contract is enforced rather than
trusted: ``_snapshot_repo`` is taken before and after the session and any
difference in HEAD, working tree, refs, reflog, or untracked/ignored files
raises ``CleanSlateMutationError``.

Its report goes to ``paths.clean_slate_report_path()`` —
``<logs_dir>/<job_id>_clean_slate_report.md`` — deliberately NOT the memory
dir, which is the directory the architect is told to read. Unlike the post-loop
run-integrity report, this one exists before the architect's first turn, so a
finding that quoted a data file would be one ``Read`` away. ``logs/`` is
collected by ``orchestrator/collect.py`` exactly like memory, so the report is
no less durable. ``clean_slate_report`` remains a holdout deny fragment
(belt-and-braces) and the prompt forbids quoting data-file contents.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.engine.paths import clean_slate_report_path

logger = logging.getLogger(__name__)

CLEAN_SLATE_STATUSES = ("clean", "contaminated", "uncertain")

# The inspector's report lands in the memory dir, beside memory.md — which the
# architect is explicitly told to read, and which it may read freely (its
# permissions are allow-by-default with holdout deny patterns). Unlike the
# run-integrity report, which is written after the loop has finished, this file
# exists before the architect's first turn, so it is a live channel: anything
# the inspector quoted while walking the repo with its shell would be one Read
# away. Fence it by name for every agent. Substring-matched, so it covers the
# `.md` suffix and any absolute prefix.
CLEAN_SLATE_REPORT_DENY_FRAGMENT = "clean_slate_report"

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*\n(\{.*?\})\s*\n```", re.DOTALL)


@dataclass
class CleanSlateContaminationError(RuntimeError):
    """Raised when the clean-slate inspector returns a `contaminated` verdict.

    Carries the inspector's findings so the entry-point handler can surface
    them structurally, mirroring `SandboxValidationError`.
    """

    message: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    report_path: str | None = None

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "findings": list(self.findings),
            "report_path": self.report_path,
        }


@dataclass
class CleanSlateMutationError(RuntimeError):
    """Raised when the clean-slate inspector changed the workspace.

    The inspector holds a Bash tool so it can read git history, and its prompt
    says the shell is read-only. This is what makes that a contract rather than
    a hope: an inspector that mutated the repository has already destroyed the
    clean slate it was asked to certify, so the run stops. It fires in the seed
    step, before any eval spend.
    """

    message: str
    changed: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {"message": self.message, "changed": list(self.changed)}


def _snapshot_repo(workspace: str) -> dict[str, str]:
    """Fingerprint everything a read-only inspection must leave untouched.

    Five keys, because no one of them is sufficient: a commit moves HEAD but
    not necessarily the tree; `git stash` moves the tree but not HEAD; a branch
    can be created without touching either; `gc`/`reflog expire` rewrites
    neither refs nor tree; and dropping a scratch file touches none of the
    four.

    `git_tree_signature` is used rather than `status --porcelain` alone because
    it hashes CONTENT -- an edit-and-revert-to-different-bytes shows up. It
    writes loose objects through a throwaway index by design (see
    specs/engine3.md "Eval identity is the git tree"), which is not a mutation
    of any of the five things measured here, so calling it does not trip the
    guard it feeds.

    Known gap: a new file created INSIDE an already-ignored directory is not
    detected. `git add -A` honours .gitignore so the tree hash misses it, and
    `status --ignored` collapses an ignored directory to one entry. Listing
    them individually (`--ignored=matching`) would enumerate every file in the
    client's `env/` venv, on both snapshots, on every run -- a real cost for a
    narrow gap that is already covered elsewhere: the workspace is
    per-container and discarded, and `/artifacts` is the only path outliving a
    run (specs/experiment-docs/experiment-fleet.md "Isolation between runs").
    """
    from src.engine.services.subprocess_service import (
        git_capture,
        git_tree_signature,
    )

    try:
        tree = git_tree_signature(workspace) or ""
    except OSError:  # workspace missing entirely (synthetic test envs)
        tree = ""

    return {
        "head": git_capture(workspace, "rev-parse", "HEAD").strip(),
        "tree": tree,
        "refs": git_capture(workspace, "show-ref"),
        "reflog": git_capture(workspace, "reflog", "--format=%H", "-1").strip(),
        "status": git_capture(workspace, "status", "--porcelain", "--ignored"),
    }


def _assert_unmutated(before: dict[str, str], after: dict[str, str]) -> None:
    """Raise `CleanSlateMutationError` if any snapshot key moved."""
    changed = [key for key in before if before[key] != after.get(key)]
    if not changed:
        return
    detail = "; ".join(
        f"{key}: {(before[key] or '(empty)')[:120]!r} -> "
        f"{(after.get(key) or '(empty)')[:120]!r}"
        for key in changed
    )
    raise CleanSlateMutationError(
        message=(
            "clean-slate inspector MUTATED the workspace (it is contractually "
            f"read-only) — aborting. Changed: {detail}"
        ),
        changed=changed,
    )


def build_clean_slate_prompt(
    *,
    workspace: str,
    initial_branch: str,
    deny_paths: list[str],
    testset_rel_path: str | None,
) -> str:
    """Build the clean-slate inspector prompt."""
    deny_list = "\n".join(f"  - `{p}`" for p in deny_paths) or "  - (none configured)"
    testset_line = (
        f"- The held-out test set is expected at `{testset_rel_path}` "
        "(relative to the repo)."
        if testset_rel_path
        else "- No test-set path was configured for this run."
    )

    return f"""You are a computer scientist working on a research project. A new \
self-improving AI optimization run is about to start in the repository at \
`{workspace}`, and the filesystem provided to it must be CLEAN — any leftover \
solutions, data, or notes from earlier runs would contaminate this treatment \
group and invalidate the experiment. Your job is to inspect and report; you \
must not modify, delete, or create anything.

## Your shell is READ-ONLY, and this is enforced
You have a Bash tool because `git log`, `git branch -a`, `git reflog` and `git fsck` are the only way to answer the questions below. It is for READING ONLY. The workspace is fingerprinted before and after this session, so NEVER write a file and don't modify the git - e.g., `git checkout`, `git switch`, `git reset`, `git stash`, `git gc`, `git add`, `git commit`, `rm`, `mv`, `touch`, etc.

You should have access to only one branch — the initial seed program. That seed program generally contains the program itself, a training data set, and a holdout testing data set. Each benchmark is a little different. Some benchmarks contain code for generating a retrieval database, or other files that are part of the benchmark preparation process, and that may be okay, depending on the criteria below. A repository may also host SEVERAL benchmarks — another benchmark's train/test data is normal and is NOT contamination for this run, which optimizes only the program named above.

## Criteria for contamination
The following file content is usually a sign of contamination. Use your best judgment.
- Any content that can be unfairly used to improve the program.
- Any content (solutions, data, memory, ideas, branches, etc.) that is clearly left behind by a previous optimization run (in git, optimization branches always contain the codeevolver- prefix in the branch name.)
- Holdout test data that is not paired with a deny path (See deny list below).


The following file content is usually acceptable, but use your best judgment.
- Codebases naturally contain documentation, readmes, example files, evaluation scripts, test cases, etc. This is usually fair to include in the repository.


## Inspect three things

1. **Git history.** Inspect the repository's history (`git log --oneline \
--decorate -50`, `git log --stat`, `git branch -a`, `git show --stat \
<sha>`) and confirm it holds no old solutions beyond the main program \
provided on `{initial_branch}`. Deleting a branch does NOT remove its \
commits, so also check the channels that survive deletion: `git reflog` \
and `git fsck --unreachable --no-progress`. A `codeevolver-` branch name \
in the reflog, or an unreachable commit whose message reads like an \
optimization iteration, means a previous run's evolved program is still \
recoverable from this repository.

2. **Repo data folders.** Enumerate the data files that actually exist \
(`ls data/`, `find . -name "*.json" -o -name "*.jsonl" -o -name "*.csv"`, \
skipping any venv/cache directories) and check each against the deny list \
below. Report NAMES, PATHS and ROW/BYTE COUNTS ONLY — never the contents of \
a data file, and never a sample row (your report is written to a file the \
optimizing agent can reach, so quoting data would be the very leak you are \
here to prevent). Flag any test-looking file for THIS benchmark \
(test/holdout/eval splits, or graded result files carrying answers or \
scores) that is not covered by a deny-path fragment. Another benchmark's \
data is not a finding.

3. **Container workspace.** If you are running inside a Docker container (or \
any dedicated run environment), check whether prior agents left ideas or \
documents that could contaminate this run: any OTHER folder you can access \
that contains data from old runs, memory from old runs, notes, ideas, or \
branch listings — e.g. stale `~/.codeevolver/` contents (memory/, traces/, \
history/ from other jobs), stray files in the home directory or /tmp, or \
sibling directories next to the workspace. This run's OWN job directory is \
expected to exist and be empty or near-empty; only OTHER jobs' directories \
are suspect.

## Deny paths
The run fences these path fragments from its agents (deny paths):
{deny_list}
{testset_line}

## Produce (markdown)

A findings report: what you inspected, what you found, and why each finding \
does or does not contaminate the run. Then end with EXACTLY ONE fenced JSON \
block of this shape:

```json
{{"status": "clean", "findings": [{{"path": "<abs or repo-relative path>", \
"reason": "<why this contaminates the run>"}}]}}
```

`status` must be `"clean"`, `"contaminated"`, or `"uncertain"`. Use \
`"contaminated"` ONLY for findings that could genuinely leak prior solutions \
or holdout knowledge into this run — a contaminated verdict ABORTS the run. \
Use `"uncertain"` when you found something you cannot confidently classify \
either way (the run proceeds, but your findings are recorded and surfaced to \
the operator — list every unclear item in `findings`). Empty-workspace noise \
(caches, venvs, lockfiles) is not contamination; list such items as findings \
with a benign reason under a `"clean"` status if worth noting.
"""


def parse_clean_slate_verdict(markdown: str | None) -> dict | None:
    """Extract the final fenced JSON verdict. Returns None when unparseable."""
    if not markdown:
        return None
    for match in reversed(_FENCED_JSON_RE.findall(markdown)):
        try:
            data = json.loads(match)
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict) or "status" not in data:
            continue
        status = str(data.get("status") or "").strip().lower()
        if status not in CLEAN_SLATE_STATUSES:
            status = "unparsed"
        findings = data.get("findings")
        if not isinstance(findings, list):
            findings = []
        return {
            "status": status,
            "findings": [f for f in findings if isinstance(f, dict)],
        }
    return None


def run_clean_slate_inspection(
    *,
    workspace: str,
    initial_branch: str,
    deny_paths: list[str],
    testset_rel_path: str | None,
    agent_model: str | None = None,
    coding_harness: str = "claude",
    coding_harness_model: str | None = None,
    max_turns: int = 25,
) -> str:
    """Run the clean-slate inspector once. Returns the recorded status.

    Raises ``CleanSlateContaminationError`` on an explicit ``contaminated``
    verdict, and ``CleanSlateMutationError`` if the session changed the
    workspace. Fails OPEN on everything else (agent error, no output,
    unparseable verdict): logs a warning and returns the status string so the
    seed record can carry it (``"clean"`` | ``"unparsed"`` | ``"error"``).
    """
    report_path = clean_slate_report_path()
    # Taken outside the try so a fail-open path cannot skip the comparison —
    # this agent holds a shell, and the guard is what makes that safe.
    before = _snapshot_repo(workspace)
    if not any(before.values()):
        # Every probe came back empty: not a readable git repo, so the guard
        # is inert. Say so rather than letting an unenforced control read as
        # an enforced one.
        logger.warning(
            "clean-slate: %s is not a readable git repository — the "
            "read-only mutation guard cannot be enforced for this session",
            workspace,
        )
    analysis = ""
    failed_open = False
    try:
        from src.engine.services.subprocess_service import (
            run_inspection_agent,
            write_file,
        )

        prompt = build_clean_slate_prompt(
            workspace=workspace,
            initial_branch=initial_branch,
            deny_paths=deny_paths,
            testset_rel_path=testset_rel_path,
        )
        logger.info("clean-slate: starting inspection of %s", workspace)
        result = run_inspection_agent(
            workspace=workspace,
            prompt=prompt,
            max_turns=max_turns,
            writable=False,
            deny_paths=list(deny_paths),
            model=agent_model,
            coding_harness=coding_harness,
            coding_harness_model=coding_harness_model,
            # The one inspection caller with a shell — see the module
            # docstring. Paid for by `_assert_unmutated` below.
            allow_bash=True,
        )
        analysis = (result or {}).get("proposed_change") or ""
        if not analysis.strip():
            logger.warning(
                "clean-slate: agent produced no output (%s) — failing open",
                (result or {}).get("error"),
            )
            failed_open = True
        else:
            write_file(report_path, analysis)
            logger.info("clean-slate: wrote report to %s", report_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("clean-slate inspection failed (fail-open): %s", exc)
        failed_open = True

    # Runs on EVERY exit path, fail-open ones included: a session that errored
    # out after `git checkout` still left the workspace wrong, and a corrupted
    # clean slate is not something to shrug at just because the agent also
    # crashed. Raises CleanSlateMutationError, deliberately outside the
    # fail-open handler above.
    _assert_unmutated(before, _snapshot_repo(workspace))

    if failed_open:
        return "error"

    verdict = parse_clean_slate_verdict(analysis)
    if verdict is None:
        logger.warning("clean-slate: no parseable verdict — failing open")
        return "unparsed"
    if verdict["status"] == "uncertain":
        summary = "; ".join(
            f"{f.get('path', '?')}: {f.get('reason', '')}"
            for f in verdict["findings"][:5]
        )
        logger.warning(
            "clean-slate: UNCERTAIN verdict (run proceeds). Findings: %s",
            summary or "(see report)",
        )
        return "uncertain"
    if verdict["status"] == "contaminated":
        findings = verdict["findings"]
        summary = "; ".join(
            f"{f.get('path', '?')}: {f.get('reason', '')}" for f in findings[:5]
        )
        # The one deliberate abort: an explicit contaminated verdict.
        raise CleanSlateContaminationError(
            message=(
                "clean-slate inspector found contamination — aborting before "
                f"any eval spend. Findings: {summary or '(see report)'}"
            ),
            findings=findings,
            report_path=report_path,
        )
    logger.info("clean-slate: verdict=%s", verdict["status"])
    return verdict["status"]


__all__ = [
    "CLEAN_SLATE_REPORT_DENY_FRAGMENT",
    "CLEAN_SLATE_STATUSES",
    "CleanSlateContaminationError",
    "CleanSlateMutationError",
    "build_clean_slate_prompt",
    "parse_clean_slate_verdict",
    "run_clean_slate_inspection",
]
