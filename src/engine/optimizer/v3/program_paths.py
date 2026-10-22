"""Copyright © 2026 440 Labs LLC

Program vs. non-program content identity (engine 3).

An iteration's identity is the content of its *program*, not the content of
its repository. The architect is instructed to record a change summary in
`codeevolver.md` as part of closing, so the commit it closes on routinely
differs from the tree it last evaluated by a documentation file alone. Treating
that as "a different program" would make the close gate demand a redundant
full eval of byte-identical code and raise spurious revert-check warnings.

This module is the single place that answers "are these two trees the same
program?". Three consumers share it:

1. the `close_iteration` gate (`agent_tools.py` check #5) — may HEAD close?
2. final-score attribution (`IterationContext.finalize`) — which eval scored
   the code that actually landed?
3. `changes_landed` — did this iteration alter the program at all?

The list is a fixed engine list rather than client-configurable on purpose: a
widened list is a way to close on unevaluated code, so it should not be
reachable from a run config.
"""

from __future__ import annotations

import fnmatch

from src.engine.services.subprocess_service import git_diff_tree_names

# Exact repo-relative paths that never constitute a program change.
NON_PROGRAM_PATHS: frozenset[str] = frozenset({
    "codeevolver.md",
    ".gitignore",
})

# Glob patterns matched against ROOT-LEVEL paths only (no "/" in the path).
#
# Deliberately NOT a blanket root-level `*.md`. A client whose program reads a
# root-level markdown prompt at runtime — `PROMPT.md`, `SYSTEM.md` — would get
# exactly the hole this module exists to prevent: the architect rewrites the
# prompt, check #5 waives the re-eval, and the pre-change score is attributed
# to the changed program. `prompts/system.md` is already treated as program
# content, and `./system.md` is the same file one directory up.
#
# Only names that cannot plausibly be read by a program are waived.
NON_PROGRAM_GLOBS: tuple[str, ...] = (
    "README*.md",
    "CHANGELOG*.md",
    "CONTRIBUTING*.md",
    "LICENSE*",
)


def is_non_program_path(path: str) -> bool:
    """True iff `path` (repo-relative) is documentation/meta, not program code."""
    if not path:
        return False
    path = path.strip()
    # Strip a leading "./" as a prefix — never as a character set, or
    # ".gitignore" would become "gitignore".
    if path.startswith("./"):
        path = path[2:]
    if path in NON_PROGRAM_PATHS:
        return True
    if "/" in path:
        # Only root-level files are waived by glob; a nested .md may be a
        # prompt, a fixture, or a doc the program reads at runtime.
        return False
    return any(fnmatch.fnmatch(path, pat) for pat in NON_PROGRAM_GLOBS)


def diff_is_non_program_only(workspace: str, a: str | None, b: str | None) -> bool:
    """True iff `a` and `b` are the same *program*.

    Identical revs, and revs whose diff touches only non-program paths, are
    program-equivalent. Returns False when either rev is missing or git cannot
    answer — an unknown diff must never be waived, so callers degrade to exact
    equality rather than to "equivalent".
    """
    if not a or not b:
        return False
    if a == b:
        return True
    names = git_diff_tree_names(workspace, a, b)
    if names is None:
        return False
    return all(is_non_program_path(name) for name in names)


def program_diff_paths(workspace: str, a: str | None, b: str | None) -> list[str]:
    """The program (non-waived) paths differing between `a` and `b`.

    Used to make the close-gate rejection actionable — the architect is told
    which files still need evaluating, not just that HEAD is unevaluated.
    """
    if not a or not b or a == b:
        return []
    names = git_diff_tree_names(workspace, a, b)
    if names is None:
        return []
    return [name for name in names if not is_non_program_path(name)]


__all__ = [
    "NON_PROGRAM_GLOBS",
    "NON_PROGRAM_PATHS",
    "diff_is_non_program_only",
    "is_non_program_path",
    "program_diff_paths",
]
