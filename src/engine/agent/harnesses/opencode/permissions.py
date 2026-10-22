"""OpenCode permission config builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_architect_permission_config(deny_paths: list[str]) -> dict[str, Any]:
    """Permission config for the OpenCode architect session.

    The architect drives its iteration with full edit/write/bash, but — like
    the coding sub-agent — it must never read or edit the holdout ``deny_paths``
    (valset / held-out test set and their trace files). The Claude harness
    enforces the same fence via its ``_make_valset_guard_hook`` PreToolUse hook;
    here it is expressed as read/edit deny patterns. The config is only honored
    when the caller also exports it as ``OPENCODE_PERMISSION`` (see
    ``OpenCodeHarness._build_architect_env_overrides``).
    """
    deny_patterns = [f"*{fragment}*" for fragment in deny_paths if fragment]
    read_rules: dict[str, str] = {"*": "allow"}
    edit_rules: dict[str, str] = {"*": "allow"}
    for pattern in deny_patterns:
        read_rules[pattern] = "deny"
        edit_rules[pattern] = "deny"
    return {
        "read": dict(read_rules),
        "grep": dict(read_rules),
        "glob": dict(read_rules),
        "list": "deny",
        "edit": dict(edit_rules),
        "write": dict(edit_rules),
        "bash": "allow",
        "task": "deny",
        "question": "deny",
        "webfetch": "deny",
        "websearch": "deny",
        # apply_patch is gpt-5.x-codex's preferred edit tool. OpenCode routes it
        # through the `edit` permission, so mirror the edit rule (allow with the
        # holdout deny patterns) rather than a blanket deny — otherwise the model
        # wastes a turn discovering apply_patch is blocked and falls back to the
        # coding agent. The valset/test fence is preserved via the deny patterns,
        # and the pre-edit guard additionally parses patch target paths.
        "apply_patch": dict(edit_rules),
        "external_directory": dict(read_rules),
        "doom_loop": "deny",
        "skill": "allow",
        "lsp": "deny",
        "todowrite": "deny",
    }


def build_coding_permission_config(deny_paths: list[str]) -> dict[str, Any]:
    """Permission config for the OpenCode coding sub-agent.

    The coding agent needs full edit/write/bash to make and commit changes, so
    this mirrors the permissive architect stance — but it additionally fences
    the holdout `deny_paths` (valset / held-out test set and their trace files)
    out of every read- and write-type tool. Without this the coding agent could
    read holdout data during a mutation; the Claude harness already enforces the
    same fence via its PreToolUse deny-path hook.

    bash stays allowed (the agent commits via git), matching the architect
    config's accepted trade-off; the holdout files are still removed from the
    read/grep/glob/edit/write/external surfaces the agent normally uses.
    """
    deny_patterns = [f"*{fragment}*" for fragment in deny_paths if fragment]

    read_rules: dict[str, str] = {"*": "allow"}
    edit_rules: dict[str, str] = {"*": "allow"}
    for pattern in deny_patterns:
        read_rules[pattern] = "deny"
        edit_rules[pattern] = "deny"

    return {
        "read": dict(read_rules),
        "grep": dict(read_rules),
        "glob": dict(read_rules),
        "list": "allow",
        "edit": dict(edit_rules),
        "write": dict(edit_rules),
        "bash": "allow",
        "task": "deny",
        "question": "deny",
        "webfetch": "deny",
        "websearch": "deny",
        "apply_patch": "deny",
        "external_directory": dict(read_rules),
        "doom_loop": "deny",
        "skill": "allow",
        "lsp": "allow",
        "todowrite": "deny",
    }


def build_inspection_permission_config(
    workspace: str | Path,
    *,
    writable: bool,
    deny_paths: list[str],
    allow_bash: bool = False,
) -> dict[str, Any]:
    from src.engine.paths import memory_dir, traces_dir

    deny_patterns = [f"*{fragment}*" for fragment in deny_paths if fragment]
    memory_pattern = str(Path(memory_dir()) / "**")
    traces_pattern = str(Path(traces_dir()) / "**")
    readable_rules: dict[str, str] = {
        "*": "allow",
        memory_pattern: "allow",
        traces_pattern: "allow",
    }
    for pattern in deny_patterns:
        readable_rules[pattern] = "deny"

    # Unlike readable_rules above, this had no "*" base -- any access outside
    # memory_dir/traces_dir (e.g. a glob tool resolving its scan root to `/`
    # or `/workspace`) fell through to OpenCode's own default for an
    # unmatched external_directory pattern, which is an interactive "ask".
    # That's unanswerable in a headless `opencode run`, so the session hangs
    # forever. Matching readable_rules' shape closes the gap.
    external_directory: dict[str, str] = {
        "*": "allow",
        memory_pattern: "allow",
        traces_pattern: "allow",
    }
    for pattern in deny_patterns:
        external_directory[pattern] = "deny"

    edit: str | dict[str, str]
    write: str | dict[str, str]
    if writable:
        # Allow-by-default with holdout denies — the same stance the architect
        # and coding agents use (build_architect/coding_permission_config), which
        # write reliably. A prior deny-by-default whitelist ({"*": "deny",
        # "{memory}/**": "allow"}) silently broke writes: OpenCode routes
        # apply_patch THROUGH the `edit` permission ("edit covers edit/write/
        # patch"), and with edit defaulting to deny the write hinged entirely on
        # the {memory}/** glob matching the direct child memory.md — which it did
        # not, so the gpt-5.x-codex apply_patch calls were denied and the agent
        # produced no writes. read/grep/glob were unaffected because they default
        # to allow. The agent is still confined: cwd = the memory dir (see
        # OpenCodeHarness.run_inspection) and bash is denied, so it cannot roam,
        # and the holdout deny_patterns (last-match-wins) still fence valset/test.
        edit = {"*": "allow"}
        for pattern in deny_patterns:
            edit[pattern] = "deny"
        write = dict(edit)
        # apply_patch is governed by the `edit` rule above; this rule is inert
        # but kept explicit so the policy reads consistently for codex models.
        apply_patch = "allow"
    else:
        edit = "deny"
        write = "deny"
        apply_patch = "deny"

    return {
        "read": readable_rules,
        "grep": dict(readable_rules),
        "glob": dict(readable_rules),
        "list": "allow",
        "edit": edit,
        "write": write,
        "apply_patch": apply_patch,
        # Bash is the clean-slate inspector's exception (allow_bash=True): its
        # job is reading git history, which read/grep/glob cannot answer. Every
        # other inspection caller keeps the deny. Note OpenCode does NOT filter
        # bash by the deny patterns, so the caller is responsible for proving
        # the session was read-only -- see clean_slate.py::_snapshot_repo.
        "bash": "allow" if allow_bash else "deny",
        "task": "deny",
        "question": "deny",
        "webfetch": "deny",
        "websearch": "deny",
        "external_directory": external_directory,
        "doom_loop": "deny",
        "skill": "allow",
        "lsp": "allow",
        "todowrite": "deny",
    }
