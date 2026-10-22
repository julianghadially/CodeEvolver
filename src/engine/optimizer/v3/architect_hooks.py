"""Copyright © 2026 440 Labs LLC

PreToolUse hook factories for the IterationArchitect session.

These are policy fences enforced by the Claude Agent SDK before any
builtin tool call. They're factored out of `iteration_architect.py` so
the class file stays focused on session orchestration; the hooks are pure
functions of their inputs and have no class coupling.

  - `_make_holdout_guard_hook` — block any Read/Edit/Write/Grep/Glob/Bash
    that references a holdout file. The architect must never touch
    holdout data directly; the only authorized path is `run_eval`, which
    dispatches through `subprocess_service` outside the SDK hook system.
  - `check_pre_edit_guard_decision` — an `apply_patch` holdout fence. Its
    targets live inside `patchText` rather than a `file_path` argument, so the
    holdout guard above cannot see them.

There is deliberately NO ordering control here (e.g. blocking
`Edit`/`Write` until a baseline eval has run). Enforcing that at the tool layer
means enumerating the ways a file can change — which leaks through Bash, a
primary edit path for current models, and applies friction asymmetrically
depending on which tool a model happens to prefer. In a cross-model, cross-arm
experiment that is a harness-introduced confound, not a control.

The per-iteration baseline requirement lives in `close_iteration` as a
deterministic check that a training eval ran (`agent_tools.py`, check 5c), and
`run_eval` records whether that baseline ran on an already-modified tree
(`IterationContext.baseline_tree_dirty`). See `specs/research_plan.md`: the
agent-directed arm is defined by the agent making these calls, so the harness
holds the *existence* of a baseline constant, not its ordering.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from src.engine.telemetry.trace_layout import HOLDOUT_DENY_FRAGMENT

from .iteration_context import IterationContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreEditGuardDecision:
    allowed: bool
    reason: str | None = None


def check_pre_edit_guard_decision(
    ctx: IterationContext,
    workspace: str,
    *,
    tool_name: str,
    tool_input: dict[str, Any],
    deny_paths: list[str] | None = None,
) -> PreEditGuardDecision:
    """Provider-neutral write-guard decision. Only `apply_patch` is gated.

    Provider adapters translate their native tool hook payloads into
    ``tool_name`` and ``tool_input``; the policy lives here.

    ``apply_patch`` (gpt-5.x-codex's preferred edit tool, OpenCode only) carries
    ``patchText`` with one or more ``*** Update/Add/Delete File:`` headers. Its
    targets are therefore invisible to the holdout guard, which matches on a
    ``file_path`` argument or a Bash command string — so the fence is applied
    here instead. ``edit``/``write``/``bash`` need nothing: OpenCode's permission
    deny patterns and the Claude holdout hook already fence those, and there is
    no longer an ordering control to apply.

    `ctx` and `workspace` are retained in the signature for the provider
    adapters that already thread them and for any future per-iteration policy;
    the current decision needs neither.
    """
    tool = tool_name.lower()
    if tool in {"apply_patch", "patch"}:
        return _check_apply_patch_decision(tool_input, deny_paths)
    return PreEditGuardDecision(allowed=True)


# apply_patch headers that name a target file: `*** Update File: <path>`,
# `*** Add File: <path>`, `*** Delete File: <path>`, and the `*** Move to:
# <path>` rename target. Captures the path to the end of the line.
_APPLY_PATCH_PATH_RE = re.compile(
    r"^\*\*\*\s+(?:Update File|Add File|Delete File|Move to):\s*(.+?)\s*$",
    re.MULTILINE,
)


def _apply_patch_target_paths(patch_text: str) -> list[str]:
    """Extract target file paths from an OpenAI-style apply_patch body."""
    if not patch_text:
        return []
    return [m.strip() for m in _APPLY_PATCH_PATH_RE.findall(patch_text) if m.strip()]


def _matches_deny_fragment(path: str, deny_paths: list[str]) -> bool:
    """True iff ``path`` contains any holdout deny fragment (substring match).

    Mirrors `_make_holdout_guard_hook`'s substring policy fence — catches both
    absolute and relative references; it is a policy fence, not cryptographic.
    """
    if not path:
        return False
    return any(fragment and fragment in path for fragment in deny_paths)


def _check_apply_patch_decision(
    tool_input: dict[str, Any],
    deny_paths: list[str] | None,
) -> PreEditGuardDecision:
    """Deny an apply_patch whose target path hits a holdout fragment.

    A Python-side holdout fence alongside OpenCode's permission deny patterns —
    defense in depth against a permission-config regression, and the reason
    this hook still exists at all. `apply_patch` is gpt-5.x-codex's preferred
    edit tool and is the one write path the Claude holdout hook cannot see
    (Claude has no apply_patch), because its targets live inside `patchText`
    rather than in a `file_path` argument.
    """
    deny = list(deny_paths or [])
    if not deny:
        return PreEditGuardDecision(allowed=True)
    patch_text = str(
        tool_input.get("patchText")
        or tool_input.get("patch")
        or tool_input.get("input")
        or ""
    )
    for path in _apply_patch_target_paths(patch_text):
        if _matches_deny_fragment(path, deny):
            return PreEditGuardDecision(
                allowed=False,
                reason=(
                    f"apply_patch targets a holdout (test-set) path "
                    f"{path!r}, which the architect must never edit."
                ),
            )
    return PreEditGuardDecision(allowed=True)


def _make_holdout_guard_hook(
    iteration_idx: int,
    testset_path: str | None = None,
    extra_deny_paths: list[str] | None = None,
) -> Callable:
    """Return a PreToolUse hook that blocks architect access to holdout files.

    Fences the held-out test set (when provided): it is locked from every
    agent for the whole run — the orchestrator scores it at finalization
    through a subprocess that is not under the SDK hook system, so this
    guard does not interfere with the legitimate evals. Engine 3 has no
    validation set; the training data (including the comparison set) is
    freely inspectable.

    Substring match is deliberate: it catches both absolute (`/workspace/data/
    …test.jsonl`) and relative (`data/…test.jsonl`) references via Read,
    Edit, Write, Grep, Glob, and Bash commands. A truly adversarial agent could
    obfuscate the path; the guard is a policy fence, not a cryptographic one.
    """
    # (label, abspath, basename) for each holdout file we protect.
    protected: list[tuple[str, str, str]] = []
    if testset_path:
        protected.append(
            ("test set", os.path.abspath(testset_path), os.path.basename(testset_path))
        )
    # The held-out set's eval traces (`iteration_{N}_holdout[_k]/`) carry the
    # test rows' inputs and outputs, so they are fenced like the file itself.
    protected.append(("test set", HOLDOUT_DENY_FRAGMENT, HOLDOUT_DENY_FRAGMENT))
    # Nice to have: fence the architect's own meta-traces. Matched as a substring fragment (any `iteration_*_architect_trace/`
    # directory in any architect_traces dir). Harmless when no such files exist.
    protected.append(
        ("architect meta-trace", "_architect_trace", "_architect_trace")
    )
    # Fence CodeEvolver's own constraint-checker diff fixtures (real optimization
    # diffs under tests/functional/test_data/constraint_diffs/) so the architect
    # can't Read/Grep/Bash them and lift program ideas. Mirrors the sub-agent
    # fence in iteration_architect.CONSTRAINT_FIXTURE_DENY_FRAGMENT.
    protected.append(
        ("constraint-checker fixtures", "constraint_diffs", "constraint_diffs")
    )
    # Client-supplied extra fences (substring fragments). Matched the same way
    # as the built-in fences; denied with a generic reason.
    for fragment in extra_deny_paths or []:
        if fragment:
            protected.append(("client-specified path", fragment, fragment))

    if not protected:
        async def _noop(_input, _tool_use_id, _context):
            return {}
        return _noop

    def _matched_label(*fragments: Any) -> str | None:
        for fragment in fragments:
            if not fragment:
                continue
            text = str(fragment)
            for label, abs_path, basename in protected:
                if abs_path in text or basename in text:
                    return label
        return None

    async def holdout_guard(input_data, _tool_use_id, _context):
        tool_name = input_data.get("tool_name", "") if isinstance(input_data, dict) else getattr(input_data, "tool_name", "")
        tool_input = input_data.get("tool_input", {}) if isinstance(input_data, dict) else getattr(input_data, "tool_input", {})
        tool_input = tool_input or {}

        label: str | None = None
        if tool_name in ("Read", "Edit", "Write"):
            label = _matched_label(tool_input.get("file_path", ""))
        elif tool_name == "Grep":
            label = _matched_label(
                tool_input.get("path", ""), tool_input.get("pattern", "")
            )
        elif tool_name == "Glob":
            label = _matched_label(
                tool_input.get("path", ""), tool_input.get("pattern", "")
            )
        elif tool_name == "Bash":
            label = _matched_label(tool_input.get("command", ""))

        if label is None:
            return {}

        if label == "architect meta-trace":
            reason = (
                "Access to CodeEvolver architect meta-traces "
                "(`iteration_*_architect_trace/`) is prohibited. These capture your own "
                "session and are not part of your task data — reason from "
                "`run_eval` and `inspect_traces` instead."
            )
        elif label == "client-specified path":
            reason = (
                "Access to this path is prohibited by the client's "
                "additional_deny_paths configuration. No agent may read, open, "
                "or edit it during the run."
            )
        else:
            reason = (
                "Access to the held-out test set is prohibited. It is locked for "
                "the entire run and is scored only by the system at finalization "
                "— no agent may read, open, or edit it."
            )
        logger.warning(
            "iteration %d: PreToolUse guard denied %s call referencing %s "
            "(tool_input=%r)",
            iteration_idx, tool_name, label, tool_input,
        )
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    return holdout_guard


# Legacy alias.
_make_valset_guard_hook = None  # removed — use _make_holdout_guard_hook

# There is no Claude-side pre-edit hook: Claude has no apply_patch tool and
# there is no ordering rule to enforce. The OpenCode adapter routes
# apply_patch through `check_pre_edit_guard_decision` for the holdout fence.


__all__ = [
    "PreEditGuardDecision",
    "check_pre_edit_guard_decision",
    "_make_holdout_guard_hook",
]
