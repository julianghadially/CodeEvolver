from __future__ import annotations


def test_architect_permissions_allow_guarded_edits_and_deny_valset_reads(tmp_path):
    from src.engine.agent.harnesses.opencode.permissions import (
        build_architect_permission_config,
    )

    valset_path = str(tmp_path / "holdout.jsonl")

    permission = build_architect_permission_config([valset_path, "holdout.jsonl"])

    # Non-holdout edits/writes stay allowed; the holdout is fenced out of both
    # the read and edit/write surfaces.
    assert permission["edit"]["*"] == "allow"
    assert permission["write"]["*"] == "allow"
    assert permission["question"] == "deny"
    assert permission["read"][f"*{valset_path}*"] == "deny"
    assert permission["edit"][f"*{valset_path}*"] == "deny"
    # apply_patch mirrors the edit rule (allowed, holdout-fenced) so gpt-5.x-codex
    # can use its preferred edit tool without losing the valset/test fence.
    assert permission["apply_patch"]["*"] == "allow"
    assert permission["apply_patch"][f"*{valset_path}*"] == "deny"


def test_architect_permissions_deny_opencode_tools_not_in_claude_allowlist():
    from src.engine.agent.harnesses.opencode.permissions import (
        build_architect_permission_config,
    )

    permission = build_architect_permission_config([])

    assert permission["task"] == "deny"
    assert permission["webfetch"] == "deny"
    assert permission["websearch"] == "deny"
    assert permission["todowrite"] == "deny"
    # apply_patch is allowed (governed by the edit rule) — see the guarded-edits
    # test above; with no holdout fragments it is simply {"*": "allow"}.
    assert permission["apply_patch"] == {"*": "allow"}
    assert permission["list"] == "deny"
    assert permission["lsp"] == "deny"


def test_inspection_permissions_are_read_only_by_default():
    from src.engine.agent.harnesses.opencode.permissions import (
        build_inspection_permission_config,
    )

    permission = build_inspection_permission_config(
        workspace="/workspace",
        writable=False,
        deny_paths=["validation.jsonl"],
    )

    assert permission["edit"] == "deny"
    assert permission["bash"] == "deny"
    assert permission["read"]["*validation.jsonl*"] == "deny"


def test_writable_inspection_allows_edits_with_holdout_denied():
    from src.engine.agent.harnesses.opencode.permissions import (
        build_inspection_permission_config,
    )

    permission = build_inspection_permission_config(
        workspace="/workspace",
        writable=True,
        deny_paths=["validation.jsonl"],
    )

    # Allow-by-default with holdout denies — the same stance the architect and
    # coding agents use. A prior deny-by-default whitelist silently blocked all
    # writes: OpenCode routes apply_patch through the `edit` permission, and the
    # whitelist's allow glob did not reliably match the direct child memory.md,
    # so codex apply_patch calls were denied. The agent stays confined via
    # cwd = memory dir + bash denied; holdout paths are still fenced (below).
    assert permission["edit"]["*"] == "allow"
    assert permission["edit"]["*validation.jsonl*"] == "deny"
    assert permission["write"]["*"] == "allow"
    assert permission["write"]["*validation.jsonl*"] == "deny"
    # apply_patch is governed by the `edit` rule; the explicit rule is inert.
    assert permission["apply_patch"] == "allow"


def test_bash_is_opt_in_and_stays_denied_by_default():
    """Only the clean-slate inspector asks for a shell; nothing else gets one.

    `allow_bash` defaults False, so a caller that forgets to think about it
    keeps today's behaviour. Writability is orthogonal: the writable path is
    the valset-reflection agent, which runs with cwd = the memory dir and must
    not roam.
    """
    from src.engine.agent.harnesses.opencode.permissions import (
        build_inspection_permission_config,
    )

    for writable in (False, True):
        default = build_inspection_permission_config(
            workspace="/workspace", writable=writable, deny_paths=[],
        )
        assert default["bash"] == "deny"

        opted_in = build_inspection_permission_config(
            workspace="/workspace", writable=writable, deny_paths=[],
            allow_bash=True,
        )
        assert opted_in["bash"] == "allow"
        # A shell is the ONLY thing that changes — in particular the inspector
        # still cannot write, which is what makes the read-only contract
        # (clean_slate.py::_snapshot_repo) something a guard can enforce.
        assert {k: v for k, v in opted_in.items() if k != "bash"} == {
            k: v for k, v in default.items() if k != "bash"
        }
