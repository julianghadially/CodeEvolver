"""Copyright © 2026 440 Labs LLC

Memory-diff watchdog (deterministic, no real SDK).

Engine 3's outcome reflection lives in memory.md, written directly by the
architect (per the `close_iteration` skill) — there is no per-iteration
reflection file. After the session exits, the loop compares memory.md's hash
against the iteration-start snapshot; if the file is unchanged while the
iteration did real work (>= 2 train evals, or HEAD moved), the session is
resumed once to ask for the reflection.

The watchdog fires only when all three hold:
  - memory.md's hash is unchanged since iteration start,
  - real work happened (>= 2 train evals OR the head sha moved), and
  - a session id was captured to resume.
"""

from __future__ import annotations

import os

from src.engine.optimizer.v3 import iteration_architect as ia_module
from src.engine.optimizer.v3 import architect_watchdogs as wd_module
from src.engine.optimizer.v3.iteration_architect import hash_file

from tests.controls._arch_helpers import make_architect


def _setup(tmp_path, monkeypatch, *, content: str = "# Memory\n\nseed notes.\n"):
    """Architect over a real memory.md whose start-hash is stamped on ctx."""
    memory_md = tmp_path / "memory.md"
    memory_md.write_text(content)
    monkeypatch.setattr(wd_module, "memory_md_path", lambda: str(memory_md))

    arch, ctx, _ = make_architect(iteration_idx=3)
    ctx.memory_md_hash_at_start = hash_file(str(memory_md))
    arch._last_session_id = "sess-1"
    # Head unmoved by default — tests that need movement re-patch git_get_sha.
    monkeypatch.setattr(
        wd_module, "git_get_sha", lambda ws, ref="HEAD": ctx.start_sha
    )
    return arch, ctx, memory_md


def _record_train_evals(ctx, n: int) -> None:
    for i in range(n):
        ctx.record_eval(f"sha{i}", 0.40 + 0.1 * i, eval_sample="train", n_rows=5)


def test_watchdog_fires_when_memory_unchanged_after_two_train_evals(
    tmp_path, monkeypatch
):
    arch, ctx, memory_md = _setup(tmp_path, monkeypatch)
    _record_train_evals(ctx, 2)

    calls: list[dict] = []

    def fake_launch(
        system_prompt,
        tool_registry,
        *,
        resume_session_id=None,
        initial_prompt_override=None,
    ):
        calls.append(
            {"resume": resume_session_id, "prompt": initial_prompt_override}
        )
        # Simulate the resumed architect writing its reflection then
        # re-closing — the "resume writes then re-close" path.
        with open(memory_md, "a") as f:
            f.write("\n## Iteration 3\nreflection written on resume.\n")
        ctx.close_called = True
        ctx.memory_updated_claimed = True

    monkeypatch.setattr(arch, "_launch_architect_provider", fake_launch)

    arch._run_memory_diff_watchdog_if_needed("SYS", {})

    assert ctx.watchdog_fired is True
    assert len(calls) == 1
    assert calls[0]["resume"] == "sess-1"
    assert "memory.md" in calls[0]["prompt"]
    assert "close_iteration" in calls[0]["prompt"]
    # The resume changed memory.md, so a second check would not fire.
    assert hash_file(str(memory_md)) != ctx.memory_md_hash_at_start
    assert ctx.close_called is True


def test_watchdog_does_not_fire_when_memory_changed(tmp_path, monkeypatch):
    arch, ctx, memory_md = _setup(tmp_path, monkeypatch)
    _record_train_evals(ctx, 2)
    # The architect already appended its reflection during the session.
    with open(memory_md, "a") as f:
        f.write("\n## Iteration 3\nalready reflected.\n")

    fired: list[bool] = []
    monkeypatch.setattr(
        arch,
        "_launch_architect_provider",
        lambda *a, **k: fired.append(True),
    )

    arch._run_memory_diff_watchdog_if_needed("SYS", {})

    assert fired == []
    assert ctx.watchdog_fired is False


def test_watchdog_does_not_fire_with_one_eval_and_unmoved_head(
    tmp_path, monkeypatch
):
    arch, ctx, _ = _setup(tmp_path, monkeypatch)
    _record_train_evals(ctx, 1)  # baseline only, HEAD still at start_sha

    fired: list[bool] = []
    monkeypatch.setattr(
        arch,
        "_launch_architect_provider",
        lambda *a, **k: fired.append(True),
    )

    arch._run_memory_diff_watchdog_if_needed("SYS", {})

    assert fired == []
    assert ctx.watchdog_fired is False


def test_watchdog_fires_when_head_moved_even_with_one_eval(tmp_path, monkeypatch):
    arch, ctx, _ = _setup(tmp_path, monkeypatch)
    _record_train_evals(ctx, 1)
    # A commit landed past the iteration-start sha — real work happened.
    monkeypatch.setattr(
        wd_module, "git_get_sha", lambda ws, ref="HEAD": "moved_sha"
    )

    calls: list[dict] = []
    monkeypatch.setattr(
        arch,
        "_launch_architect_provider",
        lambda sp, tr, *, resume_session_id=None, initial_prompt_override=None: calls.append(
            {"resume": resume_session_id, "prompt": initial_prompt_override}
        ),
    )

    arch._run_memory_diff_watchdog_if_needed("SYS", {})

    assert ctx.watchdog_fired is True
    assert len(calls) == 1


def test_watchdog_does_not_fire_without_session_id(tmp_path, monkeypatch):
    arch, ctx, _ = _setup(tmp_path, monkeypatch)
    _record_train_evals(ctx, 2)
    arch._last_session_id = None

    fired: list[bool] = []
    monkeypatch.setattr(
        arch,
        "_launch_architect_provider",
        lambda *a, **k: fired.append(True),
    )

    arch._run_memory_diff_watchdog_if_needed("SYS", {})

    assert fired == []
    assert ctx.watchdog_fired is False


def test_watchdog_does_not_fire_when_no_memory_file_exists(tmp_path, monkeypatch):
    # No memory.md at all (e.g. tests / fresh runs without a memory dir):
    # nothing to diff — never fire.
    missing = tmp_path / "does_not_exist" / "memory.md"
    monkeypatch.setattr(wd_module, "memory_md_path", lambda: str(missing))
    assert not os.path.exists(str(missing))

    arch, ctx, _ = make_architect(iteration_idx=3)
    assert ctx.memory_md_hash_at_start is None
    _record_train_evals(ctx, 2)
    arch._last_session_id = "sess-1"

    fired: list[bool] = []
    monkeypatch.setattr(
        arch,
        "_launch_architect_provider",
        lambda *a, **k: fired.append(True),
    )

    arch._run_memory_diff_watchdog_if_needed("SYS", {})

    assert fired == []
    assert ctx.watchdog_fired is False
