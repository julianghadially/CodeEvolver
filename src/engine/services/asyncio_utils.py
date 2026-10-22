"""Copyright © 2026 440 Labs LLC

Process-level asyncio setup helpers.
"""

from __future__ import annotations


def install_pidfd_child_watcher() -> str | None:
    """Install a loop-independent asyncio child watcher.

    Each `IterationArchitect.run()` calls `anyio.run(...)`, which creates
    and closes a fresh event loop per iteration. The default SIGCHLD-based
    watchers tie the Claude CLI subprocess transport to that loop; when the
    loop closes before the transport is fully torn down, asyncio emits
    `Loop ... is closed` warnings and risks zombies.

    Prefers `PidfdChildWatcher` (tracks children via pidfd, fully loop-
    independent). Falls back to `ThreadedChildWatcher` (waitpid in a
    background thread, also loop-independent) on kernels where the
    `pidfd_open` syscall raises ENOSYS. The constructor of
    `PidfdChildWatcher` does not exercise the syscall, so we probe it
    explicitly before selecting the pidfd path — otherwise the failure
    only surfaces when subprocesses are spawned, and every Claude Agent
    SDK session crashes with `OSError: [Errno 38] Function not implemented`.

    Returns the class name of the watcher that is active on the policy
    after the call (post-install on success, the pre-existing watcher on
    a no-op path). Returns `None` when we could not inspect the policy.
    Silently no-ops on Python 3.14+ (where `set_child_watcher` is removed)
    and on platforms where neither watcher class is available.
    """
    import asyncio

    policy = asyncio.get_event_loop_policy()
    set_watcher = getattr(policy, "set_child_watcher", None)
    get_watcher = getattr(policy, "get_child_watcher", None)
    if set_watcher is None:
        return _current_watcher_name(get_watcher)
    watcher = _make_loop_independent_watcher()
    if watcher is None:
        return _current_watcher_name(get_watcher)
    try:
        set_watcher(watcher)
    except (NotImplementedError, RuntimeError):
        pass
    return _current_watcher_name(get_watcher)


def _current_watcher_name(get_watcher) -> str | None:
    if get_watcher is None:
        return None
    try:
        return type(get_watcher()).__name__
    except Exception:
        return None


def _make_loop_independent_watcher():
    import asyncio

    pidfd_cls = getattr(asyncio, "PidfdChildWatcher", None)
    if pidfd_cls is not None and _pidfd_open_works():
        try:
            return pidfd_cls()
        except (OSError, NotImplementedError):
            pass
    threaded_cls = getattr(asyncio, "ThreadedChildWatcher", None)
    if threaded_cls is not None:
        try:
            return threaded_cls()
        except (OSError, NotImplementedError):
            pass
    return None


def _pidfd_open_works() -> bool:
    import os

    pidfd_open = getattr(os, "pidfd_open", None)
    if pidfd_open is None:
        return False
    try:
        fd = pidfd_open(os.getpid())
    except (OSError, NotImplementedError):
        return False
    os.close(fd)
    return True


__all__ = ["install_pidfd_child_watcher"]
