"""Tests for the in-place repo lock that prevents concurrent runs on one repo."""
from __future__ import annotations

import pytest

from src.local.repo_lock import RepoLock, RepoLockedError


def test_acquire_writes_pid_and_creates_lockfile(tmp_path):
    lock = RepoLock(tmp_path)
    lock.acquire()
    try:
        lockfile = tmp_path / ".codeevolver.lock"
        assert lockfile.exists()
    finally:
        lock.release()


def test_double_acquire_in_same_instance_is_noop(tmp_path):
    lock = RepoLock(tmp_path)
    lock.acquire()
    lock.acquire()  # second call must not raise
    lock.release()


def test_second_acquire_from_different_instance_blocks(tmp_path):
    first = RepoLock(tmp_path)
    second = RepoLock(tmp_path)
    first.acquire()
    try:
        with pytest.raises(RepoLockedError):
            second.acquire()
    finally:
        first.release()


def test_release_unlocks_so_another_can_acquire(tmp_path):
    first = RepoLock(tmp_path)
    second = RepoLock(tmp_path)
    first.acquire()
    first.release()
    second.acquire()  # must not raise
    second.release()


def test_context_manager_releases_on_exit(tmp_path):
    with RepoLock(tmp_path):
        with pytest.raises(RepoLockedError):
            RepoLock(tmp_path).acquire()
    # After the context exits the lock should be releasable to a new acquirer.
    other = RepoLock(tmp_path)
    other.acquire()
    other.release()
