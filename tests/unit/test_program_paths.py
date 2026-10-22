"""Copyright © 2026 440 Labs LLC

Program vs. non-program content identity (`v3/program_paths.py`).

This predicate decides whether two trees are "the same program". Three things
depend on it — the `close_iteration` gate, final-score attribution, and
`changes_landed` — so its edges matter: too narrow and the architect pays a
redundant full eval for a documentation commit; too wide and an
unevaluated program change can close.
"""

from __future__ import annotations

import subprocess

import pytest

from src.engine.optimizer.v3.program_paths import (
    diff_is_non_program_only,
    is_non_program_path,
    program_diff_paths,
)


def _git(ws, *args):
    subprocess.run(["git", *args], cwd=ws, check=True, capture_output=True)


@pytest.fixture()
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "program.py").write_text("K = 7\n")
    (tmp_path / "codeevolver.md").write_text("# Architecture\n")
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "system.md").write_text("You are a judge.\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "seed")
    return tmp_path


def _tree(ws, ref="HEAD") -> str:
    return subprocess.run(
        ["git", "rev-parse", f"{ref}^{{tree}}"], cwd=ws, capture_output=True, text=True
    ).stdout.strip()


def _commit(ws, message) -> str:
    _git(ws, "add", "-A")
    _git(ws, "commit", "-q", "-m", message)
    return _tree(ws)


@pytest.mark.parametrize(
    "path, expected",
    [
        ("codeevolver.md", True),
        ("README.md", True),
        ("README.dev.md", True),
        ("CHANGELOG.md", True),
        ("LICENSE", True),
        (".gitignore", True),
        ("./codeevolver.md", True),
        ("program.py", False),
        # NOT waived: a client program may read a root-level markdown prompt
        # at runtime. Waiving all root *.md would let the architect rewrite it
        # and close without a re-eval, with the pre-change score attributed to
        # the changed program.
        ("PROMPT.md", False),
        ("SYSTEM.md", False),
        ("instructions.md", False),
        # A nested .md may be a prompt template, a fixture, or a doc the
        # program reads at runtime — only root-level markdown is waived.
        ("prompts/system.md", False),
        ("docs/design.md", False),
        ("sub/.gitignore", False),
        ("", False),
    ],
)
def test_path_classification(path, expected):
    assert is_non_program_path(path) is expected


def test_doc_only_commit_is_the_same_program(repo):
    before = _tree(repo)
    (repo / "codeevolver.md").write_text("# Architecture\n\n## Change summary\nX\n")
    after = _commit(repo, "append change summary")

    assert before != after, "the trees genuinely differ"
    assert diff_is_non_program_only(str(repo), before, after) is True
    assert program_diff_paths(str(repo), before, after) == []


def test_code_change_is_a_different_program(repo):
    before = _tree(repo)
    (repo / "program.py").write_text("K = 20\n")
    after = _commit(repo, "raise k")

    assert diff_is_non_program_only(str(repo), before, after) is False
    assert program_diff_paths(str(repo), before, after) == ["program.py"]


def test_a_code_change_alongside_docs_is_still_a_program_change(repo):
    before = _tree(repo)
    (repo / "codeevolver.md").write_text("# Architecture\n\n## Change summary\nX\n")
    (repo / "program.py").write_text("K = 20\n")
    after = _commit(repo, "raise k + document it")

    assert diff_is_non_program_only(str(repo), before, after) is False
    assert program_diff_paths(str(repo), before, after) == ["program.py"]


def test_nested_markdown_counts_as_program(repo):
    """A prompt template is program content even though it is markdown."""
    before = _tree(repo)
    (repo / "prompts" / "system.md").write_text("You are a strict judge.\n")
    after = _commit(repo, "tighten the judge prompt")

    assert diff_is_non_program_only(str(repo), before, after) is False
    assert program_diff_paths(str(repo), before, after) == ["prompts/system.md"]


def test_identical_revs_are_equivalent(repo):
    tree = _tree(repo)
    assert diff_is_non_program_only(str(repo), tree, tree) is True


def test_unknown_diff_is_never_waived(repo):
    """A git failure must degrade to "not equivalent", never to "equivalent".

    Waiving on an unreadable diff would let an unevaluated program change
    close; refusing only costs a redundant eval.
    """
    assert diff_is_non_program_only(str(repo), _tree(repo), "0" * 40) is False
    assert diff_is_non_program_only(str(repo), None, _tree(repo)) is False
    assert diff_is_non_program_only(str(repo), _tree(repo), None) is False
    assert diff_is_non_program_only("/nonexistent/workspace", "a" * 40, "b" * 40) is False
