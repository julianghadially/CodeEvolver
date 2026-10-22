# Copyright © 2026 440 Labs LLC
"""scripts/prompt_only_audit.py: only string-literal edits to the signatures file
(plus codeevolver.md) pass; any other change voids the run."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.prompt_only_audit import audit_repo, strip_strings

SIG = "langProBe/hover/prompts/hover_signatures.py"
PROG = "langProBe/hover/hover_program.py"
SEED_SIG = '''"""Hover prompts."""
import dspy


class Summarize1(dspy.Signature):
    """Given the fields `claim`, `passages`, produce the fields `summary`."""

    claim: str = dspy.InputField(desc="")
    passages: str = dspy.InputField(desc="")
    summary: str = dspy.OutputField(desc="")
'''


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "ws"
    (r / "langProBe/hover/prompts").mkdir(parents=True)
    (r / SIG).write_text(SEED_SIG)
    (r / PROG).write_text("import dspy\nK = 7\n")
    (r / "codeevolver.md").write_text("# notes\n")
    subprocess.run(["git", "init", "-q", "-b", "seed", str(r)], check=True)
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "seed")
    return r


def _commit_on(repo: Path, branch: str, edits: dict[str, str | None]) -> None:
    _git(repo, "checkout", "-q", "-B", branch, "seed")
    for path, text in edits.items():
        p = repo / path
        if text is None:
            p.unlink()
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", branch)


def test_string_only_edits_and_codeevolver_md_pass(repo: Path) -> None:
    edited = SEED_SIG.replace(
        '"""Given the fields `claim`, `passages`, produce the fields `summary`."""',
        '"""Summarize only the facts in `passages` that bear on the claim.\n\n    Be terse."""',
    ).replace('claim: str = dspy.InputField(desc="")', 'claim: str = dspy.InputField(desc="the claim to verify")')
    _commit_on(repo, "codeevolver-x-iter1", {SIG: edited, "codeevolver.md": "# notes\nmore\n"})
    res = audit_repo(repo, "hover")
    assert res.verdict == "PASS", res.violations
    assert res.refs_checked == 1


@pytest.mark.parametrize(
    "edits, needle",
    [
        ({PROG: "import dspy\nK = 25\n"}, "non-prompt path changed: " + PROG),
        ({"langProBe/hover/prompts/extra.py": "X = 1\n"}, "non-prompt path changed"),
        ({SIG: None}, "deleted"),
        ({SIG: SEED_SIG + "\nimport os\n"}, "code changed"),
        ({SIG: SEED_SIG.replace("passages: str", "docs: str")}, "code changed"),  # renamed field
        ({SIG: SEED_SIG.replace("passages: str", "passages: list[str]")}, "code changed"),  # type change
        ({SIG: SEED_SIG + '    extra: str = dspy.InputField(desc="")\n'}, "code changed"),  # new field
        ({SIG: SEED_SIG.replace('desc="")\n    summary', 'desc="", prefix="P:")\n    summary')}, "code changed"),
        ({SIG: SEED_SIG + "\ndef f(:\n"}, "does not parse"),
        ({".gitignore": "*.py\n"}, "non-prompt path changed: .gitignore"),
    ],
)
def test_non_prompt_changes_void_the_run(repo: Path, edits: dict, needle: str) -> None:
    _commit_on(repo, "codeevolver-x-iter1", edits)
    res = audit_repo(repo, "hover")
    assert res.verdict == "VOID"
    assert any(needle in p for ps in res.violations.values() for p in ps), res.violations


def test_eval_anchor_refs_are_audited_even_when_branch_reverted(repo: Path) -> None:
    # The agent evaluated a code change, then reverted the branch: the program
    # still RAN, so the engine's eval anchor must void the run.
    _commit_on(repo, "codeevolver-x-iter1", {PROG: "import dspy\nK = 25\n"})
    tainted = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/codeevolver/evals/1/0", tainted)
    _git(repo, "reset", "-q", "--hard", "seed")  # branch no longer holds the change
    res = audit_repo(repo, "hover")
    assert res.verdict == "VOID"
    assert list(res.violations) == ["refs/codeevolver/evals/1/0"]


def test_strip_strings_ignores_string_content_but_not_structure() -> None:
    assert strip_strings('X = "a"\n') == strip_strings('X = "b" "c"\n')
    assert strip_strings('X = "a"\n') != strip_strings("X = 1\n")
