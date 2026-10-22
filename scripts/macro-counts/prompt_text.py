"""Mechanical extraction of a program's prompt-like text, per run and per landed iteration.

The lesson discovery session must see every run's prompt history the same way, whatever the arm. This
module parses the program package's Python files at a commit and pulls out every string literal that
could be prompt text (instruction strings, field descriptions, signature strings, docstrings, f-string
templates), each labelled with where it sits (file, enclosing class/def, assignment target or call
keyword). It is deliberately a superset: helper docstrings the LM never sees are included and labelled
as docstrings, and the session decides what is LM-visible.

Writes into the workspace:
  prompt_text/seed.md                     every block of the seed's task program file
  prompt_text/<run>/final.md              every block of the final program's task file and the files the run changed
  prompt_text/<run>/iterations.md         per landed iteration: blocks added, removed and changed vs its parent
"""
from __future__ import annotations

import ast
import difflib
from pathlib import Path

from bench_workspace import _git, landed_iterations
from runs import Run

MIN_CHARS = 30  # shorter literals are keys, labels and format bits, not prompt text
MIN_WORDS = 4


def _keep(text: str) -> bool:
    t = text.strip()
    return len(t) >= MIN_CHARS or len(t.split()) >= MIN_WORDS or "->" in t


def _joined(node: ast.JoinedStr) -> str:
    parts = []
    for v in node.values:
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            parts.append(v.value)
        elif isinstance(v, ast.FormattedValue):
            parts.append("{" + ast.unparse(v.value) + "}")
    return "".join(parts)


class _Collector(ast.NodeVisitor):
    def __init__(self, fname: str):
        self.fname = fname
        self.scope: list[str] = []
        self.label: list[str] = []
        self.blocks: list[tuple[str, str]] = []
        self.docstrings: set[int] = set()

    def _add(self, node: ast.AST, text: str) -> None:
        if not _keep(text):
            return
        where = ".".join(self.scope) or "<module>"
        kind = "docstring" if id(node) in self.docstrings else (self.label[-1] if self.label else "string")
        self.blocks.append((f"{self.fname} :: {where} :: {kind}", text.strip("\n")))

    def _scoped(self, node, name: str):
        body = getattr(node, "body", [])
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            self.docstrings.add(id(body[0].value))
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node):
        self._scoped(node, f"class {node.name}")

    def visit_FunctionDef(self, node):
        self._scoped(node, f"def {node.name}")

    visit_AsyncFunctionDef = visit_FunctionDef

    def _with_label(self, label: str, node: ast.AST) -> None:
        self.label.append(label)
        self.visit(node)
        self.label.pop()

    def visit_Assign(self, node):
        targets = ", ".join(ast.unparse(t) for t in node.targets)
        self._with_label(f"{targets} =", node.value)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self._with_label(f"{ast.unparse(node.target)} =", node.value)

    def visit_Call(self, node):
        fn = ast.unparse(node.func)
        for a in node.args:
            self._with_label(f"arg of {fn}()", a)
        for k in node.keywords:
            self._with_label(f"{k.arg}= in {fn}()", k.value)
        self.visit(node.func)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            self._add(node, node.value)

    def visit_JoinedStr(self, node):
        self._add(node, _joined(node))


def extract_blocks(source: str, fname: str) -> list[tuple[str, str]]:
    """[(location key, text)] for every prompt-like string literal in one Python file, in source order.
    Unparseable files yield one block saying so."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [(f"{fname} :: <unparseable>", f"(syntax error: {e})")]
    c = _Collector(fname)
    c.visit(tree)
    return c.blocks


def program_dir(prog: str) -> str:
    return str(Path(prog).parent)


def commit_blocks(repo: Path, sha: str, prog: str, only: set[str] | None = None) -> list[tuple[str, str]]:
    """Blocks of every .py file under the program package at ``sha`` (only the ``only`` files when given)."""
    files = [f for f in _git(repo, "ls-tree", "-r", "--name-only", sha, "--", program_dir(prog)).split()
             if f.endswith(".py") and (only is None or f in only)]
    out: list[tuple[str, str]] = []
    for f in sorted(files):
        out += extract_blocks(_git(repo, "show", f"{sha}:{f}"), f)
    return out


def _keyed(blocks: list[tuple[str, str]]) -> dict[str, str]:
    """Location keys made unique by occurrence number, so repeated locations diff positionally."""
    seen: dict[str, int] = {}
    out = {}
    for k, t in blocks:
        seen[k] = seen.get(k, 0) + 1
        out[k if seen[k] == 1 else f"{k} #{seen[k]}"] = t
    return out


def render_blocks(blocks: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"#### `{k}`\n```text\n{t}\n```" for k, t in blocks) or "(no prompt-like strings)"


def diff_blocks(old: list[tuple[str, str]], new: list[tuple[str, str]]) -> str:
    """Added, removed and changed blocks between two commits; changed blocks as a line diff."""
    a, b = _keyed(old), _keyed(new)
    out = []
    for k in b:
        if k not in a:
            out.append(f"#### ADDED `{k}`\n```text\n{b[k]}\n```")
        elif a[k] != b[k]:
            d = difflib.unified_diff(a[k].splitlines(), b[k].splitlines(), lineterm="", n=1)
            out.append(f"#### CHANGED `{k}`\n```diff\n" + "\n".join(list(d)[2:]) + "\n```")
    for k in a:
        if k not in b:
            out.append(f"#### REMOVED `{k}`\n```text\n{a[k]}\n```")
    return "\n\n".join(out) or "(no prompt-like text changed)"


def seed_sha_of(repo: Path, run: Run) -> str:
    rec = run.iters.get("0") or {}
    return rec.get("close_sha") or _git(repo, "rev-list", "--max-parents=0", "HEAD").split()[-1]


def program_text(repo: Path, sha: str, seed_sha: str, prog: str) -> str:
    """One program's prompt-like text: the task program file plus every program-package file changed since
    the seed. Names no run, so it can be judged blind."""
    changed = set(_git(repo, "diff", "--name-only", seed_sha, sha, "--", program_dir(prog)).split())
    return render_blocks(commit_blocks(repo, sha, prog, changed | {prog}))


def write_prompt_text(ws: Path, runs: list[Run], prog: str) -> dict[str, int]:
    """Write prompt_text/ into the workspace. Returns chars written per run."""
    root = ws / "prompt_text"
    root.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}
    seed_done = False
    for run in runs:
        repo = ws / "repos" / run.run_id
        rdir = root / run.run_id
        rdir.mkdir(parents=True, exist_ok=True)
        seed_sha = seed_sha_of(repo, run)
        if not seed_done:  # the seed's task program only: other modules in the package are unused baselines
            (root / "seed.md").write_text("# Prompt-like text of the seed program\n\n"
                                          + render_blocks(commit_blocks(repo, seed_sha, prog, {prog})) + "\n",
                                          encoding="utf-8")
            seed_done = True
        final = _git(repo, "rev-parse", "HEAD").strip()
        (rdir / "final.md").write_text(f"# Prompt-like text of the final program of `{run.run_id}`\n\n"
                                       f"The task program file and every file the run changed.\n\n"
                                       + program_text(repo, final, seed_sha, prog) + "\n", encoding="utf-8")
        parts = [f"# Prompt-like text changes of every landed iteration of `{run.run_id}`\n",
                 "Each section diffs the iteration's closing commit against its parent's. Only landed iterations "
                 "are listed.\n"]
        for n in sorted(landed_iterations(run)):
            rec = run.iters[str(n)]
            parent, close = rec.get("parent_sha"), rec.get("close_sha")
            head = f"## Iteration {n} (parent iteration {rec.get('parent_iteration', '?')})\n"
            if not parent or not close:
                parts.append(head + "\n(no parent or closing commit recorded)\n")
                continue
            parts.append(head + "\n" + diff_blocks(commit_blocks(repo, parent, prog),
                                                   commit_blocks(repo, close, prog)) + "\n")
        text = "\n".join(parts)
        (rdir / "iterations.md").write_text(text, encoding="utf-8")
        sizes[run.run_id] = len(text)
    return sizes
