#!/usr/bin/env python3
# Copyright © 2026 440 Labs LLC
"""Offline audit for the prompt-only ablation (orchestrator/prompt_only/).

A prompt-only run may change exactly two things relative to its seed:
  * the benchmark's signatures file, and only its STRING LITERALS (Signature
    docstrings, field `desc=` values, module-level prompt constants such as
    phantomwiki's SEARCH_WIKI_DESC) -- checked by blanking every str constant
    in both ASTs and requiring the dumps to be equal, so any new field, renamed
    field, type change, import or statement fails;
  * `codeevolver.md` (the engine/analysis agent's notes, not program).

The kernel fence (po_fence.sh) prevents other writes but has documented
residuals (git plumbing can commit any path, code placed in the signatures
file runs as root inside run_eval). This audit closes them after the fact: it
reads the collected `repo.bundle` and checks EVERY program the run produced or
evaluated -- every branch head, every `refs/codeevolver/iterations/*` close
anchor and every `refs/codeevolver/evals/*/*` evaluated tree -- against
`refs/heads/seed`. Any violation voids the run.

    python scripts/prompt_only_audit.py ~/ce-runs/pareto_evolution-hover-s1-ppo [...]
    python scripts/prompt_only_audit.py --json ~/ce-runs/*-po
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

SIGNATURES = {
    "hover": "langProBe/hover/prompts/hover_signatures.py",
    "phantomwiki": "src/program/prompts/phantomwiki_signatures.py",
}
NON_PROGRAM = {"codeevolver.md"}
SEED_REF = "refs/heads/seed"
AUDITED_PREFIXES = ("refs/heads/", "refs/codeevolver/iterations/", "refs/codeevolver/evals/")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


def strip_strings(source: str) -> str:
    """ast.dump of `source` with every str constant blanked (raises SyntaxError)."""

    class _Blank(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(value=""), node)
            return node

    return ast.dump(_Blank().visit(ast.parse(source)), include_attributes=False)


def check_tree(repo: Path, seed: str, ref: str, sig_path: str) -> list[str]:
    """Violations for `ref` vs `seed` (empty list = prompt-only)."""
    changed = [p for p in _git(repo, "diff", "--name-only", "--no-renames", seed, ref).splitlines() if p]
    problems = [f"non-prompt path changed: {p}" for p in changed if p not in NON_PROGRAM | {sig_path}]
    if sig_path in changed:
        try:
            after = _git(repo, "show", f"{ref}:{sig_path}")
        except subprocess.CalledProcessError:
            return problems + [f"{sig_path} deleted"]
        before = _git(repo, "show", f"{seed}:{sig_path}")
        try:
            if strip_strings(before) != strip_strings(after):
                problems.append(f"{sig_path}: code changed (not only string literals)")
        except SyntaxError as exc:
            problems.append(f"{sig_path}: does not parse ({exc.msg}, line {exc.lineno})")
    return problems


@dataclass
class RunAudit:
    run_dir: str
    bench: str | None = None
    verdict: str = "VOID"
    refs_checked: int = 0
    violations: dict[str, list[str]] = field(default_factory=dict)
    error: str | None = None


def audit_repo(repo: Path, bench: str, run_dir: str = "") -> RunAudit:
    """Audit an already-populated git repo (all refs fetched)."""
    out = RunAudit(run_dir=run_dir, bench=bench)
    sig = SIGNATURES[bench]
    seed = _git(repo, "rev-parse", SEED_REF).strip()
    refs = [r for r in _git(repo, "for-each-ref", "--format=%(refname)").splitlines()
            if r.startswith(AUDITED_PREFIXES) and r != SEED_REF]
    for ref in refs:
        problems = check_tree(repo, seed, ref, sig)
        if problems:
            out.violations[ref] = problems
    out.refs_checked = len(refs)
    out.verdict = "PASS" if refs and not out.violations else "VOID"
    if not refs:
        out.error = "no audited refs in bundle"
    return out


def audit_run(run_dir: Path) -> RunAudit:
    try:
        bench = json.loads((run_dir / "settings.json").read_text())["bench"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        return RunAudit(run_dir=str(run_dir), error=f"settings.json unreadable: {exc}")
    if bench not in SIGNATURES:
        return RunAudit(run_dir=str(run_dir), bench=bench, error=f"no signatures file known for {bench!r}")
    bundle = run_dir / "repo.bundle"
    if not bundle.is_file():
        return RunAudit(run_dir=str(run_dir), bench=bench, error="repo.bundle missing")
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "r.git"
        subprocess.run(["git", "init", "-q", "--bare", str(repo)], check=True)
        _git(repo, "fetch", "-q", str(bundle), "+refs/*:refs/*")
        try:
            return audit_repo(repo, bench, str(run_dir))
        except subprocess.CalledProcessError as exc:
            return RunAudit(run_dir=str(run_dir), bench=bench, error=f"git failed: {exc.stderr.strip()}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true", help="print one JSON object per run")
    args = ap.parse_args(argv)
    worst = 0
    for d in args.run_dirs:
        res = audit_run(d)
        if res.verdict != "PASS":
            worst = 1
        if args.json:
            print(json.dumps(res.__dict__))
            continue
        print(f"{res.verdict}  {d.name}  ({res.bench}, {res.refs_checked} refs checked)")
        if res.error:
            print(f"    error: {res.error}")
        for ref, problems in list(res.violations.items())[:20]:
            print(f"    {ref}: {'; '.join(problems)}")
        if len(res.violations) > 20:
            print(f"    ... {len(res.violations) - 20} more violating refs")
    return worst


if __name__ == "__main__":
    sys.exit(main())
