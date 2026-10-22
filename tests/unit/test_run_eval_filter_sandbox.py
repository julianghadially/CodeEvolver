"""Copyright © 2026 440 Labs LLC

Integration test for the `run_eval` filter, exercised against the real mounted
evaluator subprocess.

Self-contained — no client repo clone, no venv. Builds a toy workspace
(trainset + program + metric) under `/tmp`, spawns `RunEvalBinding`, and
verifies each filter shape narrows `row_ids` correctly.

Not tested here (covered by `tests/test_run_eval_filter.py` unit suite):
  - Filter normalization edge cases
  - Row-match predicate logic

Covered here end-to-end (needs real filesystem + subprocess dispatch):
  - The binding's lazy trainset load from `config.trainset_path`
  - The subprocess evaluator actually evaluating the filtered subset
  - The returned `row_ids` matching what the filter demanded
"""

from __future__ import annotations

import json
import importlib
import os
import random
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from src.engine.evaluator.config import EvaluatorConfig
from src.engine.optimizer.v3.iteration_context import IterationContext
from src.engine.optimizer.v3.run_eval_tool import RunEvalBinding, RunEvalError


_TRAINSET = [
    {"id": 0, "category": "metals",      "desc": "steel rod"},
    {"id": 1, "category": "metals",      "desc": "copper pipe"},
    {"id": 2, "category": "textiles",    "desc": "cotton fabric"},
    {"id": 3, "category": "textiles",    "desc": "wool yarn"},
    {"id": 4, "category": "metals",      "desc": "aluminum foil"},
    {"id": 5, "category": "electronics", "desc": "circuit board"},
]


def test_mounted_evaluator_extends_loaded_src_package_for_client_src(tmp_path):
    import src as engine_src
    from src.engine.evaluator.mounted import evaluate

    package_name = "client_fixture_pkg"
    package_dir = tmp_path / "src" / package_name
    package_dir.mkdir(parents=True)
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "__init__.py").write_text("VALUE = 'client-src'\n", encoding="utf-8")

    original_src_path = list(engine_src.__path__)
    sys.modules.pop(f"src.{package_name}", None)
    try:
        evaluate._add_workspace_to_import_path(str(tmp_path))

        module = importlib.import_module(f"src.{package_name}")

        assert module.VALUE == "client-src"
        assert str(tmp_path / "src") in engine_src.__path__
    finally:
        engine_src.__path__[:] = original_src_path
        sys.modules.pop(f"src.{package_name}", None)


def _make_toy_workspace() -> str:
    """Create a minimal `/tmp` workspace with toy program, metric, and trainset.

    Initializes a git repo because `RunEvalBinding` calls `git_get_sha()` to
    stamp each `eval_history` entry with the current HEAD.
    """
    workspace = tempfile.mkdtemp(prefix="filter_sandbox_test_")

    subprocess.run(
        ["git", "init", "-q", "-b", "main", workspace],
        check=True,
    )
    subprocess.run(
        [
            "git", "-C", workspace,
            "-c", "user.email=test@codeevolver.ai",
            "-c", "user.name=test",
            "commit", "--allow-empty", "-q", "-m", "seed",
        ],
        check=True,
    )

    pkg = Path(workspace) / "toy"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "program.py").write_text(textwrap.dedent('''\
        """Toy program — the evaluator auto-maps row fields to these kwargs."""
        def program(id, category, desc):
            return {"predicted": category, "desc_seen": desc}
    '''))
    (pkg / "metric.py").write_text(textwrap.dedent('''\
        """Toy metric — returns 0.5 to avoid the max_score resample branch."""
        def accuracy(output, example):
            return 0.5
    '''))

    (Path(workspace) / "trainset.json").write_text(json.dumps(_TRAINSET))

    return workspace


def _make_binding(
    workspace: str,
    config: EvaluatorConfig,
    iteration_idx: int,
    *,
    seed: int = 42,
) -> tuple[IterationContext, RunEvalBinding]:
    ctx = IterationContext(iteration_idx=iteration_idx)
    ctx.parent_locked = True
    ctx.parent_sha = "parent"
    ctx.child_branch = "main"
    binding = RunEvalBinding(
        ctx=ctx,
        workspace=workspace,
        config=config,
        trainset_ids=list(range(len(_TRAINSET))),
        rng=random.Random(seed),
        subsample_size=10,
    )
    return ctx, binding


def run_filter_sandbox_test() -> dict:
    """Exercise the filter on the real evaluator subprocess.

    Returns a dict of per-case results. `all_pass` is True iff every case's
    observed `row_ids` matches the expected set.
    """
    workspace = _make_toy_workspace()

    # The mounted evaluator adds `workspace` to sys.path for imports. For the
    # subprocess to find `toy.program`, we also need it on PYTHONPATH so the
    # subprocess environment built by `subprocess_service._build_env` inherits
    # it via `os.environ.copy()`.
    existing_pp = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{workspace}:{existing_pp}" if existing_pp else workspace

    config = EvaluatorConfig(
        program_factory="toy.program:program",
        metric_fns=["toy.metric:accuracy"],
        metric_names=["accuracy"],
        primary_metric="accuracy",
        trainset_path="trainset.json",
    )

    cases: list[dict] = []

    # --- no filter: draws from all 6 rows. The response is summary-only:
    # row_ids is echoed inline only when a filter is in effect, so
    # this case asserts via n_rows.
    _, binding = _make_binding(workspace, config, 1)
    r = binding(eval_sample="train")
    cases.append({
        "name": "no_filter",
        "filter": None,
        "n_rows": r.get("n_rows"),
        "expected_set": sorted(range(len(_TRAINSET))),
        "ok": "row_ids" not in r and r.get("n_rows") == len(_TRAINSET),
    })

    # --- eq shorthand: metals → {0, 1, 4} ---
    _, binding = _make_binding(workspace, config, 2)
    r = binding(
        eval_sample="train",
        filter={"category": "metals"},
    )
    cases.append({
        "name": "eq_metals",
        "filter": {"category": "metals"},
        "row_ids": r["row_ids"],
        "expected_set": [0, 1, 4],
        "ok": set(r["row_ids"]) == {0, 1, 4},
    })

    # --- contains: desc contains "steel" → {0} ---
    _, binding = _make_binding(workspace, config, 3)
    r = binding(
        eval_sample="train",
        filter={"desc": {"contains": "steel"}},
    )
    cases.append({
        "name": "contains_steel",
        "filter": {"desc": {"contains": "steel"}},
        "row_ids": r["row_ids"],
        "expected_set": [0],
        "ok": set(r["row_ids"]) == {0},
    })

    # --- in: category in [metals, electronics] → {0, 1, 4, 5} ---
    _, binding = _make_binding(workspace, config, 4)
    r = binding(
        eval_sample="train",
        filter={"category": {"in": ["metals", "electronics"]}},
    )
    cases.append({
        "name": "in_metals_electronics",
        "filter": {"category": {"in": ["metals", "electronics"]}},
        "row_ids": r["row_ids"],
        "expected_set": [0, 1, 4, 5],
        "ok": set(r["row_ids"]) == {0, 1, 4, 5},
    })

    # --- empty match → RunEvalError ---
    _, binding = _make_binding(workspace, config, 5)
    try:
        binding(
            eval_sample="train",
            filter={"category": "nonexistent"},
        )
        empty_error = False
        empty_msg = "(no error raised)"
    except RunEvalError as exc:
        msg = str(exc)
        empty_error = "matches 0 of" in msg
        empty_msg = msg
    cases.append({
        "name": "empty_match_raises",
        "filter": {"category": "nonexistent"},
        "error_msg": empty_msg,
        "ok": empty_error,
    })

    # --- lock enforcement: mismatched filter on 2nd call → RunEvalError ---
    _, binding = _make_binding(workspace, config, 6)
    binding(
        eval_sample="train",
        filter={"category": "metals"},
    )
    try:
        binding(
            eval_sample="train",
            filter={"category": "textiles"},
        )
        lock_error = False
        lock_msg = "(no error raised)"
    except RunEvalError as exc:
        msg = str(exc)
        lock_error = "locked" in msg
        lock_msg = msg
    cases.append({
        "name": "filter_lock_enforced",
        "error_msg": lock_msg,
        "ok": lock_error,
    })

    all_pass = all(c["ok"] for c in cases)
    return {
        "all_pass": all_pass,
        "num_cases": len(cases),
        "num_pass": sum(1 for c in cases if c["ok"]),
        "workspace": workspace,
        "cases": cases,
    }
