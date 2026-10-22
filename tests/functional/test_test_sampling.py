"""Copyright © 2026 440 Labs LLC

Finalization test-set sampling — one-row held-out set drawn three times.

At the end of every run the seed and the final primary are scored on the
held-out test set `test_samples` times (`src/engine/evaluator/holdout.py`),
so each run carries a within-run sample of its final test score rather than
one reading. This test drives `v3/finalization.score_seed_and_final` directly
against the real FactChecker fixture program — no architect session, so the
whole thing is four single-row evals of the client's LM — and asserts the
mechanics:

- ONE draw for the seed (``seed_test_samples``; every arm re-measures the
  same seed, so its readings average across the matrix) and three for the
  final program, trace suffixes ``holdout`` / ``holdout_2`` / ``holdout_3`` (sample 1
  keeps the historical name), each trace holding exactly one row;
- ``seed_test_scores`` has length 1, ``final_test_scores`` length 3, and the
  legacy ``seed_test`` / ``final_test`` are their means;
- both stamped onto the iteration history (``test_score`` + ``test_scores``).

The final "program" is a branch forked from the seed with an empty commit —
its content is identical, which is fine: what is under test is the sampling,
not the program. The seed is scored at ``seed_sha`` (a detached checkout,
never by branch name — the run-main branch's tip is the final program in
real runs); because the final's sha differs from the seed's, the
seed==final short-circuit does not fire and both take their own draws.

Caching: the engine deliberately configures NO cache (the client repo owns
its LM settings). The result reports ``identical_samples`` — whether all
three draws produced byte-identical row outputs — and whether the engine's
"an LLM response cache may be active" warning fired. The FactChecker fixture
leaves DSPy's disk cache on, so identical samples are a plausible outcome
here; that is the detector working, not a failure. Fixing it belongs in the
fixture repo.

To run::

    python local_test_app.py test_sampling
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

from tests.functional.test_e2e import DEFAULT_SEED_BRANCH, DEFAULT_TESTSET_PATH

NUM_SAMPLES = 3
SEED_SAMPLES = 1


def _git(workspace: str, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=workspace, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _trace_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def run_test_sampling(
    env,
    *,
    repo_url: str = "https://github.com/julianghadially/FactChecker",
    program: str = "src.factchecker.simple.factchecker_program:FactCheckerProgram",
    metric: str = "src.codeevolver.metric:metric",
    trainset_path: str = "data/FacTool_QA_train_10examples_normalized.jsonl",
    testset_path: str = DEFAULT_TESTSET_PATH,
    initial_branch: str = DEFAULT_SEED_BRANCH,
    primary_metric: str = "accuracy",
    num_samples: int = NUM_SAMPLES,
    seed_samples: int = SEED_SAMPLES,
    eval_timeout: int = 1800,
) -> dict:
    """Run finalization sampling in-process and return the check results."""
    from src.engine.evaluator.config import EvaluatorConfig
    from src.engine.evaluator.dataset_config import DatasetConfig
    from src.engine.evaluator.holdout import IDENTICAL_SAMPLES_WARNING, test_suffix
    from src.engine.telemetry.trace_layout import iter_rows
    from src.engine.optimizer.v3.ce_state import CEState
    from src.engine.optimizer.v3.finalization import score_seed_and_final
    from src.engine.schemas.iteration_history import IterationHistory

    env.prepare_workspace(repo_url=repo_url, branch=initial_branch)
    workspace = env.workspace_dir
    result: dict = {
        "success": False, "checks": {},
        "num_samples": num_samples, "seed_samples": seed_samples,
    }
    checks = result["checks"]
    try:
        # One held-out row, supplied inline so nothing in the fixture repo is
        # touched; DatasetConfig writes it under .codeevolver/. `testset_path`
        # is the normalized holdout (`statement` / SUPPORTED|REFUTED), the
        # shape the seed program's entry point takes.
        one_row = _trace_rows(Path(workspace) / testset_path)[:1]
        assert one_row, f"no rows in {testset_path}"
        assert "statement" in one_row[0], f"holdout not normalized: {one_row[0]}"
        datasets = DatasetConfig.prepare(
            workspace, trainset_json=None, trainset_path=trainset_path,
            comparison_set_size=None, testset_json=one_row,
        )
        assert datasets.testset_size == 1
        evaluator_config = EvaluatorConfig(
            program_factory=program, metric_fns=[metric],
            metric_names=[primary_metric], primary_metric=primary_metric,
            trainset_path=datasets.trainset_rel_path,
            testset_path=datasets.testset_rel_path or None,
        )

        # Seed branch + a distinct "final" branch (same content, empty commit).
        ts = time.strftime("%Y%m%d%H%M%S")
        seed_branch = f"codeevolver-{ts}-seed"
        final_branch = f"codeevolver-{ts}-final"
        _git(workspace, "checkout", "-q", "-b", seed_branch)
        _git(workspace, "checkout", "-q", "-b", final_branch)
        _git(workspace, "-c", "user.email=ce@test", "-c", "user.name=ce",
             "commit", "-q", "--allow-empty", "-m", "final: identical program")
        final_sha = _git(workspace, "rev-parse", "HEAD")
        seed_sha = _git(workspace, "rev-parse", seed_branch)
        state = CEState(seed_branch=seed_branch, seed_sha=seed_sha)
        state.register_lineage(final_branch, sha=final_sha, iteration=1, is_primary=True)
        history = IterationHistory()
        history[0] = {"git_branch": seed_branch}
        history[1] = {"git_branch": final_branch}

        warned: list[str] = []

        class _Catch(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.WARNING:
                    warned.append(record.getMessage())

        handler = _Catch()
        logging.getLogger("src.engine.evaluator.holdout").addHandler(handler)
        try:
            scores = score_seed_and_final(
                workspace=workspace, datasets=datasets,
                evaluator_config=evaluator_config, state=state, history=history,
                seed_branch=seed_branch, seed_sha=seed_sha,
                comparison_eval_timeout=eval_timeout,
                test_samples=num_samples, seed_test_samples=seed_samples,
            )
        finally:
            logging.getLogger("src.engine.evaluator.holdout").removeHandler(handler)

        result["scores"] = {
            "seed_test": scores.seed_test, "final_test": scores.final_test,
            "seed_test_scores": scores.seed_test_scores,
            "final_test_scores": scores.final_test_scores,
        }
        checks["seed_has_n_samples"] = len(scores.seed_test_scores) == seed_samples
        checks["final_has_n_samples"] = len(scores.final_test_scores) == num_samples
        checks["seed_mean_matches"] = (
            scores.seed_test is not None and scores.seed_test_scores
            and abs(scores.seed_test - sum(scores.seed_test_scores) / seed_samples) < 1e-9
        )
        checks["final_mean_matches"] = (
            scores.final_test is not None and scores.final_test_scores
            and abs(scores.final_test - sum(scores.final_test_scores) / num_samples) < 1e-9
        )
        checks["history_stamped"] = (
            history[0].get("test_scores") == scores.seed_test_scores
            and history[1].get("test_scores") == scores.final_test_scores
            and history[1].get("test_score") == scores.final_test
        )

        traces = Path(env.traces_dir)
        outputs: dict[int, list] = {}
        ok = True
        for it, n in ((0, seed_samples), (1, num_samples)):
            for k in range(n):
                rows = list(iter_rows(str(traces / f"iteration_{it}_{test_suffix(k)}")))
                ok = ok and len(rows) == 1
                outputs.setdefault(it, []).append(
                    json.dumps((rows or [{}])[0].get("output"), sort_keys=True, default=str)
                )
            # No stray extra draws beyond the configured count.
            ok = ok and not (traces / f"iteration_{it}_{test_suffix(n)}").exists()
        checks["one_trace_row_per_sample"] = ok
        result["trace_dirs"] = sorted(p.name for p in traces.glob("iteration_*_holdout*"))
        result["identical_samples"] = {
            it: len(set(outs)) == 1 for it, outs in outputs.items() if len(outs) > 1
        }
        result["cache_warning_fired"] = any(
            IDENTICAL_SAMPLES_WARNING.split("%d")[0] in m for m in warned
        )
        result["success"] = all(bool(v) for v in checks.values())
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            _git(workspace, "checkout", "-q", initial_branch)
        except Exception:  # noqa: BLE001
            pass
        env.cleanup()
    return result


__all__ = ["run_test_sampling", "NUM_SAMPLES"]
