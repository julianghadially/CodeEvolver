"""Fake collected run directories for the best-of-population unit tests."""

from __future__ import annotations

import json
from pathlib import Path


def make_run(
    root: Path, run_id: str, *, job_id: str, bench: str = "hover", status: str = "completed",
    final_sha: str = "f" * 40, final_test_score: float | None = 0.8, final_test_scores: list[float] | None = None,
    seed_rows: dict[int, float] | None = None, seed_failed: list[int] | None = None,
    testset_sha256: str = "t" * 64, testset_size: int | None = None, models: dict | None = None,
    with_bundle: bool = True, candidates: list[dict] | None = None, posthoc: dict | None = None,
    final_draw_lines: list[dict] | None = None,
) -> Path:
    arm, _, rest = run_id.partition("-")
    seed = rest.split("-s", 1)[1].split("-")[0] if "-s" in rest else "1"
    d = root / run_id
    d.mkdir(parents=True)
    st = {"run_id": run_id, "bench": bench, "arm": arm, "seed": seed, "job_id": job_id, "status": status,
          "final_candidate": {"git_branch": "b", "git_sha": final_sha}, "final_test_score": final_test_score,
          "final_test_scores": final_test_scores if final_test_scores is not None
          else ([final_test_score] if final_test_score is not None else []),
          "seed_test_score": 0.5, "seed_test_scores": [0.5]}
    (d / "status.json").write_text(json.dumps(st))
    (d / "settings.json").write_text(json.dumps({
        "models": models if models is not None else {"coding_harness_model": "m"},
        "dataset_hashes": {"trainset": "a" * 64, "testset": testset_sha256},
    }))
    if with_bundle:
        (d / "repo.bundle").write_bytes(b"x")
    if candidates is not None:
        (d / "candidates.jsonl").write_text("\n".join(json.dumps(c) for c in candidates) + "\n")
    if posthoc is not None:
        (d / "posthoc_holdout_rescore.json").write_text(json.dumps(posthoc))
    hist = d / ".codeevolver" / "history" / job_id
    hist.mkdir(parents=True)
    n = testset_size or (max(seed_rows) + 1 if seed_rows else 0)
    (hist / "datasets.json").write_text(json.dumps({"testset_sha256": testset_sha256, "testset_size": n,
                                                     "trainset_sha256": "a" * 64}))
    lines = []
    if seed_rows is not None:
        lines.append({"eval_id": "seed", "ts": 1.0, "iteration": 0, "eval_sample": "holdout", "suffix": "holdout",
                      "split": "test", "n_rows": len(seed_rows), "score": sum(seed_rows.values()) / max(1, len(seed_rows)),
                      "row_scores": {str(k): v for k, v in seed_rows.items()},
                      "failed_rows": [str(x) for x in (seed_failed or [])]})
    for extra in final_draw_lines or []:
        lines.append(extra)
    (hist / "row_scores.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
    return d
