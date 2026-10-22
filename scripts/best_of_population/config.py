"""Copyright © 2026 440 Labs LLC

Audit configuration + per-benchmark spec (program, metric, dataset paths).

`BenchSpec` is assembled from two existing sources so nothing is duplicated:
`orchestrator/benchmarks.yaml` (via `orchestrator.fleet_config`) for the
client checkout, experiment dir and gitignored assets, and
`experiments/<experiment_dir>/input.py::OPTIMIZE_CONFIG_LOCAL` for the
program factory, metric, dataset paths, input keys and thread count —
exactly what a research run submitted.

Two config objects: `AuditConfig` is what the CLI persists (`config.json`);
`AnalysisConfig` is the smaller dict the evaluation driver ships to its
worker processes (`evaluate._worker_main` rebuilds it from a plain dict).
"""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from orchestrator.fleet_config import Benchmark, load_catalog
from src.engine.evaluator.config import EvaluatorConfig

DEFAULT_ARMS = ("freeform", "pareto_evolution")
CONFIG_FILENAME = "config.json"
DEFAULT_OUT = str(Path.home() / "ce-analysis" / "bop")


@dataclass(frozen=True)
class BenchSpec:
    slug: str
    experiment_dir: str
    repo_path: str
    assets: tuple[str, ...]
    program_factory: str
    metric_fn: str
    trainset_path: str
    testset_path: str
    input_keys: tuple[str, ...] | None
    num_threads: int
    comparison_set_size: int
    subsample_eval_timeout: int
    comparison_eval_timeout: int
    python_version: str | None = None

    def evaluator_config(self) -> EvaluatorConfig:
        """Mirror of `optimizer_loop.py`'s single-metric bridge."""
        return EvaluatorConfig(
            program_factory=self.program_factory,
            metric_fns=[self.metric_fn],
            metric_names=["accuracy"],
            primary_metric="accuracy",
            num_threads=self.num_threads,
            trainset_path=self.trainset_path,
            testset_path=self.testset_path or None,
            input_keys=list(self.input_keys) if self.input_keys else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_bench_spec(
    slug: str, *, config_name: str = "OPTIMIZE_CONFIG_LOCAL", catalog: dict[str, Benchmark] | None = None
) -> BenchSpec:
    catalog = catalog or load_catalog()
    if slug not in catalog:
        raise KeyError(f"unknown benchmark slug {slug!r}; known: {sorted(catalog)}")
    bench = catalog[slug]
    module = importlib.import_module(f"experiments.{bench.experiment_dir}.input")
    cfg = getattr(module, config_name)
    return BenchSpec(
        slug=slug,
        experiment_dir=bench.experiment_dir,
        repo_path=str(bench.repo_path()),
        assets=tuple(bench.assets),
        program_factory=cfg["program"],
        metric_fn=cfg["metric"],
        trainset_path=cfg["trainset_path"],
        testset_path=cfg.get("testset_path") or "",
        input_keys=tuple(cfg["input_keys"]) if cfg.get("input_keys") else None,
        num_threads=int(cfg.get("num_threads", 8)),
        comparison_set_size=int(cfg.get("comparison_set_size") or 0),
        subsample_eval_timeout=int(cfg.get("subsample_eval_timeout", 7200)),
        comparison_eval_timeout=int(cfg.get("comparison_eval_timeout", 14400)),
        python_version=cfg.get("python_version"),
    )


@dataclass
class AnalysisConfig:
    """The evaluation driver's config (what `execute_tasks` ships to workers)."""

    out: str
    runs_home: str = str(Path(os.environ.get("CE_RUNS_DIR") or Path.home() / "ce-runs").expanduser())
    data_runs: str = str(Path.home() / "ce-data" / "runs")
    ce_home: str = str(Path.home() / ".codeevolver")
    benches: list[str] = field(default_factory=list)
    arms: list[str] = field(default_factory=lambda: list(DEFAULT_ARMS))
    include_seed: bool = False
    workers: int = 1
    dotenv: str | None = None
    config_name: str = "OPTIMIZE_CONFIG_LOCAL"

    @property
    def out_path(self) -> Path:
        return Path(self.out).expanduser()

    def with_overrides(self, **kw: Any) -> "AnalysisConfig":
        data = asdict(self)
        for k, v in kw.items():
            if v is not None:
                data[k] = v
        return AnalysisConfig(**data)


@dataclass
class AuditConfig:
    """What the CLI persists in `<out>/config.json`."""

    out: str = DEFAULT_OUT
    runs_roots: list[str] = field(default_factory=lambda: [str(Path(os.environ.get("CE_RUNS_DIR") or Path.home() / "ce-runs").expanduser())])
    ce_home: str = str(Path.home() / ".codeevolver")
    benches: list[str] = field(default_factory=list)
    arms: list[str] = field(default_factory=lambda: list(DEFAULT_ARMS))
    workers: int = 1
    dotenv: str | None = None
    config_name: str = "OPTIMIZE_CONFIG_LOCAL"
    reps_select: int = 5
    reps_confirm: int = 5
    top_k: int = 5
    threshold: float = 0.5
    eval_timeout: int | None = None       # None -> the bench's comparison_eval_timeout
    include_void_seed: bool = False

    @property
    def out_path(self) -> Path:
        return Path(self.out).expanduser()

    def evaluator_cfg(self) -> AnalysisConfig:
        return AnalysisConfig(
            out=str(self.out_path), runs_home=self.runs_roots[0] if self.runs_roots else "",
            ce_home=self.ce_home, benches=list(self.benches), arms=list(self.arms),
            workers=self.workers, dotenv=self.dotenv, config_name=self.config_name,
        )

    def save(self) -> Path:
        path = self.out_path / CONFIG_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, out: str) -> "AuditConfig":
        path = Path(out).expanduser() / CONFIG_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run `select-run` first (it writes the config)")
        data = json.loads(path.read_text())
        data["out"] = str(Path(out).expanduser())
        return cls(**data)

    def with_overrides(self, **kw: Any) -> "AuditConfig":
        data = asdict(self)
        for k, v in kw.items():
            if v is not None:
                data[k] = v
        return AuditConfig(**data)


def analysis_paths(cfg: AnalysisConfig | AuditConfig, bench: str) -> dict[str, Path]:
    """Evaluation-side paths (worktrees, venvs, traces, the row store)."""
    root = cfg.out_path
    return {
        "root": root,
        "bare_repo": root / "repos" / f"{bench}.git",
        "worktrees": root / "worktrees" / bench,
        "venvs": root / "venvs",
        "traces": root / "traces" / bench,
        "scores_dir": root / "scores" / bench,
        "rows": root / "scores" / bench / "rows.jsonl",
        "evals": root / "scores" / bench / "evals.jsonl",
        "attempts": root / "scores" / bench / "attempts.jsonl",
        "materialize": root / "materialize" / f"{bench}.json",
    }


def audit_paths(cfg: AuditConfig, bench: str) -> dict[str, Path]:
    """Frozen-artifact and report paths for one benchmark."""
    root = cfg.out_path
    p = analysis_paths(cfg, bench)
    p.update({
        "selection": root / "selection" / f"{bench}.json",
        "difficult": root / "difficult" / f"{bench}.json",
        "results_dir": root / "results",
        "logs": root / "logs",
        "summary_csv": root / "results" / "summary.csv",
        "summary_md": root / "results" / "summary.md",
        "summary_json": root / "results" / "summary.json",
    })
    return p


def arm_path(cfg: AuditConfig, kind: str, bench: str, arm: str) -> Path:
    """`candidates` | `portfolio` | `winners` | `results` | `rows_csv` for one (bench, arm)."""
    root = cfg.out_path
    if kind == "rows_csv":
        return root / "results" / f"{bench}-{arm}-rows.csv"
    return root / kind / f"{bench}-{arm}.json"


def env_preamble(cfg: AnalysisConfig | AuditConfig | None = None) -> None:
    """Process-wide env the engine's evaluator path expects. Idempotent."""
    os.environ.setdefault("SANDBOX_MODE", "local")
    os.environ.setdefault("APP_MODE", "local")
    if cfg and cfg.dotenv:
        from src.engine.optimizer.v3._entry import _load_dotenv

        _load_dotenv(os.path.expanduser(cfg.dotenv))


__all__ = [
    "AnalysisConfig",
    "AuditConfig",
    "BenchSpec",
    "CONFIG_FILENAME",
    "DEFAULT_ARMS",
    "DEFAULT_OUT",
    "analysis_paths",
    "arm_path",
    "audit_paths",
    "env_preamble",
    "load_bench_spec",
]
