#!/usr/bin/env python3
"""Copyright © 2026 440 Labs LLC

Per-researcher fleet configuration.

Every member of the research team runs a DIFFERENT subset of benchmarks off the same
repo, the same images and the same controller. Nothing about "which benchmarks"
or "where are the client repos" may therefore be hardcoded -- both come from the
environment, resolved against the shared catalog in `benchmarks.yaml`:

    export CE_BENCHMARKS=hover,phantomwiki,ragqa      # required: your subset
    export CE_REPO_ROOT=~/Documents/code              # default for all repos
    export CE_REPO_PHANTOMWIKI=/mnt/big/phantom-wiki  # per-benchmark override

Expect up to ~8 benchmarks per person. `slots_per_host` is a concurrency cap,
NOT the benchmark count: with 8 benchmarks and 3 slots a host runs three at a
time and cycles through the rest as slots free up.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CATALOG_PATH = HERE / "benchmarks.yaml"
DEFAULT_REPO_ROOT = "~/Documents/code"


def runs_dir() -> Path:
    """Where collected runs live on the controller: $CE_RUNS_DIR, else ~/ce-runs."""
    return Path(os.environ.get("CE_RUNS_DIR") or Path.home() / "ce-runs").expanduser()


@dataclass(frozen=True)
class Benchmark:
    slug: str
    experiment_dir: str
    repo_dirname: str
    est_hours: float
    needs_retriever: bool
    needs_gpu: bool = False
    assets: tuple[str, ...] = ()
    # Client-repo branch the seed bundle is archived from. `main` for a repo
    # whose seed program lives there; arcagi and paperreview do not (see
    # benchmarks.yaml). Wrong branch = wrong seed program
    seed_branch: str = "main"
    runtime_reqs: tuple[str, ...] = ()

    @property
    def bundle_name(self) -> str:
        return f"{self.slug}.bundle"

    def image(self, registry: str, tag: str) -> str:
        return f"{registry}/codeevolver-runner:{self.slug}-{tag}"

    def repo_path(self) -> Path:
        """Client checkout: CE_REPO_<SLUG> wins, else CE_REPO_ROOT/<repo_dirname>."""
        override = os.environ.get(f"CE_REPO_{self.slug.upper()}")
        if override:
            return Path(override).expanduser()
        root = os.environ.get("CE_REPO_ROOT", DEFAULT_REPO_ROOT)
        return Path(root).expanduser() / self.repo_dirname


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, Benchmark]:
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    out: dict[str, Benchmark] = {}
    for slug, spec in raw.items():
        out[slug] = Benchmark(
            slug=slug,
            experiment_dir=spec.get("experiment_dir", slug),
            repo_dirname=spec.get("repo_dirname", slug),
            est_hours=float(spec.get("est_hours", 24)),
            needs_retriever=bool(spec.get("needs_retriever", False)),
            needs_gpu=bool(spec.get("needs_gpu", False)),
            assets=tuple(spec.get("assets") or ()),
            seed_branch=str(spec.get("seed_branch") or "main"),
            runtime_reqs=tuple(spec.get("runtime_reqs") or ()),
        )
    return out


def selected_benchmarks(
    explicit: str | None = None, catalog: dict[str, Benchmark] | None = None
) -> list[Benchmark]:
    """Resolve this researcher's benchmark subset from CE_BENCHMARKS.

    Deliberately has no default: silently running someone else's benchmark set
    would burn days of provider spend before anyone noticed.
    """
    catalog = catalog if catalog is not None else load_catalog()
    raw = explicit if explicit is not None else os.environ.get("CE_BENCHMARKS", "")
    slugs = [s.strip() for s in raw.replace(" ", ",").split(",") if s.strip()]

    if not slugs:
        raise SystemExit(
            "CE_BENCHMARKS is not set. Choose your subset from "
            f"{', '.join(sorted(catalog))} — e.g.\n"
            "    export CE_BENCHMARKS=hover,phantomwiki,ragqa"
        )
    unknown = [s for s in slugs if s not in catalog]
    if unknown:
        raise SystemExit(
            f"unknown benchmark(s): {', '.join(unknown)}\n"
            f"known: {', '.join(sorted(catalog))}\n"
            "Add an entry to orchestrator/benchmarks.yaml (and an "
            "experiments/<experiment_dir>/input.py) first."
        )
    # Preserve catalog order so run ordering is stable regardless of how the
    # env var was typed.
    return [catalog[s] for s in catalog if s in slugs]


def needs_retriever(benchmarks: list[Benchmark]) -> bool:
    """True when this host's subset includes a benchmark with a retriever sidecar."""
    return any(b.needs_retriever for b in benchmarks)


def needs_gpu(benchmarks: list[Benchmark]) -> bool:
    """True when this host's subset includes a benchmark needing GPU + X display."""
    return any(b.needs_gpu for b in benchmarks)


def describe(benchmarks: list[Benchmark]) -> str:
    rows = [
        f"  {b.slug:<14} experiments/{b.experiment_dir:<18} "
        f"{str(b.repo_path()):<48} ~{b.est_hours:g}h"
        # Surfaced only when it is not `main`: a non-default seed branch is the
        # one thing about a benchmark that silently builds the wrong bundle.
        f"{'  [seed: ' + b.seed_branch + ']' if b.seed_branch != 'main' else ''}"
        f"{'  [retriever]' if b.needs_retriever else ''}"
        f"{'  [gpu]' if b.needs_gpu else ''}"
        for b in benchmarks
    ]
    return "\n".join(rows)


if __name__ == "__main__":
    import sys

    benches = selected_benchmarks()
    if len(sys.argv) > 1 and sys.argv[1] == "--slugs":
        print(" ".join(b.slug for b in benches))
    elif len(sys.argv) > 1 and sys.argv[1] == "--needs-retriever":
        print("yes" if needs_retriever(benches) else "no")
    elif len(sys.argv) > 1 and sys.argv[1] == "--needs-gpu":
        print("yes" if needs_gpu(benches) else "no")
    else:
        print(f"CE_BENCHMARKS -> {len(benches)} benchmark(s):")
        print(describe(benches))
        missing = [b for b in benches if not b.repo_path().exists()]
        if missing:
            print("\nWARNING: client repo not found for: "
                  + ", ".join(f"{b.slug} ({b.repo_path()})" for b in missing))
