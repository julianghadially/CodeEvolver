#!/usr/bin/env python3
"""Copyright © 2026 440 Labs LLC

Build seed bundles and runner images for THIS researcher's benchmark subset.

Driven entirely by CE_BENCHMARKS + CE_REPO_ROOT / CE_REPO_<SLUG> (see
fleet_config.py), so two people running different benchmarks share this script
with no edits and no hardcoded home directories.

    export CE_BENCHMARKS=hover,phantomwiki,ragqa
    python orchestrator/build.py bundles
    python orchestrator/build.py images --push
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from fleet_config import Benchmark, needs_retriever, selected_benchmarks

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = REPO_ROOT / "runner" / "bundles"
ASSET_DIR = REPO_ROOT / "runner" / "assets"
RUNTIME_REQS_DIR = REPO_ROOT / "runner" / "runtime-reqs"
# Overridable for a fleet that must not touch the committed pin (a rebuilt
# image under its own tag beside live fleets): controller.py reads the same
# CE_IMAGE_LOCK, so `ce build` pins into, and `dispatch()` resolves from, one
# file.
LOCK_PATH = Path(os.environ.get(
    "CE_IMAGE_LOCK", REPO_ROOT / "orchestrator" / "images.lock.json"
))

# Fleet-owned interpreter shims staged into the same per-benchmark context
# dir as the runtime_reqs manifests. These live under runner/ in THIS repo
# (not the client repo), so they are keyed by slug here rather than declared
# in benchmarks.yaml, whose paths are all client-repo-relative.
RUNNER_SHIMS = {"alfworld": ("alfworld-python",)}


def stage_runtime_reqs(b: Benchmark) -> Path:
    """Stage the benchmark's declared runtime manifests + fleet shims.

    Catalog-driven, exactly like `stage_assets`: only files listed under
    `runtime_reqs:` in orchestrator/benchmarks.yaml are staged, and a declared
    file that is missing fails the build loudly. (Probing every client repo
    for a fixed filename would silently match generic names -- a benchmark's
    own requirements.txt -- in repos that have no extra runtime at all.)

    The per-benchmark directory is recreated from scratch each time so a file
    staged by an earlier build cannot linger, and always exists even when
    empty: Dockerfile.bench COPYs it unconditionally.
    """
    dest_root = RUNTIME_REQS_DIR / b.slug
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True)
    (dest_root / ".keep").touch()
    repo = b.repo_path()
    for name in b.runtime_reqs:
        src = repo / name
        if not src.is_file():
            raise SystemExit(
                f"{b.slug}: declared runtime manifest not found at {src}\n"
                f"It is listed under `runtime_reqs:` in orchestrator/benchmarks.yaml."
            )
        shutil.copy2(src, dest_root / name)
    for name in RUNNER_SHIMS.get(b.slug, ()):
        shutil.copy2(REPO_ROOT / "runner" / name, dest_root / name)
    return dest_root


def docker_cmd(host: str | None) -> list[str]:
    """`docker`, optionally aimed at a remote daemon over SSH.

    The fleet's rule is that nothing is installed on a worker -- the controller
    drives it through `docker --context`. The same applies to a build droplet:
    pointing the build at an amd64 context runs it natively there while the
    bundles, the staged assets and the client checkouts all stay on this
    machine. Docker ships the build context over the SSH connection, so the
    droplet needs no CodeEvolver clone, no client repos, and no copy of the
    gitignored data assets -- only Docker and an authorized key.
    """
    return ["docker"] if not host else ["docker", "--context", host]


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    env = {**os.environ, "DOCKER_BUILDKIT": "1"}
    return subprocess.run(cmd, check=kw.pop("check", True), env=env, **kw)


def out(cmd: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def seed_branch_for(b: Benchmark, override: str | None) -> str:
    """Which client-repo branch to archive for `b`.

    The catalog is the default source (`benchmarks.yaml::seed_branch`) because
    the right branch is a property of the benchmark, not of the operator's
    shell history: `main` is wrong for arcagi (an older solver lineage with no
    trainval pool) and does not exist at all for paperreview. An explicit
    `--branch` still wins, but it applies to EVERY benchmark in the build, so
    it is for one-off sweeps only.
    """
    return override or b.seed_branch


def build_bundle(b: Benchmark, branch: str) -> Path:
    """Squash the client repo to ONE orphan commit and bundle it.

    A contamination control, not packaging convenience. The LangProBe forks
    committed the holdout test sets under simple_eval/results; deleting the
    files left the blobs reachable in history, and the deny-path fences match
    file paths and Bash command text, not blobs. A single orphan commit removes
    that channel and drops prior runs' codeevolver-* branches at the same time.
    """
    repo = b.repo_path()
    if not repo.is_dir():
        raise SystemExit(
            f"{b.slug}: client repo not found at {repo}\n"
            f"Set CE_REPO_ROOT, or CE_REPO_{b.slug.upper()}=<path> for this one."
        )

    # A fresh `git clone` creates a local branch for the remote HEAD only, so on
    # a new machine `seed-program-deepseek-v4flash` exists solely as
    # `origin/seed-program-deepseek-v4flash` and a bare `rev-parse` fails. That
    # is the FIRST thing a new operator hits, and "branch does not exist" would
    # send them off to edit benchmarks.yaml for a branch that is really there.
    # Fall back to the remote-tracking ref, and archive whichever one resolved.
    for ref in (branch, f"origin/{branch}"):
        try:
            short = out(["git", "rev-parse", "--short", ref], cwd=repo)
            break
        except subprocess.CalledProcessError:
            continue
    else:
        # Genuinely absent under both spellings: a fork whose seed program lives
        # on a differently-named branch, or a checkout that was never fetched.
        # The raw `fatal: Needed a single revision` names neither the repo nor
        # the benchmark, so spell out both and the two ways out.
        raise SystemExit(
            f"{b.slug}: branch {branch!r} does not exist in {repo} "
            f"(tried {branch!r} and 'origin/{branch}')\n"
            f"Run `git -C {repo} fetch --all` if it is simply not fetched yet, "
            f"else fix `seed_branch` for {b.slug} in "
            f"orchestrator/benchmarks.yaml, or pass --branch."
        ) from None
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    dest = BUNDLE_DIR / b.bundle_name

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # `git archive` exports the TREE only -- no history comes across, which
        # is the whole point. It also honours .gitignore-tracked state exactly:
        # only committed files appear, so a stray local venv cannot ride along.
        archive = tmp_path / "tree.tar"
        with archive.open("wb") as fh:
            # `ref`, not `branch`: on a fresh clone the tree lives only under
            # the remote-tracking spelling that rev-parse settled on above.
            subprocess.run(["git", "archive", "--format=tar", ref],
                           cwd=repo, stdout=fh, check=True)
        work = tmp_path / "work"
        work.mkdir()
        sh(["tar", "-xf", str(archive), "-C", str(work)])

        sh(["git", "init", "-q", "-b", "seed", str(work)])
        # git add --force (or -f) stages files even if they match .gitignore.
        # Some trees include files that would not pass the .gitignore, and
        # e.g., `*.txt` vs requirements.txt
        sh(["git", "-C", str(work), "add", "-A", "--force"])
        sh(["git", "-C", str(work),
            "-c", "user.email=fleet@codeevolver", "-c", "user.name=fleet",
            "commit", "-q", "-m", f"seed: {b.slug} @ {short}"])
        depth = out(["git", "-C", str(work), "rev-list", "--count", "HEAD"])
        if depth != "1":
            raise SystemExit(f"{b.slug}: expected history depth 1, got {depth}")
        dest.unlink(missing_ok=True)
        sh(["git", "-C", str(work), "bundle", "create", str(dest), "seed"])

    size_mb = dest.stat().st_size / 1e6
    print(f"  {b.slug}: {dest.name}  depth=1  {size_mb:.1f} MB  (from {branch} @ {short})")
    return dest


def stage_assets(b: Benchmark) -> Path:
    """Copy the benchmark's declared data assets into the image build context.

    `build_bundle` exports the COMMITTED tree, so a file the client repo
    gitignores cannot ride in the bundle -- and the ones this covers are
    gitignored precisely because they are far too large to commit. They travel
    as an image layer instead, and `runner/entrypoint.py` places them into the
    workspace after the seed clone.

    The per-benchmark directory is always created, empty or not: Dockerfile.bench
    COPYs it unconditionally and a missing directory fails the build.
    """
    dest_root = ASSET_DIR / b.slug
    dest_root.mkdir(parents=True, exist_ok=True)
    (dest_root / ".keep").touch()
    if not b.assets:
        return dest_root

    repo = b.repo_path()
    for rel in b.assets:
        src = repo / rel
        if not src.is_file():
            raise SystemExit(
                f"{b.slug}: declared asset not found at {src}\n"
                f"It is listed under `assets:` in orchestrator/benchmarks.yaml."
            )
        dest = dest_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        st = src.stat()
        if dest.exists():
            dst_st = dest.stat()
            if (dst_st.st_size, int(dst_st.st_mtime)) == (st.st_size, int(st.st_mtime)):
                print(f"  {b.slug}: asset {rel} up to date ({st.st_size / 1e6:.0f} MB)")
                continue
        print(f"  {b.slug}: staging asset {rel} ({st.st_size / 1e6:.0f} MB)")
        shutil.copy2(src, dest)
    return dest_root


def record_digest(b: Benchmark, image: str, host: str | None) -> str | None:
    """Pin the pushed image by content digest in images.lock.json.

    A tag is a mutable pointer: `latest` means "whatever was pushed last", and
    the registry is shared across the team. Between arm 1 and arm 3 -- weeks --
    a colleague's push, a re-provisioned droplet or a pruned image is enough for
    `docker run <tag>` to fetch different bytes than the arm before it, and
    nothing in the run would show that it happened. `dispatch()` prefers this
    digest, so the matrix is frozen against the registry rather than against
    everyone's discipline. Re-running `ce build` is what moves it.
    """
    res = subprocess.run(
        [*docker_cmd(host), "image", "inspect", image,
         # `println` rather than `join` with a "\n" literal: Python turns the
         # escape into a real newline inside the Go template, the template
         # fails to parse, stdout is empty, and every image "has no digest".
         "--format", "{{range .RepoDigests}}{{println .}}{{end}}"],
        capture_output=True, text=True,
    )
    repo = image.rsplit(":", 1)[0]
    digest = next(
        (d for d in res.stdout.split() if d.startswith(repo + "@")), None
    )
    if not digest:
        print(f"  {b.slug}: WARNING no digest found; dispatch falls back to the tag")
        return None
    lock = {}
    if LOCK_PATH.exists():
        lock = json.loads(LOCK_PATH.read_text() or "{}")
    lock[b.slug] = {"image": image, "digest": digest}
    LOCK_PATH.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(f"  {b.slug}: pinned {digest.rsplit('@', 1)[1][:19]}...")
    return digest


def build_image(b: Benchmark, registry: str, tag: str, push: bool,
                platform: str = "linux/amd64", host: str | None = None) -> None:
    image = b.image(registry, tag)
    bundle = BUNDLE_DIR / b.bundle_name
    if not bundle.exists():
        raise SystemExit(f"{b.slug}: missing {bundle} — run `build.py bundles` first")
    # Idempotent: re-copies only what changed, but guarantees the COPY target
    # exists when `images` is run without `bundles` first.
    stage_assets(b)
    stage_runtime_reqs(b)
    sh([*docker_cmd(host), "build", "--platform", platform, "-f", "Dockerfile.bench",
        "--build-arg", f"BENCH={b.slug}", "-t", image, "."],
       cwd=REPO_ROOT / "runner")
    if push:
        # Pushed BY the build daemon, so a remote builder needs its own registry
        # login (`doctl registry login` on the droplet). Nothing is pulled back
        # to this machine.
        sh([*docker_cmd(host), "push", image])
        # Only after a push: an unpushed image has no registry digest, and a
        # local-only build (matrix.local.yaml) wants the tag anyway.
        record_digest(b, image, host)


def retriever_repo() -> Path:
    """The ragqa-retriever-server checkout: CE_REPO_RETRIEVER, else next to the
    client repos under CE_REPO_ROOT."""
    override = os.environ.get("CE_REPO_RETRIEVER")
    if override:
        return Path(override).expanduser()
    root = os.environ.get("CE_REPO_ROOT", "~/Documents/code")
    return Path(root).expanduser() / "ragqa-retriever-server"


def build_retriever(registry: str, tag: str, push: bool,
                    platform: str = "linux/amd64", host: str | None = None) -> None:
    """The ragqa retriever sidecar image, `<registry>/ragqa-retriever:<tag>`.

    Not a benchmark layer: compose.host.yml starts one per host that runs ragqa
    (provision.sh, `--profile ragqa`), and nothing else produces the image it
    pulls. The 3.7 GB index is NOT baked in -- the sidecar mounts /data/ragqa,
    which provision.sh populates -- so this image is ~1 GB of torch and rebuilds
    only when the server code changes.
    """
    repo = retriever_repo()
    if not (repo / "Dockerfile").is_file():
        raise SystemExit(
            f"retriever: no Dockerfile at {repo}\n"
            f"Set CE_REPO_RETRIEVER=<path to ragqa-retriever-server>."
        )
    image = f"{registry}/ragqa-retriever:{tag}"
    sh([*docker_cmd(host), "build", "--platform", platform, "-t", image, "."], cwd=repo)
    if push:
        sh([*docker_cmd(host), "push", image])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target",
                    choices=["bundles", "base", "images", "retriever", "pin", "all"])
    # Default: $REGISTRY, the same value the controller and provision.sh use.
    ap.add_argument("--registry", default=os.environ.get("REGISTRY"))
    ap.add_argument("--tag", default="latest")
    ap.add_argument(
        "--branch", default=None,
        help="client-repo branch to seed from, for EVERY benchmark in this "
             "build. Default: each benchmark's `seed_branch` from "
             "benchmarks.yaml (itself defaulting to main).",
    )
    ap.add_argument("--push", action="store_true")
    ap.add_argument(
        "--no-retriever", action="store_true",
        help="skip the ragqa-retriever sidecar image even when ragqa is in the "
             "subset. Its inputs (pyproject/uv.lock/server code) change rarely "
             "and a rebuild re-resolves torch against whatever python:3.11-slim "
             "and uv:latest are today, so rebuild it only when they changed.",
    )
    # DigitalOcean droplets are x86_64. Building on an Apple-Silicon Mac without
    # this produces arm64 images that will not run on the fleet at all -- and
    # MongoDB publishes no arm64 Debian packages, so the base image cannot even
    # build natively there. Emulated builds work but are slow: prefer an amd64
    # build droplet when rebuilding the torch-heavy benchmark layers.
    ap.add_argument("--platform", default="linux/amd64")
    ap.add_argument(
        "--host", default=os.environ.get("CE_BUILD_HOST") or None,
        help="docker context to build on (e.g. an amd64 build droplet). "
             "Default: the local daemon.",
    )
    args = ap.parse_args()
    if args.target != "bundles" and not args.registry:
        ap.error("set REGISTRY (or pass --registry): images are named <registry>/codeevolver-runner:<bench>-<tag>")

    benches = selected_benchmarks()
    where = f" on context {args.host}" if args.host else ""
    print(f"benchmarks: {', '.join(b.slug for b in benches)}{where}\n")

    if args.target == "pin":
        # Re-record the pushed digests without building: for images that were
        # pushed by a build whose pin step failed, or pushed by hand.
        print("== pin pushed digests ==")
        for b in benches:
            record_digest(b, b.image(args.registry, args.tag), args.host)
        return 0

    if args.target in ("bundles", "all"):
        print("== seed bundles ==")
        for b in benches:
            build_bundle(b, seed_branch_for(b, args.branch))
            stage_assets(b)
            stage_runtime_reqs(b)

    if args.target in ("base", "images", "all"):
        print("\n== base image ==")
        sh([*docker_cmd(args.host), "build", "--platform", args.platform,
            "-f", "runner/Dockerfile.base",
            "-t", "codeevolver-runner:base", "."], cwd=REPO_ROOT)

    if args.target in ("images", "all"):
        print("\n== benchmark images ==")
        for b in benches:
            build_image(b, args.registry, args.tag, args.push, args.platform,
                        args.host)

    # Rides along with `images` only when the subset needs it, so a colleague
    # who never runs ragqa never builds torch for a sidecar they will not start.
    if args.target == "retriever" or (
        args.target in ("images", "all") and needs_retriever(benches)
        and not args.no_retriever
    ):
        print("\n== ragqa retriever image ==")
        build_retriever(args.registry, args.tag, args.push, args.platform, args.host)

    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
