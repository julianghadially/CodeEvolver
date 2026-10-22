"""Copyright © 2026 440 Labs LLC

Deterministic coverage for the experiment-fleet layer (runner/ + orchestrator/).

These are the behaviours a 6-day, 45-run matrix depends on and that are
expensive to discover live: the config-override contract the containerized
runner submits through, cooperative cancellation, and the scheduler's
slot/queue/assignment arithmetic.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# `experiments.run` resolves its API base URL and key at IMPORT time and, under
# the default APP_MODE=prod, raises KeyError('PROD_API_KEY'). These are offline
# unit tests, so pin local mode before the lazy imports inside the tests below
# -- otherwise the suite only passes when the caller happens to export APP_MODE.
os.environ.setdefault("APP_MODE", "local")
# The fleet is per-researcher: benchmarks come from CE_BENCHMARKS, never from a
# committed file. Pin (not setdefault) the three-benchmark subset these tests
# assert on -- the whole point of CE_BENCHMARKS is that every researcher exports
# a DIFFERENT one, so `setdefault` let an ambient `CE_BENCHMARKS=hover` silently
# turn "36 runs = 3 benchmarks x 4 arms x 3 seeds" into 12 and fail four tests
# for reasons that have nothing to do with the code under test.
os.environ["CE_BENCHMARKS"] = "hover,phantomwiki,ragqa"
# The registry is deployment config ($REGISTRY), not part of the committed
# matrices; pin a fixed one so image names are deterministic here.
os.environ["REGISTRY"] = "registry.example.com/ce"
# Scheduling tests run against a fixed 5-host, 3-slot fixture matrix rather
# than a committed experiment, so editing an experiment cannot break them.
os.environ["CE_MATRIX"] = str(
    Path(__file__).resolve().parent / "test_data" / "fleet_matrix.yaml"
)
# The controller tees its logs to CE_FLEET_LOG (default
# ~/.codeevolver/logs/fleet.log); resolve it at controller import time to a
# throwaway so unit runs never write into the operator's real fleet log.
import tempfile as _tempfile  # noqa: E402

os.environ.setdefault(
    "CE_FLEET_LOG", os.path.join(_tempfile.gettempdir(), "ce-test-fleet.log")
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "orchestrator"))
sys.path.insert(0, str(REPO_ROOT / "runner"))


# ---------------------------------------------------------------------------
# experiments/run.py — the contract the runner container submits through
# ---------------------------------------------------------------------------

def test_apply_overrides_parses_json_but_keeps_bare_strings():
    from experiments.run import apply_overrides

    out = apply_overrides(
        {"repo_url": "/Users/me/repo", "num_threads": 25},
        ["repo_url=/workspace", "debug_max_iterations=2", "enable_meta_reflection=false"],
    )
    # A path must survive as a string, not become a JSON parse error.
    assert out["repo_url"] == "/workspace"
    assert out["debug_max_iterations"] == 2 and isinstance(out["debug_max_iterations"], int)
    assert out["enable_meta_reflection"] is False
    assert out["num_threads"] == 25, "untouched keys must pass through"


def test_apply_overrides_rejects_malformed_pairs():
    from experiments.run import apply_overrides

    with pytest.raises(SystemExit):
        apply_overrides({}, ["repo_url"])


def test_overrides_apply_before_required_key_validation():
    """`--set repo_url=...` must be able to SATISFY a missing required key.

    The runner never edits experiments/<bench>/input.py; it points repo_url at
    the container's /workspace clone. If validation ran first, a config that
    omits repo_url would fail before the override could land.
    """
    from experiments.run import resolve_config

    module = types.SimpleNamespace(
        OPTIMIZE_CONFIG_LOCAL={
            "program": "p.Prog", "metric": "p.metric",
            "trainset_path": "data/train.json",
        }
    )
    name, config = resolve_config(
        module, "hover", "OPTIMIZE_CONFIG_LOCAL", ["repo_url=/workspace"]
    )
    assert name == "OPTIMIZE_CONFIG_LOCAL"
    assert config["repo_url"] == "/workspace"


def test_resolve_config_still_rejects_genuinely_missing_keys():
    from experiments.run import resolve_config

    module = types.SimpleNamespace(OPTIMIZE_CONFIG_LOCAL={"program": "p.Prog"})
    with pytest.raises(SystemExit):
        resolve_config(module, "hover", "OPTIMIZE_CONFIG_LOCAL", ["repo_url=/workspace"])


# ---------------------------------------------------------------------------
# POST /job/{id}/cancel — the writer the cooperative cancel path was missing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_marks_job_cancelled():
    """`check-cancelled` already existed and was polled every iteration; nothing
    ever WROTE the status. This is that writer."""
    from src.api.main import cancel_job

    user = MagicMock(clerk_id="local-operator")
    job = {"job_id": "job_1", "status": "running", "clerk_id": "local-operator"}
    db = MagicMock()
    db.jobs.find_one = AsyncMock(return_value=job)

    with patch("src.api.main.get_database", return_value=db), \
         patch("src.api.main.record_job_status_update", new=AsyncMock()) as writer:
        resp = await cancel_job("job_1", user)

    assert resp.status == "cancelled" and resp.already_terminal is False
    writer.assert_awaited_once()
    assert writer.await_args.args[1] == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
async def test_cancel_is_a_noop_on_terminal_jobs(status):
    """Idempotent: the controller may cancel a run that just finished on its own."""
    from src.api.main import cancel_job

    user = MagicMock(clerk_id="local-operator")
    db = MagicMock()
    db.jobs.find_one = AsyncMock(
        return_value={"job_id": "job_1", "status": status, "clerk_id": "local-operator"}
    )

    with patch("src.api.main.get_database", return_value=db), \
         patch("src.api.main.record_job_status_update", new=AsyncMock()) as writer:
        resp = await cancel_job("job_1", user)

    assert resp.already_terminal is True and resp.status == status
    writer.assert_not_awaited()


# ---------------------------------------------------------------------------
# orchestrator matrix + scheduler
# ---------------------------------------------------------------------------

def _controller():
    import controller
    return controller


def _fleet_config():
    import fleet_config
    return fleet_config


def _catalog_with_extras(*slugs: str) -> dict:
    """The live catalog plus synthetic entries, for "a teammate runs more".

    Tests about SELECTION mechanics (env-driven subsets, slot capping) need
    more benchmark names than the fleet currently ships, but they must not
    assert on which ones are live: `benchmarks.yaml` is a catalog that gains
    and loses entries as the research matrix moves, and pinning tests to it
    means every catalog edit turns unrelated tests red. (It did: `factchecker`
    and `hotpot` were hardcoded here and broke when the catalog shrank to
    three.) Synthesising the extras keeps the mechanics under test and the
    catalog free to change.
    """
    fc = _fleet_config()
    catalog = dict(fc.load_catalog())
    for slug in slugs:
        catalog[slug] = fc.Benchmark(
            slug=slug,
            experiment_dir=slug,
            repo_dirname=f"{slug}-repo",
            est_hours=24,
            needs_retriever=False,
        )
    return catalog


def test_matrix_expands_to_the_full_balanced_run_set():
    C = _controller()
    matrix = C.load_matrix()
    runs = C.build_runs(matrix)

    assert len(runs) == 45, "3 benchmarks x 5 arms x 3 seeds"
    assert len({r.run_id for r in runs}) == 45, "run ids must be unique"
    # Five arms on five hosts, one arm each: the load is even, so no host is
    # the matrix's critical path.
    per_host = sorted(sum(1 for r in runs if r.host == host) for host in matrix["hosts"])
    assert per_host == [9, 9, 9, 9, 9]
    for arm in matrix["arms"]:
        assert sum(1 for r in runs if r.arm == arm) == 9


def test_each_host_runs_one_arm_and_all_benchmarks_per_seed():
    C = _controller()
    matrix = C.load_matrix()
    runs = C.build_runs(matrix)
    expected = {b.slug for b in _fleet_config().selected_benchmarks()}

    for seed in matrix["seeds"]:
        for host in matrix["hosts"]:
            arm = matrix["assignment"][host]
            sel = [r for r in runs if r.seed == seed and r.host == host]
            assert {r.arm for r in sel} == {arm}
            assert {r.bench for r in sel} == expected


def test_no_two_arms_share_a_host():
    """`ee` and `asa` are the pair this exists for.

    asa IS ee plus enforced population-tree traversal, so the treatment effect
    is the delta between them; co-hosting made them contend for the same CPU,
    RAM, disk and ragqa retriever. One arm per host removes that.
    """
    C = _controller()
    matrix = C.load_matrix()
    assert len(set(matrix["assignment"].values())) == len(matrix["hosts"])

    by_host: dict[str, set[str]] = {}
    for r in C.build_runs(matrix):
        by_host.setdefault(r.host, set()).add(r.arm)
    for host, arms in by_host.items():
        assert len(arms) == 1, f"{host} carries {sorted(arms)}"


def test_load_matrix_rejects_hosts_assignment_drift(tmp_path):
    """Adding a host without assigning it an arm silently drops nothing;
    assigning an arm to an unlisted host silently drops 9 runs. Fail at load."""
    C = _controller()
    path = tmp_path / "matrix.yaml"
    path.write_text(
        "hosts: [ce-a, ce-b]\n"
        "seeds: [1]\n"
        "assignment:\n  ce-a: ee\n  ce-z: asa\n"
    )
    with pytest.raises(ValueError, match="hosts and assignment disagree"):
        C.load_matrix(path)


_ABLATION_MATRIX = (
    "hosts: [ce-h, ce-p]\n"
    "arms: [pareto_evolution, freeform]\n"
    "seeds: [1, 2, 3]\n"
    "assignment:\n"
    "  ce-h: [pareto_evolution, freeform]\n"
    "  ce-p: [pareto_evolution, freeform]\n"
    "host_benchmarks:\n  ce-h: [hover]\n  ce-p: [phantomwiki]\n"
    "slots_per_host: 6\n"
    "defaults: {poll_seconds: 1, rsync_seconds: 999999}\n"
)


def test_list_valued_assignment_puts_both_arms_on_one_host(tmp_path):
    C = _controller()
    path = tmp_path / "matrix.yaml"
    path.write_text(_ABLATION_MATRIX + "bench_concurrency: 6\n")
    runs = C.build_runs(C.load_matrix(path))

    assert len(runs) == 12, "2 benchmarks x 2 arms x 3 seeds"
    for host, bench in (("ce-h", "hover"), ("ce-p", "phantomwiki")):
        mine = [r for r in runs if r.host == host]
        assert {r.bench for r in mine} == {bench}
        assert {(r.arm, r.seed) for r in mine} == {
            (a, s) for a in ("pareto_evolution", "freeform") for s in (1, 2, 3)
        }


def test_load_matrix_rejects_an_assigned_arm_missing_from_arms(tmp_path):
    C = _controller()
    path = tmp_path / "matrix.yaml"
    path.write_text(_ABLATION_MATRIX.replace("[pareto_evolution, freeform]\n  ce-p", "[pareto_evolution, freefrom]\n  ce-p"))
    with pytest.raises(ValueError, match="not in `arms`"):
        C.load_matrix(path)


@pytest.mark.parametrize("cap, per_host", [(6, 6), (None, 1)])
def test_bench_concurrency_caps_runs_of_one_benchmark_per_host(monkeypatch, tmp_path, cap, per_host):
    """Default (absent) keeps one run per benchmark per host; 6 fills the host."""
    C = _controller()
    path = tmp_path / "matrix.yaml"
    path.write_text(_ABLATION_MATRIX + (f"bench_concurrency: {cap}\n" if cap else ""))
    matrix = C.load_matrix(path)
    monkeypatch.setattr(C, "load_matrix", lambda *a, **k: matrix)
    monkeypatch.setattr(C, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(C, "live_containers", lambda host: {})
    monkeypatch.setattr(C, "collect", lambda *a, **k: None)
    dispatched: list[tuple[str, str]] = []
    monkeypatch.setattr(C, "dispatch", lambda m, r: dispatched.append((r.run_id, r.host)))

    args = types.SimpleNamespace(arm=None, bench=None, once=True, drain=False, dry_run=False)
    C.cmd_run(args)

    for host in ("ce-h", "ce-p"):
        assert sum(1 for _, h in dispatched if h == host) == per_host


def _run_scheduler_once(monkeypatch, tmp_path, live: dict[str, dict[str, str]]):
    C = _controller()
    monkeypatch.setattr(C, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(C, "live_containers", lambda host: live[host])
    monkeypatch.setattr(C, "collect", lambda *a, **k: None)
    monkeypatch.setattr(
        C, "finish",
        lambda m, r, st: st["runs"].setdefault(r.run_id, {}).update({"status": "completed"}),
    )
    dispatched: list[tuple[str, str]] = []
    monkeypatch.setattr(C, "dispatch", lambda m, r: dispatched.append((r.run_id, r.host)))

    args = types.SimpleNamespace(arm=None, bench=None, once=True, drain=False, dry_run=False)
    C.cmd_run(args)
    return dispatched


def test_cold_start_fills_every_slot_with_seed_one(monkeypatch, tmp_path):
    live = {h: {} for h in ("ce-a", "ce-b", "ce-c", "ce-d", "ce-e")}
    dispatched = _run_scheduler_once(monkeypatch, tmp_path, live)

    assert len(dispatched) == 15, "5 hosts x 3 slots"
    assert all(rid.endswith("-s1") for rid, _ in dispatched)
    for host in live:
        assert sum(1 for _, h in dispatched if h == host) == 3


def test_a_freed_slot_advances_only_its_own_benchmark_queue(monkeypatch, tmp_path):
    """A 24h hover run must not wait on the 48h phantomwiki run beside it."""
    live = {
        "ce-a": {
            "ce-ee-hover-s1": "exited",
            "ce-ee-phantomwiki-s1": "running",
            "ce-ee-ragqa-s1": "running",
        },
        "ce-b": {}, "ce-c": {}, "ce-d": {}, "ce-e": {},
    }
    dispatched = _run_scheduler_once(monkeypatch, tmp_path, live)

    on_a = [rid for rid, host in dispatched if host == "ce-a"]
    assert len(on_a) == 1, f"two slots are still busy; got {on_a}"
    assert "hover" in on_a[0], "the freed slot belongs to the hover queue"
    # ce-a carries `ee` for every seed, so the hover queue advances to the next
    # seed of the same arm while the two other benchmarks keep their slots.
    assert on_a[0] == "ee-hover-s2"


def test_running_containers_are_never_double_dispatched(monkeypatch, tmp_path):
    live = {
        "ce-a": {
            "ce-ee-hover-s1": "running",
            "ce-ee-phantomwiki-s1": "running",
            "ce-ee-ragqa-s1": "running",
        },
        "ce-b": {}, "ce-c": {}, "ce-d": {}, "ce-e": {},
    }
    dispatched = _run_scheduler_once(monkeypatch, tmp_path, live)
    assert not any(host == "ce-a" for _, host in dispatched)


def test_unreachable_host_is_treated_as_empty_not_fatal(monkeypatch, tmp_path):
    """A momentarily unreachable host must not abort the whole matrix."""
    C = _controller()
    monkeypatch.setattr(C, "STATE_PATH", tmp_path / "state.json")

    def boom(host):
        if host == "ce-c":
            raise OSError("ssh: connect to host ce-c: Connection refused")
        return {}

    monkeypatch.setattr(C, "live_containers", boom)
    runs = C.build_runs(C.load_matrix())
    status = C.reconcile(C.load_matrix(), runs, {"runs": {}})
    assert all(v == "queued" for v in status.values())


def test_drain_dispatches_nothing_new(monkeypatch, tmp_path):
    C = _controller()
    monkeypatch.setattr(C, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(C, "live_containers", lambda host: {})
    monkeypatch.setattr(C, "collect", lambda *a, **k: None)
    dispatched: list = []
    monkeypatch.setattr(C, "dispatch", lambda m, r: dispatched.append(r.run_id))

    args = types.SimpleNamespace(arm=None, bench=None, once=True, drain=True, dry_run=False)
    C.cmd_run(args)
    assert dispatched == []


# ---------------------------------------------------------------------------
# egress canary verdict logic
# ---------------------------------------------------------------------------

def test_preflight_fails_when_a_denied_host_is_reachable(monkeypatch):
    """A fence that fails OPEN must abort the run, not quietly contaminate it."""
    import preflight

    monkeypatch.setattr(
        preflight, "_probe",
        lambda host, port=443, timeout=10.0: {
            "host": host, "reachable": host == "github.com", "via": "proxy", "detail": "",
        },
    )
    monkeypatch.setattr(preflight, "_probe_node", lambda host, timeout=10.0: None)

    report = preflight.check()
    assert report["ok"] is False
    assert "github.com" in report["leaked_hosts"]


def test_preflight_fails_when_a_required_provider_is_unreachable(monkeypatch):
    import preflight

    monkeypatch.setattr(
        preflight, "_probe",
        lambda host, port=443, timeout=10.0: {
            "host": host,
            "reachable": host not in preflight.DENIED_HOSTS and host != "api.deepinfra.com",
            "via": "proxy", "detail": "",
        },
    )
    monkeypatch.setattr(preflight, "_probe_node", lambda host, timeout=10.0: None)

    report = preflight.check()
    assert report["ok"] is False
    assert report["unreachable_allowed_hosts"] == ["api.deepinfra.com"]


def _fenced_probes(monkeypatch, preflight):
    monkeypatch.setattr(
        preflight, "_probe",
        lambda host, port=443, timeout=10.0: {
            "host": host, "reachable": host not in preflight.DENIED_HOSTS,
            "via": "proxy", "detail": "",
        },
    )
    monkeypatch.setattr(
        preflight, "_probe_node",
        lambda host, timeout=10.0: {"host": host, "runtime": "node",
                                    "reachable": False, "detail": "BLOCKED"},
    )


def test_preflight_passes_when_fence_closed_and_providers_reachable(monkeypatch):
    import preflight

    _fenced_probes(monkeypatch, preflight)
    monkeypatch.setattr(
        preflight, "_check_ripgrep", lambda: {"ok": True, "detail": "/usr/bin/rg"}
    )

    report = preflight.check()
    assert report["ok"] is True and not report["leaked_hosts"]
    assert report["ripgrep"]["ok"] is True


def test_preflight_fails_when_agent_cannot_run_git(monkeypatch):
    """A root-owned workspace the agent user cannot `git rev-parse` in means
    every agent-side git command dies with "dubious ownership" — abort before
    spend."""
    import preflight

    _fenced_probes(monkeypatch, preflight)
    monkeypatch.setattr(
        preflight, "_check_ripgrep", lambda: {"ok": True, "detail": "/usr/bin/rg"}
    )
    monkeypatch.setattr(
        preflight, "_check_agent_git",
        lambda workspace=None: {"configured": True, "ok": False,
                                "detail": "git as 'ce-agent' failed: dubious ownership"},
    )

    report = preflight.check()
    assert report["ok"] is False
    assert report["agent_git"]["ok"] is False


def test_preflight_fails_when_ripgrep_is_missing(monkeypatch):
    """OpenCode's grep/glob/skill tools shell out to `rg`. Without it on PATH
    OpenCode tries to download ripgrep from github.com — which the egress fence
    terminates — and every one of those tools returns "ripgrep execution
    failed" for the whole run. Abort before any spend."""
    import preflight

    _fenced_probes(monkeypatch, preflight)
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)

    report = preflight.check()
    assert report["ok"] is False
    assert report["ripgrep"]["ok"] is False
    assert "rg not on PATH" in report["ripgrep"]["detail"]


def test_denied_hosts_cover_the_known_contamination_channels():
    """The LangProBe forks' git history still carries the holdout test sets."""
    import preflight

    assert "github.com" in preflight.DENIED_HOSTS
    assert "raw.githubusercontent.com" in preflight.DENIED_HOSTS
    assert "huggingface.co" in preflight.DENIED_HOSTS
    assert not any("github" in h for h in preflight.ALLOWED_HOSTS)


# ---------------------------------------------------------------------------
# per-researcher benchmark selection
# ---------------------------------------------------------------------------

def test_benchmark_slug_maps_to_its_experiments_directory():
    """The slug is NOT the directory for most benchmarks.

    phantomwiki -> experiments/phantom-wiki and ragqa ->
    experiments/RAGQAArenaTech. `experiments.run --program` takes the directory,
    so a run dispatched with the slug fails at submission.
    """
    fc = _fleet_config()
    catalog = fc.load_catalog()
    assert catalog["phantomwiki"].experiment_dir == "phantom-wiki"
    assert catalog["ragqa"].experiment_dir == "RAGQAArenaTech"

    for b in catalog.values():
        assert (REPO_ROOT / "experiments" / b.experiment_dir).is_dir(), (
            f"{b.slug} points at experiments/{b.experiment_dir}, which does not exist"
        )


def test_dispatch_passes_the_experiments_directory_not_the_slug(monkeypatch):
    C, fc = _controller(), _fleet_config()
    captured: list[list[str]] = []
    monkeypatch.setattr(C, "docker", lambda host, *a, **k: captured.append(list(a)))
    monkeypatch.setattr(C, "ssh", lambda *a, **k: None)
    monkeypatch.setattr(C, "host_addr", lambda m, h: "root@x")

    run = next(r for r in C.build_runs(C.load_matrix()) if r.bench == "phantomwiki")
    C.dispatch(C.load_matrix(), run)

    args = captured[0]
    assert "CE_PROGRAM=phantom-wiki" in args
    assert "CE_BENCH=phantomwiki" in args


def test_selection_comes_from_env_not_from_a_committed_file(monkeypatch):
    """Teammates run different subsets off the same repo."""
    fc = _fleet_config()
    catalog = _catalog_with_extras("factchecker")
    monkeypatch.setenv("CE_BENCHMARKS", "hover,factchecker")
    assert [b.slug for b in fc.selected_benchmarks(catalog=catalog)] == [
        "hover", "factchecker",
    ]


def test_unset_selection_fails_loudly_rather_than_defaulting(monkeypatch):
    """A silent default would burn days of provider spend on the wrong set."""
    fc = _fleet_config()
    monkeypatch.delenv("CE_BENCHMARKS", raising=False)
    with pytest.raises(SystemExit, match="CE_BENCHMARKS is not set"):
        fc.selected_benchmarks()
    with pytest.raises(SystemExit, match="unknown benchmark"):
        fc.selected_benchmarks(explicit="not-a-benchmark")


def test_retriever_is_only_required_when_the_subset_includes_ragqa():
    """A colleague not running ragqa must never start its ~8GB sidecar."""
    fc = _fleet_config()
    catalog = _catalog_with_extras("factchecker")
    assert fc.needs_retriever(
        fc.selected_benchmarks(explicit="hover,factchecker", catalog=catalog)
    ) is False
    assert fc.needs_retriever(
        fc.selected_benchmarks(explicit="hover,ragqa", catalog=catalog)
    ) is True


def test_seed_branch_defaults_to_main_and_is_read_from_the_catalog(tmp_path):
    """`main` is only a DEFAULT; a benchmark may seed from anywhere.

    The default silently built the wrong tree for arcagi (an older solver
    lineage with no trainval pool, so the run has no trainset) and hard-failed
    for paperreview (no `main` branch exists), and `make bundles` passes no
    --branch at all -- so the branch has to live in the shared catalog.
    """
    fc = _fleet_config()
    catalog_file = tmp_path / "benchmarks.yaml"
    catalog_file.write_text(
        "plain:\n"
        "  experiment_dir: plain\n"
        "  repo_dirname: plain-repo\n"
        "offmain:\n"
        "  experiment_dir: offmain\n"
        "  repo_dirname: offmain-repo\n"
        "  seed_branch: seed-program-deepseek-v4flash\n"
    )
    catalog = fc.load_catalog(catalog_file)

    assert catalog["plain"].seed_branch == "main"
    assert catalog["offmain"].seed_branch == "seed-program-deepseek-v4flash"
    # A non-default branch is the one benchmark property that silently produces
    # a wrong-but-successful build, so `make show` has to name it.
    described = fc.describe(list(catalog.values()))
    assert "[seed: seed-program-deepseek-v4flash]" in described
    assert "[seed: main]" not in described


def test_benchmarks_seeded_off_main_declare_their_branch():
    """Guards the live catalog against losing a `seed_branch` line.

    Written per-slug-if-present rather than as a hardcoded list: the catalog
    gains and loses entries, and this must not turn red for a benchmark that
    was simply retired.
    """
    fc = _fleet_config()
    catalog = fc.load_catalog()
    for slug in ("arcagi", "arcagi2", "paperreview"):
        if slug in catalog:
            assert catalog[slug].seed_branch == "seed-program-deepseek-v4flash", (
                f"{slug}'s seed program does not live on main; without "
                "seed_branch, `make bundles` builds the wrong tree or fails"
            )
    for b in catalog.values():
        assert b.seed_branch, f"{b.slug} has an empty seed_branch"


def test_bundle_branch_prefers_the_catalog_and_yields_to_an_explicit_flag():
    """--branch is a one-off sweep switch, not the source of truth.

    It applies to EVERY benchmark in the build, so it cannot be where a
    per-benchmark branch lives -- but it must still win when passed.
    """
    import build

    fc = _fleet_config()
    off_main = fc.Benchmark(
        slug="offmain", experiment_dir="offmain", repo_dirname="offmain-repo",
        est_hours=24, needs_retriever=False,
        seed_branch="seed-program-deepseek-v4flash",
    )
    plain = fc.Benchmark(
        slug="plain", experiment_dir="plain", repo_dirname="plain-repo",
        est_hours=24, needs_retriever=False,
    )

    assert build.seed_branch_for(off_main, None) == "seed-program-deepseek-v4flash"
    assert build.seed_branch_for(plain, None) == "main"
    assert build.seed_branch_for(off_main, "experimental") == "experimental"


def test_repo_path_honours_root_and_per_benchmark_overrides(monkeypatch):
    fc = _fleet_config()
    catalog = fc.load_catalog()
    hover = catalog["hover"]
    monkeypatch.setenv("CE_REPO_ROOT", "/srv/code")
    assert str(hover.repo_path()) == f"/srv/code/{hover.repo_dirname}"
    monkeypatch.setenv("CE_REPO_HOVER", "/mnt/fast/hover-clone")
    assert str(hover.repo_path()) == "/mnt/fast/hover-clone"
    # The override is per-SLUG, so two benchmarks sharing a checkout stay
    # independently redirectable — which is what makes it safe for hover and
    # ragqa to name the same repo_dirname.
    assert catalog["ragqa"].repo_path() != hover.repo_path()


def test_hover_and_ragqa_share_one_checkout_but_not_one_image():
    """The per-benchmark LangProBe forks are consolidated to a single repo.

    The fork existed so several programs could run locally at once; every run
    now gets its own container, so the split only bought three copies that
    drifted apart. What must NOT collapse is run isolation: the bundle and the
    image are keyed by slug, so the two benchmarks still seed and execute
    independently from the same source repo.
    """
    fc = _fleet_config()
    catalog = fc.load_catalog()
    hover, ragqa = catalog["hover"], catalog["ragqa"]

    assert hover.repo_dirname == ragqa.repo_dirname == "LangProBe-CodeEvolver"
    assert hover.bundle_name != ragqa.bundle_name
    assert hover.image("reg", "latest") != ragqa.image("reg", "latest")
    # And the dotted program/metric paths still resolve from that one root.
    assert hover.experiment_dir == "hover"
    assert ragqa.experiment_dir == "RAGQAArenaTech"


def test_slots_cap_concurrency_independently_of_benchmark_count(monkeypatch, tmp_path):
    """Up to ~8 benchmarks per researcher; a host still runs only slots_per_host."""
    C = _controller()
    fc = _fleet_config()
    catalog = _catalog_with_extras("factchecker", "hotpot")
    monkeypatch.setenv("CE_BENCHMARKS", "hover,phantomwiki,ragqa,factchecker,hotpot")
    monkeypatch.setattr(fc, "load_catalog", lambda *a, **k: catalog)
    monkeypatch.setattr(C, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(C, "live_containers", lambda host: {})
    monkeypatch.setattr(C, "collect", lambda *a, **k: None)
    dispatched: list[tuple[str, str]] = []
    monkeypatch.setattr(C, "dispatch", lambda m, r: dispatched.append((r.run_id, r.host)))

    matrix = C.load_matrix()
    args = types.SimpleNamespace(arm=None, bench=None, once=True, drain=False, dry_run=False)
    C.cmd_run(args)

    assert len(C.build_runs(matrix)) == 5 * 5 * 3, "5 benchmarks x 5 arms x 3 seeds"
    for host in matrix["hosts"]:
        n = sum(1 for _, h in dispatched if h == host)
        assert n == matrix["slots_per_host"], f"{host} got {n}, cap is 3"


# ---------------------------------------------------------------------------
# mid-run integrity check — runner trigger + orchestrator warning
# ---------------------------------------------------------------------------

def _entrypoint():
    import entrypoint as ep
    return ep


def test_integrity_trigger_fires_once_past_threshold(monkeypatch, tmp_path):
    ep = _entrypoint()
    monkeypatch.setattr(ep, "INTEGRITY_AT", "0.2")
    monkeypatch.setattr(ep, "INTEGRITY_LOG", tmp_path / "integrity_check.log")
    launched = []

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            launched.append((cmd, kwargs))

    monkeypatch.setattr(ep.subprocess, "Popen", FakePopen)

    # Below threshold: keep asking.
    assert ep.maybe_launch_integrity_check(
        "job_x", {"total_metric_calls": 100}, 1000
    ) is None
    assert launched == []

    # At threshold: launches the safe-mode CLI, non-blocking.
    proc = ep.maybe_launch_integrity_check(
        "job_x", {"total_metric_calls": 200}, 1000
    )
    assert isinstance(proc, FakePopen)
    (cmd, kwargs), = launched
    assert "inspect_run_integrity.py" in cmd[1]
    assert "--job-id" in cmd and "job_x" in cmd
    assert "--output" in cmd and str(ep.INTEGRITY_JSON) in cmd
    # Runs detached from the poll loop's stdout -- into a log under
    # /artifacts, not DEVNULL, so a missing verdict leaves a record of why.
    assert kwargs["stdout"] is not None
    assert kwargs["stdout"].name == str(tmp_path / "integrity_check.log")
    assert kwargs["stderr"] is ep.subprocess.STDOUT


def test_integrity_check_is_reaped_with_its_exit_code(monkeypatch, tmp_path, capsys):
    ep = _entrypoint()
    monkeypatch.setattr(ep, "INTEGRITY_JSON", tmp_path / "integrity.json")
    monkeypatch.setattr(ep, "INTEGRITY_LOG", tmp_path / "integrity_check.log")

    class FakeProc:
        returncode = None

        def poll(self):
            return self.returncode

    proc = FakeProc()
    assert ep.reap_integrity_check(proc) is False  # still running
    proc.returncode = 1
    assert ep.reap_integrity_check(proc) is True
    out = capsys.readouterr().out
    assert "inspector exited rc=1" in out and "verdict ABSENT" in out


def test_integrity_trigger_disabled_cases(monkeypatch):
    ep = _entrypoint()
    called = []
    monkeypatch.setattr(
        ep.subprocess, "Popen", lambda *a, **k: called.append(a)
    )

    # CE_INTEGRITY_AT=off disables.
    monkeypatch.setattr(ep, "INTEGRITY_AT", "off")
    assert ep.maybe_launch_integrity_check(
        "j", {"total_metric_calls": 999999}, 1000
    ) == "disabled"

    # Unknown budget disables (never guesses).
    monkeypatch.setattr(ep, "INTEGRITY_AT", "0.2")
    assert ep.maybe_launch_integrity_check(
        "j", {"total_metric_calls": 999999}, None
    ) == "disabled"

    # Garbage threshold disables rather than crashing the poll loop.
    monkeypatch.setattr(ep, "INTEGRITY_AT", "soon")
    assert ep.maybe_launch_integrity_check(
        "j", {"total_metric_calls": 999999}, 1000
    ) == "disabled"

    assert called == [], "no subprocess may be spawned in any disabled case"


def test_max_metric_calls_config_beats_env(monkeypatch):
    """The config now carries CE_SET, so it is the authority; env is a fallback.

    (Was `..._ce_set_beats_env` and took the raw --set list. It now takes the
    resolved config, because reading anything OTHER than the submitted config
    is what let the trigger disagree with the run.)
    """
    ep = _entrypoint()
    monkeypatch.setenv("CE_MAX_METRIC_CALLS", "1111")
    assert ep._max_metric_calls({"max_metric_calls": 5000}) == 5000
    assert ep._max_metric_calls({"other": 1}) == 1111
    monkeypatch.delenv("CE_MAX_METRIC_CALLS")
    assert ep._max_metric_calls({}) is None
    assert ep._max_metric_calls({"max_metric_calls": "abc"}) is None


def test_write_status_folds_integrity_verdict(monkeypatch, tmp_path):
    import json as _json

    ep = _entrypoint()
    monkeypatch.setattr(ep, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(ep, "INTEGRITY_JSON", tmp_path / "integrity.json")
    (tmp_path / "integrity.json").write_text(
        '{"cheating_category": "no_cheating", "integrity_pct": 95.0}'
    )
    ep.write_status("job_x", {"status": "completed"}, started=0.0, exit_reason="natural")
    blob = _json.loads((tmp_path / "status.json").read_text())
    assert blob["integrity"] == {
        "cheating_category": "no_cheating", "integrity_pct": 95.0,
    }

    # Absent/corrupt verdict → null, never a crash.
    (tmp_path / "integrity.json").write_text("{corrupt")
    ep.write_status("job_x", {"status": "completed"}, started=0.0, exit_reason="natural")
    assert _json.loads((tmp_path / "status.json").read_text())["integrity"] is None


def test_controller_warns_once_on_dirty_verdict(monkeypatch, tmp_path, capsys):
    import json as _json

    import controller as ctl

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(ctl, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(ctl, "LOG_PATH", tmp_path / "fleet.log")

    run = ctl.Run(run_id="ee-hover-s1", host="ce-a", bench="hover", arm="ee", seed=1)
    verdict_dir = tmp_path / "ce-runs" / run.run_id
    verdict_dir.mkdir(parents=True)
    (verdict_dir / "integrity.json").write_text(_json.dumps({
        "cheating_category": "impactful_cheating",
        "cheating_rationale": "hardcoded hover test answers in pipeline.py",
        "integrity_pct": 90.0,
    }))

    state = {"runs": {}}
    ctl.check_integrity(run, state)
    out = capsys.readouterr().out
    assert "INTEGRITY WARNING ee-hover-s1" in out
    assert "impactful_cheating" in out
    # Durable: the warning also landed in fleet.log, and state remembers it.
    assert "INTEGRITY WARNING" in (tmp_path / "fleet.log").read_text()
    assert state["runs"][run.run_id]["integrity_warned"] is True

    # Second sweep: the single mid-run verdict is never re-processed — no
    # duplicate warning, and the file isn't even re-read.
    (verdict_dir / "integrity.json").write_text("{corrupt on purpose")
    ctl.check_integrity(run, state)
    assert "INTEGRITY WARNING" not in capsys.readouterr().out


def test_controller_clean_verdict_records_without_warning(monkeypatch, tmp_path, capsys):
    import json as _json

    import controller as ctl

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(ctl, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(ctl, "LOG_PATH", tmp_path / "fleet.log")

    run = ctl.Run(run_id="ee-ragqa-s1", host="ce-a", bench="ragqa", arm="ee", seed=1)
    verdict_dir = tmp_path / "ce-runs" / run.run_id
    verdict_dir.mkdir(parents=True)
    (verdict_dir / "integrity.json").write_text(_json.dumps({
        "cheating_category": "no_cheating", "integrity_pct": 100.0,
    }))

    state = {"runs": {}}
    ctl.check_integrity(run, state)
    assert "INTEGRITY WARNING" not in capsys.readouterr().out
    assert state["runs"][run.run_id]["integrity"]["cheating_category"] == "no_cheating"
    assert not state["runs"][run.run_id].get("integrity_warned")

    # A clean verdict is also processed exactly once — later sweeps skip.
    (verdict_dir / "integrity.json").write_text("{corrupt on purpose")
    ctl.check_integrity(run, state)
    assert state["runs"][run.run_id]["integrity"]["cheating_category"] == "no_cheating"


def test_controller_low_pct_warns_even_without_cheating(monkeypatch, tmp_path, capsys):
    import json as _json

    import controller as ctl

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(ctl, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(ctl, "LOG_PATH", tmp_path / "fleet.log")

    run = ctl.Run(run_id="ee-pw-s1", host="ce-a", bench="phantomwiki", arm="ee", seed=1)
    verdict_dir = tmp_path / "ce-runs" / run.run_id
    verdict_dir.mkdir(parents=True)
    (verdict_dir / "integrity.json").write_text(_json.dumps({
        "cheating_category": "no_cheating", "integrity_pct": 40.0,
    }))

    state = {"runs": {}}
    ctl.check_integrity(run, state)
    out = capsys.readouterr().out
    assert "INTEGRITY WARNING" in out and "integrity_pct=40" in out


def test_ce_set_reaches_the_container_so_opt_in_flags_can_be_armed(monkeypatch):
    """`dispatch()` forwards ONLY `env_passthrough`, and CE_SET was missing.

    specs/engine3.md documents the clean-slate inspector as "default False
    everywhere -- fleet enables via CE_SET", but matrix.yaml listed only the
    four provider keys, so the variable was silently dropped and the inspector
    could not be armed on the fleet at all. Same for CE_MAX_METRIC_CALLS, which
    the mid-run integrity trigger reads to size its budget.
    """
    C = _controller()
    matrix = C.load_matrix()
    assert "CE_SET" in matrix["env_passthrough"]
    assert "CE_MAX_METRIC_CALLS" in matrix["env_passthrough"]

    captured: list[list[str]] = []
    monkeypatch.setattr(C, "docker", lambda host, *a, **k: captured.append(list(a)))
    monkeypatch.setattr(C, "ssh", lambda *a, **k: None)
    monkeypatch.setattr(C, "host_addr", lambda m, h: "root@x")
    monkeypatch.setenv("CE_SET", "enable_clean_slate_inspector=true")

    run = C.build_runs(matrix)[0]
    C.dispatch(matrix, run)
    assert "CE_SET=enable_clean_slate_inspector=true" in captured[0]


def test_optional_passthrough_keys_do_not_warn_when_unset(monkeypatch):
    """Unset is their NORMAL state — warning every dispatch is noise.

    A missing provider key really does fail the run and must stay loud; a
    missing CE_SET is the default. Lumping them together trains the operator
    to skim past both.
    """
    C = _controller()
    monkeypatch.setattr(C, "docker", lambda host, *a, **k: None)
    monkeypatch.setattr(C, "ssh", lambda *a, **k: None)
    monkeypatch.setattr(C, "host_addr", lambda m, h: "root@x")
    monkeypatch.delenv("CE_SET", raising=False)
    monkeypatch.delenv("CE_MAX_METRIC_CALLS", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    logged: list[str] = []
    monkeypatch.setattr(C, "log", lambda msg: logged.append(msg))

    matrix = C.load_matrix()
    C.dispatch(matrix, C.build_runs(matrix)[0])

    warnings = [m for m in logged if "WARNING" in m]
    assert not any("CE_SET" in m for m in warnings)
    assert not any("CE_MAX_METRIC_CALLS" in m for m in warnings)
    assert any("ANTHROPIC_API_KEY" in m for m in warnings), (
        "a missing provider key must still be loud"
    )


# ---------------------------------------------------------------------------
# research-run controls: budget source, research mode, settings banner
# ---------------------------------------------------------------------------

def test_budget_comes_from_the_submitted_config(monkeypatch):
    """The trigger's budget must equal the RUN's budget, always.

    A budget read only from CE_SET / CE_MAX_METRIC_CALLS would miss a run
    configured entirely from input.py and silently disable the mid-run
    integrity check. Reading the submitted config removes the possibility of
    the two disagreeing.
    """
    ep = _entrypoint()
    monkeypatch.delenv("CE_MAX_METRIC_CALLS", raising=False)

    assert ep._max_metric_calls({"max_metric_calls": 5000}) == 5000
    # CE_SET is already folded into the config by effective_config(), so an
    # override arrives here as the config value itself.
    assert ep._max_metric_calls({"max_metric_calls": 20}) == 20
    # Floats and numeric strings both land as ints.
    assert ep._max_metric_calls({"max_metric_calls": "300"}) == 300

    # No budget in the config: fall back to the env var, then give up.
    assert ep._max_metric_calls({}) is None
    monkeypatch.setenv("CE_MAX_METRIC_CALLS", "750")
    assert ep._max_metric_calls({}) == 750
    # A config budget always beats the fallback.
    assert ep._max_metric_calls({"max_metric_calls": 42}) == 42
    # Garbage degrades to the next source rather than raising mid-run.
    assert ep._max_metric_calls({"max_metric_calls": "not-a-number"}) == 750


def test_effective_config_matches_what_submit_sends(monkeypatch):
    """One resolution, in submit()'s own override order."""
    ep = _entrypoint()
    cfg = ep.effective_config(
        "hover", "OPTIMIZE_CONFIG_LOCAL", ["max_metric_calls=20"]
    )
    # repo_url is forced to the container workspace, exactly as submit() does.
    assert cfg["repo_url"] == str(ep.WORKSPACE)
    # extra_sets come AFTER repo_url, so they win.
    assert cfg["max_metric_calls"] == 20
    # ...and untouched keys survive from input.py.
    assert cfg["program"] == "langProBe.hover.hover_pipeline.HoverMultiHopPipeline"


def test_research_mode_arms_controls_but_ce_set_still_wins(monkeypatch):
    ep = _entrypoint()
    monkeypatch.setenv("CE_RESEARCH_MODE", "true")

    injected = ep.research_mode_defaults([])
    assert sorted(injected) == [
        "comparison_credits_initial=1",
        "comparison_credits_per_iteration=1",
        "enable_clean_slate_inspector=true",
        "enable_meta_reflection=true",
        "enable_run_integrity_agent=true",
        "seed_test_samples=1",
        "test_samples=3",
        "variance_estimate_samples=5",
    ]

    # An explicit CE_SET value is neither overwritten nor duplicated — turning a
    # control off for a deliberate experiment has to stay possible.
    injected = ep.research_mode_defaults(["enable_clean_slate_inspector=false"])
    assert not any("enable_clean_slate_inspector" in i for i in injected)
    assert len(injected) == 7

    for value in ("false", "0", ""):
        monkeypatch.setenv("CE_RESEARCH_MODE", value)
        assert ep.research_mode_defaults([]) == []


def test_settings_banner_never_prints_a_provider_key(monkeypatch, tmp_path):
    """Hard rule: the banner is rsynced to the controller.

    Making misconfiguration visible must not make credentials visible. The
    fingerprint still answers the question an operator has — did all five hosts
    get the same key?
    """
    ep = _entrypoint()
    monkeypatch.setattr(ep, "ARTIFACTS", tmp_path)
    secret = "sk-ant-DO-NOT-LOG-THIS-VALUE"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    monkeypatch.delenv("GMI_API_KEY", raising=False)
    monkeypatch.setenv("CE_RESEARCH_MODE", "true")

    printed: list[str] = []
    monkeypatch.setattr(ep, "log", lambda msg: printed.append(msg))

    settings = ep.log_settings(
        "hover", "hover", "ee", "OPTIMIZE_CONFIG_LOCAL",
        {"max_metric_calls": 20, "enable_clean_slate_inspector": True,
         "coding_harness_model": "deepinfra/zai-org/GLM-5.2"},
        ["max_metric_calls=20"],
    )

    banner = "\n".join(printed)
    on_disk = (tmp_path / "settings.json").read_text()
    for blob in (banner, on_disk):
        assert secret not in blob
        assert "present (sha256:" in blob
        assert "GMI_API_KEY" in blob and "MISSING" in blob

    # The controls block is the point of the exercise.
    assert settings["controls"]["clean_slate_inspector"] is True
    assert settings["controls"]["integrity_check_midrun"].endswith("of 20")
    assert settings["research_mode"] is True
    assert settings["models"]["coding_harness_model"] == "deepinfra/zai-org/GLM-5.2"


def test_settings_banner_names_a_disabled_midrun_check(monkeypatch, tmp_path):
    """A check that turned itself off must say so, not omit itself."""
    ep = _entrypoint()
    monkeypatch.setattr(ep, "ARTIFACTS", tmp_path)
    monkeypatch.delenv("CE_MAX_METRIC_CALLS", raising=False)
    settings = ep.build_settings("hover", "hover", "ee", "C", {}, [])
    assert settings["controls"]["integrity_check_midrun"] == "disabled (no budget)"


# ---------------------------------------------------------------------------
# workspace snapshot — the one thing git structurally cannot carry
# ---------------------------------------------------------------------------

def _git_workspace(tmp_path):
    """A workspace shaped like a finished run: two branches, ignored files."""
    import subprocess as sp

    ws = tmp_path / "workspace"
    (ws / "src").mkdir(parents=True)
    (ws / "src" / "program.py").write_text("x = 1\n")
    (ws / ".gitignore").write_text("env/\noutput/\n")
    for cmd in (
        ["init", "-q", "-b", "main"], ["add", "-A"],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "seed"],
        ["checkout", "-q", "-b", "codeevolver-final"],
    ):
        sp.run(["git", *cmd], cwd=ws, check=True, capture_output=True)
    (ws / "src" / "program.py").write_text("x = 2\n")
    for cmd in (
        ["add", "-A"],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "evolved"],
        ["checkout", "-q", "main"],   # leave HEAD on the WRONG branch
    ):
        sp.run(["git", *cmd], cwd=ws, check=True, capture_output=True)

    # The whole point: an artifact the run built that git will never carry.
    (ws / "output").mkdir()
    (ws / "output" / "corpus_index.pkl").write_bytes(b"expensive-to-rebuild")
    # ...and the two things that must NOT be carried.
    (ws / "env" / "lib").mkdir(parents=True)
    (ws / "env" / "lib" / "torch.so").write_bytes(b"0" * 4096)
    (ws / "src" / "__pycache__").mkdir()
    (ws / "src" / "__pycache__" / "program.pyc").write_bytes(b"junk")
    return ws


def test_workspace_snapshot_carries_what_git_cannot(monkeypatch, tmp_path):
    """Commits use `git add -A` with no --force, so ignored files never land in
    repo.bundle and die with the container. A program that builds and then
    depends on one could not reproduce its own scores."""
    import tarfile as tf

    ep = _entrypoint()
    ws = _git_workspace(tmp_path)
    monkeypatch.setattr(ep, "WORKSPACE", ws)
    monkeypatch.setattr(ep, "ARTIFACTS", tmp_path / "artifacts")
    (tmp_path / "artifacts").mkdir()
    monkeypatch.setattr(ep, "log", lambda msg: None)

    ep.export_workspace({"final_candidate": {"git_branch": "codeevolver-final"}})

    manifest = json.loads((tmp_path / "artifacts" / "workspace_manifest.json").read_text())
    # HEAD was deliberately left on main; the snapshot must correct it.
    assert manifest["intended_branch"] == "codeevolver-final"
    assert manifest["checkout_ok"] is True
    assert manifest["actual_branch"] == "codeevolver-final"
    assert "output/" in " ".join(manifest["untracked_paths"])

    with tf.open(tmp_path / "artifacts" / "workspace.tar.gz") as tar:
        names = tar.getnames()
    assert "./output/corpus_index.pkl" in names, "the ignored artifact is the point"
    assert "./src/program.py" in names
    # Excluded: the staged venv, git objects (repo.bundle has them, deduped in a
    # shared bare repo), and pure caches.
    assert not any("/env/" in n or n.endswith("/env") for n in names)
    assert not any(n.startswith("./.git/") for n in names)
    assert not any("__pycache__" in n for n in names)


def test_workspace_snapshot_never_fails_a_finished_run(monkeypatch, tmp_path):
    """It is a backup. A run that produced results must not die exporting it."""
    ep = _entrypoint()
    ws = _git_workspace(tmp_path)
    monkeypatch.setattr(ep, "WORKSPACE", ws)
    monkeypatch.setattr(ep, "ARTIFACTS", tmp_path / "artifacts")
    (tmp_path / "artifacts").mkdir()
    monkeypatch.setattr(ep, "log", lambda msg: None)

    # A branch that does not exist: archive HEAD as-is and say so.
    ep.export_workspace({"final_candidate": {"git_branch": "no-such-branch"}})
    manifest = json.loads((tmp_path / "artifacts" / "workspace_manifest.json").read_text())
    assert manifest["checkout_ok"] is False
    assert manifest["actual_branch"] == "main"
    assert (tmp_path / "artifacts" / "workspace.tar.gz").exists()

    # A missing workspace entirely: recorded, not raised.
    monkeypatch.setattr(ep, "WORKSPACE", tmp_path / "gone")
    ep.export_workspace({})
    manifest = json.loads((tmp_path / "artifacts" / "workspace_manifest.json").read_text())
    assert "error" in manifest


def test_traces_are_neither_transferred_nor_merged():
    """Largest artifact class, and nothing downstream reads it.

    Both halves matter: dropping it from JOB_TREES alone would still pay to
    rsync it off the host.
    """
    sys.path.insert(0, str(REPO_ROOT / "orchestrator"))
    import collect

    assert "traces" not in collect.JOB_TREES
    assert ".codeevolver/traces" in collect.RSYNC_EXCLUDES
    # The per-session record of what the architect DID stays.
    assert "architect_traces" in collect.JOB_TREES
    assert "logs" in collect.JOB_TREES, "the clean-slate report lives here now"


def test_teardown_waits_for_the_optimizers_post_run_agents(monkeypatch, tmp_path):
    """A terminal job status is not the end of the optimizer's work.

    `optimizer_loop` calls `set_completed()` BEFORE meta-reflection and the
    run-integrity inspector, deliberately, so an advisory agent can never fail
    a run that already succeeded. `poll()` returns on that status, so without
    this wait the container would tear down inside one poll interval and kill
    both agents, leaving no meta_reflection.md and `None` in every
    `integrity_*` field.
    """
    ep = _entrypoint()
    monkeypatch.setattr(ep, "LOCAL_LOG_ROOT", tmp_path)
    monkeypatch.setattr(ep, "POST_RUN_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(ep, "log", lambda msg: None)

    alive = {"count": 3}

    def fake_alive(pid):
        alive["count"] -= 1
        return alive["count"] > 0

    monkeypatch.setattr(ep, "_optimizer_alive", fake_alive)
    (tmp_path / "job_x.pid").write_text("4242\n")

    ep.await_optimizer_exit("job_x", timeout=30.0)
    assert alive["count"] == 0, "must keep waiting until the optimizer is gone"


def test_teardown_wait_is_bounded_and_survives_a_missing_pidfile(monkeypatch, tmp_path):
    """An advisory agent must never own the container.

    A hung post-run agent costs a host slot on the fleet, so the wait gives up
    and exports what exists. A missing pidfile (any non-LocalLauncher path)
    must degrade to the old behaviour rather than raise.
    """
    ep = _entrypoint()
    monkeypatch.setattr(ep, "LOCAL_LOG_ROOT", tmp_path)
    monkeypatch.setattr(ep, "POST_RUN_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(ep, "log", lambda msg: None)
    monkeypatch.setattr(ep, "_optimizer_alive", lambda pid: True)

    (tmp_path / "job_hung.pid").write_text("4242\n")
    ep.await_optimizer_exit("job_hung", timeout=0.05)  # returns, does not hang

    ep.await_optimizer_exit("job_absent", timeout=30.0)  # no pidfile, no raise


def test_optimizer_liveness_is_not_fooled_by_pid_reuse():
    """Identity comes from the cmdline, not from the pid merely existing.

    A one-run-per-container image recycles pids freely, so a signal-0 probe
    would report whatever inherited the optimizer's pid as "still working" and
    hold teardown for the whole timeout. This process is pytest, and it is
    unambiguously alive — so a True here would mean the check is a liveness
    probe wearing an identity check's name.
    """
    ep = _entrypoint()
    assert ep._optimizer_alive(os.getpid()) is False


@pytest.mark.skipif(not Path("/proc/self/cmdline").exists(), reason="needs /proc")
def test_optimizer_liveness_recognises_a_real_entry_shim(tmp_path):
    """The positive half, on the platform the runner actually runs on.

    Every method's shim is `.../optimization_entry.py`
    (`local_launcher._ENTRY_SCRIPT_REL`), so one substring covers every method.
    """
    ep = _entrypoint()
    shim = tmp_path / "optimization_entry.py"
    shim.write_text("import time; time.sleep(30)\n")
    proc = subprocess.Popen([sys.executable, str(shim), "--config", "/tmp/c.json"])
    try:
        assert ep._optimizer_alive(proc.pid) is True
    finally:
        proc.kill()
        proc.wait()


# --- stall warning (runner heartbeat + controller check) ---------------------


def test_runner_writes_progress_heartbeat(monkeypatch, tmp_path):
    import json as _json

    ep = _entrypoint()
    monkeypatch.setattr(ep, "PROGRESS_JSON", tmp_path / "progress.json")
    ep.write_progress({"status": "running", "current_iteration": 3, "total_metric_calls": 150})
    blob = _json.loads((tmp_path / "progress.json").read_text())
    assert blob["current_iteration"] == 3 and blob["total_metric_calls"] == 150
    assert blob["status"] == "running" and blob["ts"] > 0


def test_controller_stall_warning_fires_once_after_four_hours(monkeypatch, tmp_path, capsys):
    import json as _json

    import controller as ctl

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(ctl, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(ctl, "LOG_PATH", tmp_path / "fleet.log")
    run = ctl.Run(run_id="simple_loop-hover-s1", host="ce-a", bench="hover", arm="simple_loop", seed=1)
    run_dir = tmp_path / "ce-runs" / run.run_id
    run_dir.mkdir(parents=True)
    heartbeat = run_dir / "progress.json"
    heartbeat.write_text(_json.dumps({"current_iteration": 2, "total_metric_calls": 90}))
    state = {"runs": {}}

    ctl.check_stalled(run, state, now=1000.0)
    ctl.check_stalled(run, state, now=1000.0 + 3 * 3600)
    assert "STALLED" not in capsys.readouterr().out

    ctl.check_stalled(run, state, now=1000.0 + 4 * 3600)
    out = capsys.readouterr().out
    assert f"STALLED WARNING {run.run_id}" in out and "4.0h" in out
    assert "STALLED WARNING" in (tmp_path / "fleet.log").read_text()

    ctl.check_stalled(run, state, now=1000.0 + 5 * 3600)
    assert "STALLED" not in capsys.readouterr().out

    heartbeat.write_text(_json.dumps({"current_iteration": 3, "total_metric_calls": 140}))
    ctl.check_stalled(run, state, now=1000.0 + 6 * 3600)
    assert state["runs"][run.run_id]["progress"]["warned"] is False
    ctl.check_stalled(run, state, now=1000.0 + 11 * 3600)
    assert "STALLED WARNING" in capsys.readouterr().out
def test_bundle_seeds_from_a_branch_that_exists_only_as_a_remote_ref(tmp_path, monkeypatch):
    """A fresh `git clone` is the state every NEW operator starts from.

    Clone creates a local branch for the remote HEAD and nothing else, so the
    seed branch exists solely as `origin/<branch>` and a bare `git rev-parse`
    fails. Before the fallback, `build.py bundles` died on a fresh checkout
    telling the operator to fix `seed_branch` for a branch that was really
    there -- the one wrong move, since editing the catalog breaks the machines
    where it already works.
    """
    import build

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git = ["git", "-C", str(upstream)]
    build.sh(["git", "init", "-q", "-b", "main", str(upstream)])
    build.sh(git + ["config", "user.email", "t@t"])
    build.sh(git + ["config", "user.name", "t"])
    (upstream / "seed.txt").write_text("on the seed branch\n")
    build.sh(git + ["add", "-A"])
    build.sh(git + ["commit", "-q", "-m", "main commit"])
    build.sh(git + ["checkout", "-q", "-b", "seed-program-deepseek-v4flash"])
    (upstream / "marker.txt").write_text("seed-only file\n")
    build.sh(git + ["add", "-A"])
    build.sh(git + ["commit", "-q", "-m", "seed commit"])
    build.sh(git + ["checkout", "-q", "main"])

    clone = tmp_path / "ARC-AGI"
    build.sh(["git", "clone", "-q", str(upstream), str(clone)])
    # Precondition: the seed branch is NOT a local branch after cloning.
    local = build.out(["git", "-C", str(clone), "branch", "--format=%(refname:short)"])
    assert "seed-program-deepseek-v4flash" not in local.split()

    bench = _fleet_config().Benchmark(
        slug="freshclone", experiment_dir="freshclone", repo_dirname="ARC-AGI",
        est_hours=24, needs_retriever=False,
        seed_branch="seed-program-deepseek-v4flash",
    )
    monkeypatch.setenv("CE_REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(build, "BUNDLE_DIR", tmp_path / "bundles")

    dest = build.build_bundle(bench, "seed-program-deepseek-v4flash")
    assert dest.is_file()

    # The bundle must carry the SEED branch's tree, not the clone's local main.
    check = tmp_path / "check"
    build.sh(["git", "clone", "-q", "-b", "seed", str(dest), str(check)])
    assert (check / "marker.txt").is_file(), "bundled the wrong branch's tree"


def test_bundle_reports_both_spellings_when_a_branch_is_truly_absent(tmp_path, monkeypatch):
    """The fallback must not turn a real typo into a confusing failure."""
    import build

    repo = tmp_path / "ARC-AGI"
    repo.mkdir()
    build.sh(["git", "init", "-q", "-b", "main", str(repo)])
    build.sh(["git", "-C", str(repo), "config", "user.email", "t@t"])
    build.sh(["git", "-C", str(repo), "config", "user.name", "t"])
    (repo / "f.txt").write_text("x\n")
    build.sh(["git", "-C", str(repo), "add", "-A"])
    build.sh(["git", "-C", str(repo), "commit", "-q", "-m", "c"])

    bench = _fleet_config().Benchmark(
        slug="typo", experiment_dir="typo", repo_dirname="ARC-AGI",
        est_hours=24, needs_retriever=False, seed_branch="nope",
    )
    monkeypatch.setenv("CE_REPO_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as excinfo:
        build.build_bundle(bench, "nope")
    msg = str(excinfo.value)
    assert "'nope'" in msg and "origin/nope" in msg


def test_arcagi2_is_registered_and_shaped_like_the_other_arc_arm():
    """arcagi2 must be dispatchable and comparable to arcagi.

    The two ARC arms are meant to differ in the BENCHMARK and nothing else, so
    a drift in budget or experiment_dir silently turns a paired comparison into
    two unrelated runs. `experiment_dir` is also the one field that is not the
    slug (`arcagi2` -> `experiments/arc-agi-2`), which is exactly the mistake
    the fleet docs call out.
    """
    catalog = _fleet_config().load_catalog()
    if "arcagi2" not in catalog:  # retired: nothing to guard
        return
    a2 = catalog["arcagi2"]
    assert a2.experiment_dir == "arc-agi-2"
    assert a2.repo_dirname == "ARC-AGI-2"
    assert not a2.needs_retriever
    assert a2.assets == ()

    if "arcagi" in catalog:
        # est_hours is deliberately allowed to differ (v2 rows are slower);
        # the shape around it is not.
        assert a2.seed_branch == catalog["arcagi"].seed_branch


def test_every_catalog_benchmark_has_an_input_py():
    """A slug with no experiments/<dir>/input.py cannot be dispatched at all.

    `experiments.run --program <experiment_dir>` is what the runner invokes, so
    a catalog entry pointing at a missing directory fails only once a container
    is already up and the image has been built and pushed — long after the
    cheap moment to catch it.
    """
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    catalog = _fleet_config().load_catalog()
    missing = [
        f"{slug} -> experiments/{b.experiment_dir}/input.py"
        for slug, b in catalog.items()
        if not (repo_root / "experiments" / b.experiment_dir / "input.py").is_file()
    ]
    assert not missing, f"catalog entries with no input.py: {missing}"


# --- DEEPSEEK_API_KEY / api.deepseek.com are run-wide, not per-benchmark -------


def test_deepseek_key_warns_on_every_benchmark_and_host_is_required_globally(monkeypatch):
    """Every wired benchmark (appworld and alfworld included) runs
    DeepSeek-V4-Flash through the gmi -> deepseek -> deepinfra chain, so the
    key must not be scoped in BENCH_CREDENTIALS and the host must be in the
    global ALLOWED_HOSTS rather than BENCH_ALLOWED_HOSTS."""
    import controller
    import preflight

    assert "DEEPSEEK_API_KEY" not in controller.BENCH_CREDENTIALS
    assert "DEEPSEEK_API_KEY" not in controller.OPTIONAL_PASSTHROUGH
    assert "api.deepseek.com" in preflight.ALLOWED_HOSTS
    for hosts in preflight.BENCH_ALLOWED_HOSTS.values():
        assert "api.deepseek.com" not in hosts
    for bench in ("appworld", "alfworld", "hover", ""):
        monkeypatch.setenv("CE_BENCH", bench)
        assert "api.deepseek.com" in preflight.allowed_hosts()


# ---------------------------------------------------------------------------
# Seed-sharded matrices + run_suffix
# ---------------------------------------------------------------------------

def _seed_sharded_matrix(tmp_path, extra: str = "") -> Path:
    path = tmp_path / "matrix.yaml"
    path.write_text(
        "hosts: [ce-x-a, ce-x-b, ce-x-c]\n"
        "seeds: [1, 2, 3]\n"
        "assignment:\n  ce-x-a: pareto_evolution\n  ce-x-b: pareto_evolution\n  ce-x-c: pareto_evolution\n"
        "host_seeds:\n  ce-x-a: [1]\n  ce-x-b: [2]\n  ce-x-c: [3]\n"
        + extra
    )
    return path


def test_host_seeds_puts_one_seed_per_host_and_run_suffix_lands_on_every_id(tmp_path):
    """cmd_run never runs a benchmark twice on a host (busy_benches), so three
    seeds of one arm need three hosts; host_seeds expresses that. run_suffix
    keeps a re-run's ids distinct from the kept originals."""
    C = _controller()
    matrix = C.load_matrix(_seed_sharded_matrix(tmp_path, 'run_suffix: "-redo"\n'))
    runs = C.build_runs(matrix)
    benches = {b.slug for b in _fleet_config().selected_benchmarks()}

    assert len(runs) == 3 * len(benches)
    for host, seed in (("ce-x-a", 1), ("ce-x-b", 2), ("ce-x-c", 3)):
        sel = [r for r in runs if r.host == host]
        assert {r.seed for r in sel} == {seed}
        assert {r.bench for r in sel} == benches
        assert all(r.run_id == f"pareto_evolution-{r.bench}-s{seed}-redo" for r in sel)
        assert all(r.container == f"ce-{r.run_id}" for r in sel)
    assert all(isinstance(r.seed, int) for r in runs), "CE_SEED must stay numeric"


def test_matrices_without_the_new_keys_expand_exactly_as_before(tmp_path):
    """fleet.py imports build_runs and the live launchers' `fleet down` step
    loads whatever controller.py is on disk, so the default path must be
    byte-identical to the expansion without these keys."""
    C = _controller()
    path = tmp_path / "matrix.yaml"
    path.write_text("hosts: [ce-a, ce-b]\nseeds: [1, 2]\nassignment:\n  ce-a: ee\n  ce-b: asa\n")
    runs = C.build_runs(C.load_matrix(path))
    benches = [b.slug for b in _fleet_config().selected_benchmarks()]
    assert [r.run_id for r in runs] == [
        f"{arm}-{b}-s{seed}"
        for seed in (1, 2) for arm in ("ee", "asa") for b in benches
    ]


def test_load_matrix_rejects_stray_host_seeds_hosts_and_unknown_seeds(tmp_path):
    C = _controller()
    path = tmp_path / "matrix.yaml"
    path.write_text(
        "hosts: [ce-a]\nseeds: [1]\nassignment:\n  ce-a: ee\nhost_seeds:\n  ce-z: [1]\n"
    )
    with pytest.raises(ValueError, match="host_seeds names unknown hosts"):
        C.load_matrix(path)
    path.write_text(
        "hosts: [ce-a]\nseeds: [1]\nassignment:\n  ce-a: ee\nhost_seeds:\n  ce-a: [7]\n"
    )
    with pytest.raises(ValueError, match="lists seeds not in"):
        C.load_matrix(path)


def test_load_matrix_rejects_a_run_suffix_results_tables_cannot_parse(tmp_path):
    C = _controller()
    path = tmp_path / "matrix.yaml"
    path.write_text('hosts: [ce-a]\nseeds: [1]\nassignment:\n  ce-a: ee\nrun_suffix: "redo"\n')
    with pytest.raises(ValueError, match="run_suffix must match"):
        C.load_matrix(path)
    path.write_text('hosts: [ce-a]\nseeds: [1]\nassignment:\n  ce-a: ee\nrun_suffix: "-Redo_2"\n')
    with pytest.raises(ValueError, match="run_suffix must match"):
        C.load_matrix(path)


def test_build_passes_no_retriever_through_to_build_py(monkeypatch):
    """`ce build --no-retriever` must reach build.py, which otherwise rebuilds
    and pushes the ragqa-retriever sidecar whenever ragqa is in the subset."""
    C = _controller()
    calls: list[list[str]] = []
    monkeypatch.setattr(C, "running_runs", lambda host: [])
    monkeypatch.setattr(
        C.subprocess, "run",
        lambda cmd, **kw: calls.append(list(cmd)) or types.SimpleNamespace(returncode=0),
    )
    args = types.SimpleNamespace(
        registry=None, tag=None, host="ce-build", platform="linux/amd64",
        no_pull=True, force=False, no_retriever=True,
    )
    assert C.cmd_build(args) == 0
    assert calls and all("--no-retriever" in c for c in calls)
    assert all("--host" in c and "ce-build" in c for c in calls)
