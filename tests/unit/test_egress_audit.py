"""Copyright © 2026 440 Labs LLC

Deterministic coverage for orchestrator/egress_audit.py — the machine check
that every task-LM connection a run made happened inside a `run_eval` window
(or the seed / finalization phases the root driver owns).

Fixtures are synthetic: a job log in the optimizer's `asctime | LEVEL | logger
| msg` shape and a squid access log in the gateway's `ce` logformat, in both
its seven-field form and the nine-field form (`%tr %>lp` appended).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "orchestrator"))

import egress_audit as ea  # noqa: E402

T0 = 1_788_400_000.0  # 2026-09-03T01:46:40Z


def _log_line(ts: float, msg: str, logger: str = "src.engine.optimizer.v3.run_eval_tool") -> str:
    import datetime as dt

    stamp = dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return f"{stamp},000 | INFO | {logger} | {msg}"


def _job_log(tmp_path: Path) -> Path:
    lines = [
        _log_line(T0, "run config: method=ee", "src.engine.optimizer.v3.optimizer_loop"),
        _log_line(T0 + 10, "seed-validate: running environment check on first 5 training row(s)",
                  "src.engine.sandbox.verify_environment"),
        _log_line(T0 + 70, "seed-validate: primary=0.5988 system_errors=0/5 (0.0%)",
                  "src.engine.sandbox.verify_environment"),
        _log_line(T0 + 600, "iteration 1: run_eval start (sample=train, suffix=parent, rows=50, timeout=7200)"),
        _log_line(T0 + 900, "iteration 1: run_eval done (id=abc, sample=train, suffix=parent, primary_score=0.4, sha=1, tree=2 first eval)"),
        # A timed-out eval: start with no done, timeout carried on the line.
        _log_line(T0 + 2000, "iteration 2: run_eval start (sample=train, suffix=child, rows=50, timeout=3000)"),
        _log_line(T0 + 9000, "iteration 3: run_eval start (sample=comparison, suffix=comparison, rows=200, timeout=14400)"),
        _log_line(T0 + 9800, "iteration 3: run_eval done (id=def, sample=comparison, suffix=comparison, primary_score=0.8, sha=3, tree=4)"),
        _log_line(T0 + 9900, "optimizer loop: budget exhausted at iteration 4", "src.engine.optimizer.v3.optimizer_loop"),
        _log_line(T0 + 10500, "finalization: scoring seed (sha=1, test rows=100 x 1 samples, comparison rows=200)",
                  "src.engine.optimizer.v3.finalization"),
        _log_line(T0 + 13000, "finalization: test sample 1/1 (iteration 0) -> 0.5", "src.engine.evaluator.holdout"),
        # Post-finalization agents log too; they must not shorten the window.
        _log_line(T0 + 14500, "meta-reflection: starting", "src.engine.optimizer.meta.meta_reflection"),
    ]
    p = tmp_path / "job_x.log"
    p.write_text("\n".join(lines) + "\n")
    return p


def _access(ts: float, ip: str, sni: str, status: str = "TCP_TUNNEL/200", extra: str = "") -> str:
    return f"{ts:.3f} {ip} {status} 512 CONNECT 1.2.3.4:443 {sni}{extra}"


def test_parse_eval_windows_pairs_starts_closes_timeouts_and_seed(tmp_path):
    with open(_job_log(tmp_path)) as fh:
        facts = ea.parse_eval_windows(fh)
    kinds = [(w.kind, w.start - T0, w.end - T0) for w in facts.windows]
    assert ("seed", 10.0, 70.0) in kinds
    assert ("eval", 600.0, 900.0) in kinds
    assert ("eval-timeout", 2000.0, 5000.0) in kinds  # start + timeout=3000
    assert ("eval", 9000.0, 9800.0) in kinds
    assert facts.first_ts == T0 and facts.last_ts == T0 + 14500
    assert facts.finalization_start == T0 + 10500


def test_simple_loop_finalization_starts_at_the_last_session_summary():
    lines = [
        _log_line(T0, "iteration 3: run_eval start (sample=train, suffix=parent, rows=50, timeout=3000)"),
        _log_line(T0 + 300, "iteration 3: run_eval done (id=x, sample=train, suffix=parent, primary_score=0.8, sha=1, tree=2)"),
        _log_line(T0 + 400, "simple loop session 3: evals 4630/4500 rows (this session: 150), score=0.8",
                  "src.engine.optimizer.simple_loop.loop"),
    ]
    # Loop still live (no holdout line yet): no finalization marker.
    assert ea.parse_eval_windows(lines).finalization_start is None
    lines.append(_log_line(T0 + 900, "finalization: test sample 1/1 (iteration 0) -> 0.55",
                           "src.engine.evaluator.holdout"))
    assert ea.parse_eval_windows(lines).finalization_start == T0 + 400


def test_parse_access_line_handles_seven_and_nine_fields():
    seven = ea.parse_access_line(_access(T0, "172.31.66.4", "api.gmi-serving.com"))
    assert seven is not None and seven.elapsed_ms is None and seven.local_port is None
    nine = ea.parse_access_line(_access(T0, "172.31.66.4", "api.gmi-serving.com", extra=" 1500 3130"))
    assert nine is not None and nine.elapsed_ms == 1500 and nine.local_port == 3130
    assert ea.parse_access_line("garbage") is None


def test_audit_counts_only_task_traffic_outside_windows(tmp_path):
    job_log = _job_log(tmp_path)
    ip, other = "172.31.66.4", "172.31.66.5"
    lines = [
        # Startup canary: providers probed from root before the run spends.
        _access(T0 + 2, ip, "api.deepseek.com", "NONE_NONE/000"),
        _access(T0 + 2, ip, "api.deepseek.com"),
        # Inside eval 1.
        _access(T0 + 700, ip, "api.gmi-serving.com", "NONE_NONE/000"),
        _access(T0 + 700, ip, "api.gmi-serving.com"),
        # Closed 90 s after eval 1 ended: within grace (keep-alive).
        _access(T0 + 990, ip, "api.gmi-serving.com"),
        # Between eval 1 and the timed-out eval 2: VIOLATION.
        _access(T0 + 1500, ip, "api.gmi-serving.com", "NONE_NONE/000"),
        _access(T0 + 1500, ip, "api.gmi-serving.com"),
        # Inside the timed-out eval's window.
        _access(T0 + 4000, ip, "api.gmi-serving.com"),
        # Harness provider between evals: never a violation.
        _access(T0 + 6000, ip, "api.deepinfra.com"),
        # Another container's out-of-window call: not ours.
        _access(T0 + 6000, other, "api.gmi-serving.com"),
        # Between the last eval and the finalization marker: VIOLATION.
        _access(T0 + 9950, ip, "api.gmi-serving.com"),
        # Nine-field line closing late but opened inside eval 3 (elapsed 400 s).
        _access(T0 + 10100, ip, "api.gmi-serving.com", extra=" 400000 3130"),
        # After the loop (finalization holdout evals, then auditors): allowed.
        _access(T0 + 12000, ip, "api.gmi-serving.com"),
        _access(T0 + 14800, ip, "api.gmi-serving.com"),
        # After the run ended: ignored.
        _access(T0 + 20000, ip, "api.gmi-serving.com"),
    ]
    gw = tmp_path / "access.log"
    gw.write_text("\n".join(lines) + "\n")

    result = ea.audit(job_log, gw, ip, T0, T0 + 15000)
    assert result["verdict"] == "violations"
    assert result["violation_lines"] == 3
    assert result["violation_connections_est"] == 2
    assert result["canary_lines"] == 2
    assert [v["at"] for v in result["violations"]] == [
        "2026-09-03T02:11:40Z", "2026-09-03T02:11:40Z", "2026-09-03T04:32:30Z",
    ]
    assert result["violation_clusters"][0]["hosts"] == ["api.gmi-serving.com"]
    assert len(result["violation_clusters"]) == 2
    assert result["timed_out_evals"] == 1
    assert result["loop_end"] == T0 + 10500
    # 2 canary + 2 eval1 + 1 grace + 2 violation + 1 timeout-window + 1 pre-final
    # violation + 1 backdated + 2 post-loop
    assert result["task_provider_lines"] == 12


def test_live_run_has_no_post_loop_window(tmp_path):
    """Mid-run (no finalization marker yet) nothing after the last eval is
    excused: an in-flight eval is a timeout window, everything else counts."""
    lines = [
        _log_line(T0, "iteration 1: run_eval start (sample=train, suffix=parent, rows=50, timeout=3000)"),
        _log_line(T0 + 300, "iteration 1: run_eval done (id=x, sample=train, suffix=parent, primary_score=0.8, sha=1, tree=2)"),
    ]
    job_log = tmp_path / "job_live.log"
    job_log.write_text("\n".join(lines) + "\n")
    gw = tmp_path / "access.log"
    gw.write_text(_access(T0 + 2000, "172.31.66.4", "api.gmi-serving.com") + "\n")
    result = ea.audit(job_log, gw, "172.31.66.4", T0, T0 + 3000)
    assert result["loop_end"] is None and result["violation_lines"] == 1


def test_audit_is_clean_when_all_traffic_is_inside_windows(tmp_path):
    job_log = _job_log(tmp_path)
    ip = "172.31.66.4"
    gw = tmp_path / "access.log"
    gw.write_text("\n".join([
        _access(T0 + 650, ip, "api.gmi-serving.com"),
        _access(T0 + 9500, ip, "api.deepseek.com"),
        _access(T0 + 11000, ip, "api.gmi-serving.com"),  # finalization
        _access(T0 + 14900, ip, "api.gmi-serving.com"),  # still finalization
    ]) + "\n")
    result = ea.audit(job_log, gw, ip, T0, T0 + 15000)
    assert result["verdict"] == "clean" and result["violation_lines"] == 0
    assert "clean" in ea.summarize(result)


def test_infer_container_ip_breaks_simultaneous_starts_by_end_and_signature(tmp_path):
    gw = tmp_path / "access.log"
    lines = [
        # An earlier run used .4 an hour ago: it is not "fresh" at T0.
        _access(T0 - 3000, "172.31.66.4", "api.gmi-serving.com"),
        # Three containers start within seconds of T0.
        _access(T0 + 5, "172.31.66.4", "api.gmi-serving.com"),
        _access(T0 + 6, "172.31.66.5", "api.gmi-serving.com"),
        _access(T0 + 7, "172.31.66.6", "api.gmi-serving.com"),
        _access(T0 + 100, "172.31.66.5", "julianghadially--colbert-server-wiki-colbertservice-serve.modal.run"),
        _access(T0 + 100, "172.31.66.6", "julianghadially--colbert-server-phantom-wiki-colbertserv.modal.run"),
        _access(T0 + 50000, "172.31.66.5", "api.gmi-serving.com"),
        _access(T0 + 90000, "172.31.66.6", "api.gmi-serving.com"),
    ]
    gw.write_text("\n".join(lines) + "\n")
    # .4 was active before the start window: excluded outright.
    assert ea.infer_container_ip(gw, T0) is None  # .5 and .6 both qualify
    assert ea.infer_container_ip(gw, T0, ended_at=T0 + 50000) == "172.31.66.5"
    assert ea.infer_container_ip(gw, T0, bench="phantomwiki") == "172.31.66.6"
    assert ea.infer_container_ip(gw, T0, bench="hover") == "172.31.66.5"


def test_resolve_job_log_prefers_the_log_with_the_most_evals(tmp_path):
    run_dir = tmp_path / "run"
    logs = run_dir / ".codeevolver" / "logs"
    logs.mkdir(parents=True)
    (logs / "job_rogue.log").write_text(_log_line(T0, "iteration 1: run_eval start (sample=train, suffix=parent, rows=50)") + "\n")
    real = logs / "job_real.log"
    real.write_text("\n".join(_log_line(T0 + i, f"iteration {i}: run_eval start (sample=train, suffix=parent, rows=50)") for i in range(3)) + "\n")
    (logs / "job_real_detailed_agent_log.log").write_text("x" * 10)
    assert ea.resolve_job_log(run_dir) == real
    assert ea.resolve_job_log(tmp_path / "missing") is None


def test_task_provider_matching_is_suffix_based():
    assert ea.is_task_provider("api.gmi-serving.com")
    assert ea.is_task_provider("api.deepseek.com")
    assert ea.is_task_provider("openrouter.ai")
    assert not ea.is_task_provider("api.deepinfra.com")
    assert not ea.is_task_provider("api.openai.com")
    assert not ea.is_task_provider("-")


def test_cli_exit_codes(tmp_path, monkeypatch):
    monkeypatch.setattr(ea, "RUNS_HOME", tmp_path)
    run_dir = tmp_path / "r1" / ".codeevolver" / "logs"
    run_dir.mkdir(parents=True)
    (run_dir / "job_1.log").write_text(_job_log(tmp_path).read_text())
    gw = tmp_path / "gw.log"
    gw.write_text(_access(T0 + 5, "172.31.66.9", "api.gmi-serving.com") + "\n"
                  + _access(T0 + 1500, "172.31.66.9", "api.gmi-serving.com") + "\n")
    assert ea.main(["--run-id", "r1", "--gateway-log", str(gw), "--end", str(T0 + 15000)]) == 1
    gw.write_text(_access(T0 + 5, "172.31.66.9", "api.gmi-serving.com") + "\n")
    assert ea.main(["--run-id", "r1", "--gateway-log", str(gw), "--end", str(T0 + 15000)]) == 0
    assert ea.main(["--run-id", "nope", "--gateway-log", str(gw)]) == 2
    assert ea.main(["--run-id", "r1", "--gateway-log", str(tmp_path / "absent.log")]) == 2
