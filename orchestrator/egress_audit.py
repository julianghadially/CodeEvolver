#!/usr/bin/env python3
"""Copyright © 2026 440 Labs LLC

Machine audit: did a run's task-LM traffic all happen inside `run_eval`?

Policy: every task-LM call must be budgeted, i.e. issued by the evaluator
subprocess that `run_eval` spawns. Re-scoring cached outputs offline is fine.
The closing run-integrity agent can only *guess* at this from the architect
transcript; the squid gateway's access log can answer it mechanically, because
the runner network is `internal: true` and the gateway is the only path off
the host (`runner/compose.host.yml`, `runner/squid.conf`).

What this joins:

* the run's job log (`/artifacts/.codeevolver/logs/<job_id>.log`, UTC), which
  brackets every eval with `iteration N: run_eval start (... timeout=T)` /
  `iteration N: run_eval done (...)` lines, plus `seed-validate: running` /
  `seed-validate: primary=`; a timed-out eval logs a start with no done, so it
  is closed at `start + T`;
* the host gateway `access.log` (`logformat ce %ts.%03tu %>a %Ss/%03>Hs %<st
  %rm %ru %ssl::>sni`), one line per intercepted TLS step: the peek step logs
  as `NONE_NONE/000`, the spliced tunnel as `TCP_TUNNEL/200`, so a connection is
  roughly two lines, both stamped at close.

Anything after the last eval is the finalization phase (holdout evals run by
the root driver, then the meta-reflection / integrity agents, which use the
harness provider, not the task providers) and is treated as one window. The
startup egress canary probes the task providers from root in the first
seconds of the run and is excluded by `CANARY_GRACE`.

A task-provider connection outside every window is a *violation*: some
process other than the evaluator called the task model. It is listed with its
timestamp so the architect trace can be checked for the command that did it.

Container IPs are recycled across runs on a host, so the join keys on
(container_ip, [started_at, ended_at]). When a run did not record its IP,
`infer_container_ip` recovers it from the log (the IP whose traffic begins at
the run's start and that was silent for the hour before).

CLI:

    egress_audit.py --run-id <run_id> --host <host> \\
        [--gateway-log PATH] [--ip 172.31.66.4] [--start EPOCH] [--end EPOCH] \\
        [--grace 120] [--json out.json]

Exit 0 clean, 1 violations, 2 inputs missing.
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import glob
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

RUNS_HOME = Path(os.environ.get("CE_RUNS_DIR") or Path.home() / "ce-runs").expanduser()

# Task-LM providers: reachable only by the evaluator by policy. The harness
# provider (DeepInfra), Anthropic (constraint gate), OpenAI (RAGQA judge inside
# run_eval, OpenCode small model) and the retrievers are deliberately absent:
# the architect and its sub-agents legitimately reach them between evals.
TASK_PROVIDER_SNI: tuple[str, ...] = (
    "api.gmi-serving.com",
    "api.deepseek.com",
    "openrouter.ai",
)

# Seconds a connection may close after its window ends and still count as
# in-window: keep-alive sockets held by the evaluator close when it exits.
DEFAULT_GRACE = 120.0
# The entrypoint's egress canary probes the providers from root before the
# run starts spending; lines this soon after the run start are not evidence.
CANARY_GRACE = 90.0
# Evaluator timeout when the start line does not carry one.
DEFAULT_EVAL_TIMEOUT = 3000.0

_LOG_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ \| \w+ \| (?P<logger>\S+) \| (?P<msg>.*)$"
)
_EVAL_START = re.compile(
    r"iteration (?P<it>\d+): run_eval start \(sample=(?P<sample>\w+), suffix=(?P<suffix>\w+),"
    r" rows=\d+(?:, timeout=(?P<timeout>\d+))?"
)
_EVAL_DONE = re.compile(r"iteration (?P<it>\d+): run_eval done \(")
_SEED_START = "seed-validate: running environment check"
_SEED_DONE = "seed-validate: primary="
# Where the loop ends and the root driver's finalization begins. v3 arms log
# `finalization: scoring seed (...)` before the first holdout draw
# (v3/finalization.py); simple_loop has no such line — its last
# `simple loop session N: evals ...` summary is the loop's end, and the
# finalization has begun once a `finalization: test sample` line exists.
_FINALIZATION_START = "finalization: scoring"
_SIMPLE_LOOP_SESSION_END = re.compile(r"simple loop session \d+: evals ")
_HOLDOUT_SAMPLE = "finalization: test sample"


@dataclass(frozen=True)
class Window:
    start: float
    end: float
    kind: str  # eval | eval-timeout | seed | post-loop


def _log_ts(text: str) -> float:
    return (
        dt.datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        .replace(tzinfo=dt.timezone.utc)
        .timestamp()
    )


@dataclass
class LogFacts:
    windows: list[Window]
    first_ts: float | None
    last_ts: float | None
    finalization_start: float | None  # None while the loop is still running


def parse_eval_windows(lines, default_timeout: float = DEFAULT_EVAL_TIMEOUT) -> LogFacts:
    """Eval windows and phase markers from a job log.

    Starts and dones are paired FIFO per iteration; a start left open is a
    timed-out eval and closes at `start + timeout`. `finalization_start` is
    the loop's end (see the marker notes above) and None for a live run; the
    post-loop window itself needs the run end, so `audit` adds it.
    """
    windows: list[Window] = []
    open_starts: dict[str, list[tuple[float, float]]] = {}
    seed_start: float | None = None
    first = last = None
    finalization_start: float | None = None
    last_session_end: float | None = None
    holdout_seen = False
    for raw in lines:
        m = _LOG_LINE.match(raw.rstrip("\n"))
        if not m:
            continue
        ts = _log_ts(m.group("ts"))
        first = ts if first is None else first
        last = ts
        msg = m.group("msg")
        if _SEED_START in msg:
            seed_start = ts
        elif _SEED_DONE in msg and seed_start is not None:
            windows.append(Window(seed_start, ts, "seed"))
            seed_start = None
        elif (s := _EVAL_START.search(msg)):
            timeout = float(s.group("timeout") or default_timeout)
            open_starts.setdefault(s.group("it"), []).append((ts, timeout))
        elif (d := _EVAL_DONE.search(msg)):
            pending = open_starts.get(d.group("it"))
            start = pending.pop(0)[0] if pending else ts - 1.0
            windows.append(Window(start, ts, "eval"))
        elif _FINALIZATION_START in msg and finalization_start is None:
            finalization_start = ts
        elif _SIMPLE_LOOP_SESSION_END.search(msg):
            last_session_end = ts
        elif _HOLDOUT_SAMPLE in msg:
            holdout_seen = True
    for pending in open_starts.values():
        for start, timeout in pending:
            windows.append(Window(start, start + timeout, "eval-timeout"))
    windows.sort(key=lambda w: w.start)
    if finalization_start is None and holdout_seen and last_session_end is not None:
        finalization_start = last_session_end
    return LogFacts(windows, first, last, finalization_start)


@dataclass(frozen=True)
class AccessLine:
    ts: float
    client: str
    status: str
    sni: str
    elapsed_ms: float | None  # present once the logformat carries %tr
    local_port: int | None  # present once the logformat carries %>lp


def parse_access_line(line: str) -> AccessLine | None:
    p = line.split()
    if len(p) < 7:
        return None
    try:
        ts = float(p[0])
    except ValueError:
        return None
    elapsed = port = None
    if len(p) >= 9:
        try:
            elapsed = float(p[7])
            port = int(p[8])
        except ValueError:
            pass
    return AccessLine(ts, p[1], p[2], p[6], elapsed, port)


def is_task_provider(sni: str, task_sni=TASK_PROVIDER_SNI) -> bool:
    return any(sni == h or sni.endswith("." + h.lstrip(".")) or sni.endswith(h) for h in task_sni)


def _in_window(windows: list[Window], starts: list[float], t0: float, t1: float, grace: float) -> Window | None:
    """The window overlapping [t0 - grace, t1 + grace], if any."""
    i = bisect.bisect_right(starts, t1 + grace) - 1
    for j in (i, i - 1, i - 2):
        if 0 <= j < len(windows):
            w = windows[j]
            if w.start - grace <= t1 and t0 <= w.end + grace:
                return w
    return None


def audit(
    job_log: Path,
    gateway_log: Path,
    container_ip: str,
    started_at: float,
    ended_at: float,
    *,
    grace: float = DEFAULT_GRACE,
    task_sni: tuple[str, ...] = TASK_PROVIDER_SNI,
    max_listed: int = 50,
) -> dict:
    with open(job_log, errors="replace") as fh:
        facts = parse_eval_windows(fh)
    windows = list(facts.windows)
    run_start = started_at if started_at else (facts.first_ts or 0.0)
    loop_end = facts.finalization_start
    if loop_end is not None:
        # Holdout draws by the root driver, then the meta-reflection and
        # integrity agents (harness provider only). One window to the end.
        windows.append(Window(loop_end, ended_at, "post-loop"))
    windows.sort(key=lambda w: w.start)
    starts = [w.start for w in windows]

    task_lines = 0
    canary_lines = 0
    outside: list[AccessLine] = []
    with open(gateway_log, errors="replace") as fh:
        for raw in fh:
            al = parse_access_line(raw)
            if al is None or al.client != container_ip:
                continue
            if al.ts < run_start or al.ts > ended_at + grace:
                continue
            if not is_task_provider(al.sni, task_sni):
                continue
            task_lines += 1
            if al.ts < run_start + CANARY_GRACE:
                canary_lines += 1
                continue
            t0 = al.ts - (al.elapsed_ms or 0.0) / 1000.0
            if _in_window(windows, starts, t0, al.ts, grace) is None:
                outside.append(al)

    clusters: list[list[AccessLine]] = []
    for al in outside:
        if clusters and al.ts - clusters[-1][-1].ts <= 300:
            clusters[-1].append(al)
        else:
            clusters.append([al])

    def iso(t: float) -> str:
        return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "container_ip": container_ip,
        "started_at": run_start,
        "ended_at": ended_at,
        "job_log": str(job_log),
        "gateway_log": str(gateway_log),
        "eval_windows": sum(1 for w in windows if w.kind.startswith("eval")),
        "timed_out_evals": sum(1 for w in windows if w.kind == "eval-timeout"),
        "loop_end": loop_end,  # None while the loop is still running
        "task_provider_lines": task_lines,
        "canary_lines": canary_lines,
        # ~2 log lines per TLS connection (peek + splice).
        "violation_lines": len(outside),
        "violation_connections_est": (len(outside) + 1) // 2,
        "violation_clusters": [
            {
                "start": iso(c[0].ts),
                "end": iso(c[-1].ts),
                "lines": len(c),
                "hosts": sorted({al.sni for al in c}),
            }
            for c in clusters
        ],
        "violations": [
            {"ts": al.ts, "at": iso(al.ts), "sni": al.sni, "status": al.status}
            for al in outside[:max_listed]
        ],
        "verdict": "violations" if outside else "clean",
    }


# SNI fragments that identify a benchmark's container when several runs start
# on a host within seconds of each other.
BENCH_SNI_SIGNATURE: dict[str, str] = {
    "hover": "colbert-server-wiki",
    "phantomwiki": "phantom-wiki",
    "ragqa": "api.openai.com",  # the pairwise judge
}


def infer_container_ip(
    gateway_log: Path,
    started_at: float,
    ended_at: float | None = None,
    bench: str | None = None,
    *,
    window: float = 180.0,
    quiet_before: float = 3600.0,
) -> str | None:
    """Recover a run's container IP from the gateway log when the run did not
    record one. Candidates are the IPs whose traffic begins within `window` of
    `started_at` after an hour of silence; ties are broken by the run end
    (`ended_at`, the container's last line) and then by the benchmark's SNI
    signature. None when nothing or several things still qualify."""
    seen_before: set[str] = set()
    candidates: set[str] = set()
    last_seen: dict[str, float] = {}
    # per IP, per signature: how many lines carried it after the start
    sig_hits: dict[str, dict[str, int]] = {}
    with open(gateway_log, errors="replace") as fh:
        for raw in fh:
            al = parse_access_line(raw)
            if al is None:
                continue
            if started_at - quiet_before <= al.ts < started_at - window:
                seen_before.add(al.client)
            elif started_at - window <= al.ts <= started_at + window:
                candidates.add(al.client)
            if al.ts >= started_at - window:
                last_seen[al.client] = al.ts
                for b, sig in BENCH_SNI_SIGNATURE.items():
                    if sig in al.sni:
                        hits = sig_hits.setdefault(al.client, {})
                        hits[b] = hits.get(b, 0) + 1

    def signed_as(ip: str) -> set[str]:
        """Benchmarks whose signature this IP carries. The openai judge counts
        only in volume: other arms touch api.openai.com a handful of times."""
        out = set()
        for b, n in sig_hits.get(ip, {}).items():
            if b == "ragqa" and n < 500:
                continue
            out.add(b)
        return out

    live = candidates - seen_before
    if len(live) > 1 and ended_at is not None:
        by_end = {ip for ip in live if abs(last_seen.get(ip, 0.0) - ended_at) <= 600}
        live = by_end or live
    if len(live) > 1 and bench:
        if bench in BENCH_SNI_SIGNATURE:
            by_sig = {ip for ip in live if bench in signed_as(ip)}
        else:  # paperreview / arcagi2: whatever carries no retriever/judge signature
            by_sig = {ip for ip in live if not signed_as(ip)}
        live = by_sig or live
    return next(iter(live)) if len(live) == 1 else None


def resolve_job_log(run_dir: Path) -> Path | None:
    """The run's optimizer log: the staged copy under the run dir (present
    mid-run and kept after the final collect), else the merged copy under
    ~/.codeevolver/logs/<job_id>.log. A run dir can hold a second, aborted
    job's log (a re-dispatch); the one with the most eval starts is the run."""
    best, best_n = None, -1
    patterns = [str(run_dir / ".codeevolver" / "logs" / "job_*.log")]
    status = run_dir / "status.json"
    if status.exists():
        try:
            job_id = json.loads(status.read_text()).get("job_id")
        except (OSError, json.JSONDecodeError):
            job_id = None
        if job_id:
            patterns.append(str(Path.home() / ".codeevolver" / "logs" / f"{job_id}.log"))
    for pat in patterns:
        for path in glob.glob(pat):
            if "detailed_agent_log" in path:
                continue
            try:
                with open(path, errors="replace") as fh:
                    n = sum(1 for line in fh if "run_eval start" in line)
            except OSError:
                continue
            if n > best_n:
                best, best_n = Path(path), n
    return best


def gateway_log_path(host: str) -> Path:
    return RUNS_HOME / f"gateway-{host}.access.log"


def summarize(result: dict) -> str:
    est = result["violation_connections_est"]
    return (
        f"{result['verdict']}: {est} task-provider connection(s) outside eval windows "
        f"({result['violation_lines']} log lines of {result['task_provider_lines']}; "
        f"{result['eval_windows']} eval windows, {result['timed_out_evals']} timed out)"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--host", help="fleet host; gateway log defaults to ~/ce-runs/gateway-<host>.access.log")
    ap.add_argument("--gateway-log", type=Path)
    ap.add_argument("--ip", help="container IP on ce-internal (inferred from the log when omitted)")
    ap.add_argument("--start", type=float, help="run start (epoch); default: first job-log line")
    ap.add_argument("--end", type=float, help="run end (epoch); default: status.json finished_at or now")
    ap.add_argument("--grace", type=float, default=DEFAULT_GRACE)
    ap.add_argument("--json", type=Path, help="write the full result here")
    args = ap.parse_args(argv)

    run_dir = RUNS_HOME / args.run_id
    job_log = resolve_job_log(run_dir)
    if job_log is None:
        print(f"{args.run_id}: no job log under {run_dir}", file=sys.stderr)
        return 2
    gateway_log = args.gateway_log or (gateway_log_path(args.host) if args.host else None)
    if gateway_log is None or not gateway_log.exists():
        print(f"{args.run_id}: gateway log missing ({gateway_log})", file=sys.stderr)
        return 2

    with open(job_log, errors="replace") as fh:
        facts = parse_eval_windows(fh)
    started = args.start or facts.first_ts
    ended = args.end
    if ended is None:
        status = run_dir / "status.json"
        if status.exists():
            try:
                fin = json.loads(status.read_text()).get("finished_at")
                if fin:
                    ended = _log_ts(fin.replace("T", " ").rstrip("Z"))
            except (OSError, json.JSONDecodeError, ValueError):
                ended = None
    if ended is None:
        ended = dt.datetime.now(dt.timezone.utc).timestamp()
    if started is None:
        print(f"{args.run_id}: job log has no timestamped lines", file=sys.stderr)
        return 2

    bench = None
    settings = run_dir / "settings.json"
    if settings.exists():
        try:
            bench = json.loads(settings.read_text()).get("bench")
        except (OSError, json.JSONDecodeError):
            bench = None
    if not bench and args.run_id.count("-") >= 2:
        bench = args.run_id.split("-")[1]
    ip = args.ip or infer_container_ip(gateway_log, started, ended, bench)
    if not ip:
        print(f"{args.run_id}: cannot infer container IP at {started}; pass --ip", file=sys.stderr)
        return 2

    result = audit(job_log, gateway_log, ip, started, ended, grace=args.grace)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
    print(f"{args.run_id} ({ip}): {summarize(result)}")
    for c in result["violation_clusters"]:
        print(f"  {c['start']} -> {c['end']} lines={c['lines']} {c['hosts']}")
    return 1 if result["violation_lines"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
