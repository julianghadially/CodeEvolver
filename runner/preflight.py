#!/usr/bin/env python3
"""Copyright © 2026 440 Labs LLC

Egress canary — run before any provider spend, inside the runner container.

The research guarantee is that an agent cannot reach the public git hosts that
carry the original benchmark repositories (whose history still contains the
holdout test sets). That guarantee is enforced by the squid SNI allowlist on
the host, but a misconfigured network would fail *open* and silently produce a
contaminated run. This module turns the guarantee into a per-run recorded
assertion: DENIED hosts must be unreachable and ALLOWED hosts must be
reachable, or the run aborts before it costs anything.

Node is probed separately and deliberately: Node/undici does not honour
HTTP_PROXY, so an env-var-only proxy would fence Python while leaving the
OpenCode harness with open internet. If node is present it must be fenced too.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
from typing import Any

# Hosts that must NOT be reachable. The first three are the contamination
# channel we actually care about; the search engines close the "look up the
# answer" path.
DENIED_HOSTS = [
    "github.com",
    "raw.githubusercontent.com",
    "huggingface.co",
    "www.google.com",
]

# Hosts EVERY run genuinely needs. Kept narrow on purpose: every entry here is
# a hole in the fence and has to be justifiable in the paper.
ALLOWED_HOSTS = [
    "api.anthropic.com",
    "api.deepinfra.com",
    # DeepSeek's first-party API. Every wired benchmark's seed program --
    # appworld and alfworld included -- pins DeepSeek-V4-Flash behind the
    # gmi -> deepseek -> deepinfra provider chain, so this is a
    # run-wide requirement, not a per-benchmark one. A host whose
    # /opt/ce/squid.conf predates the `.deepseek.com` ACL hard-aborts here
    # at startup until it is re-provisioned.
    "api.deepseek.com",
    "api.gmi-serving.com",
    "pypi.org",
]

# Providers only ONE benchmark calls. Listing these in ALLOWED_HOSTS would make
# every benchmark REQUIRE them: `check()` folds an unreachable allowed host into
# a False verdict and entrypoint.py turns that into a hard abort, so a hover run
# on a host whose /opt/ce/squid.conf predates the entry would die at startup over
# a provider it never calls. Keyed by CE_BENCH, which the image bakes in
# (Dockerfile.bench) and dispatch() re-passes per run.
#
# This scopes the REQUIREMENT, not the fence. There is one squid.conf per host,
# shared by every benchmark on it, so the SNI allowlist stays host-wide: keeping
# a benchmark-specific provider genuinely unreachable for the others means not
# co-scheduling them on the same host.
BENCH_ALLOWED_HOSTS = {
    # ALFWorld's task model routes through OpenRouter (openrouter.ai/api/v1).
    "alfworld": ["openrouter.ai"],
}


def allowed_hosts() -> list[str]:
    """The reachability contract for THIS run: shared providers + this bench's."""
    return ALLOWED_HOSTS + BENCH_ALLOWED_HOSTS.get(
        os.environ.get("CE_BENCH", "").strip(), []
    )


PROBE_TIMEOUT = 10.0


def _probe(host: str, port: int = 443, timeout: float = PROBE_TIMEOUT) -> dict[str, Any]:
    """Reachability probe that completes a real TLS handshake, not just a TCP one.

    The fence is a squid SNI allowlist reached by TRANSPARENT interception: the
    runner's default route points at the gateway container, whose PREROUTING
    REDIRECT sends :443 to squid's ssl-bump listener. Squid therefore accepts
    the TCP connection for every destination and only decides once it has
    peeked at the ClientHello. A bare `socket.create_connection` consequently
    succeeds against github.com just as it does against api.anthropic.com, and
    would report the fence WIDE OPEN on a correctly configured host.

    Sending SNI and finishing the handshake is the probe that matches the
    control: an allowed host splices through to the origin and presents the
    origin's certificate, a denied one is cut off with no usable session. The
    default context verifies that certificate, so a spliced connection is also
    proof that nothing is being MITM'd.

    When HTTPS_PROXY is set we probe through it instead — a direct socket would
    then fail merely because the container has no route, which would also
    "pass" on a proxy whose allowlist is wide open. Issuing a real CONNECT
    tests the thing we care about: what the allowlist permits.
    """
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    try:
        if proxy:
            target = proxy.split("://", 1)[-1]
            p_host, _, p_port = target.partition(":")
            with socket.create_connection((p_host, int(p_port or 3128)), timeout) as sock:
                sock.settimeout(timeout)
                sock.sendall(
                    f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode()
                )
                status_line = sock.recv(256).decode("latin-1", "replace").split("\r\n")[0]
            ok = " 200" in status_line
            return {"host": host, "reachable": ok, "via": "proxy", "detail": status_line.strip()}
        ctx = ssl.create_default_context()
        with socket.create_connection((host, port), timeout) as sock:
            sock.settimeout(timeout)
            with ctx.wrap_socket(sock, server_hostname=host) as tls:
                detail = f"handshake ok ({tls.version()})"
        return {"host": host, "reachable": True, "via": "direct", "detail": detail}
    except Exception as exc:  # noqa: BLE001 - any failure is "unreachable"
        return {"host": host, "reachable": False, "via": "proxy" if proxy else "direct",
                "detail": f"{type(exc).__name__}: {exc}"}


def _probe_node(host: str, timeout: float = PROBE_TIMEOUT) -> dict[str, Any] | None:
    """Probe from Node, which is the runtime OpenCode's HTTP client lives in."""
    if not shutil.which("node"):
        return None
    script = (
        "const c=new AbortController();"
        f"setTimeout(()=>c.abort(),{int(timeout * 1000)});"
        f"fetch('https://{host}',{{signal:c.signal}})"
        ".then(r=>{console.log('REACHABLE '+r.status);process.exit(0)})"
        ".catch(e=>{console.log('BLOCKED '+e.message);process.exit(0)});"
    )
    try:
        out = subprocess.run(
            ["node", "-e", script], capture_output=True, text=True, timeout=timeout + 10
        ).stdout.strip()
    except subprocess.TimeoutExpired:
        out = "BLOCKED timeout"
    return {"host": host, "runtime": "node", "reachable": out.startswith("REACHABLE"),
            "detail": out}


def _check_agent_user() -> dict[str, Any]:
    """Prove the unprivileged agent user is usable BEFORE any spend.

    The engine spawns every agent subprocess with `user=$CODEEVOLVER_AGENT_USER`
    (see `src/engine/paths.py::agent_user`) so the kernel — not deny paths —
    fences the sequestered holdout. If the user is missing or the container
    lacks CAP_SETUID/CAP_SETGID (the orchestrator matrix must cap-add them
    back after --cap-drop=ALL), the very first architect session would die
    mid-run; fail here instead.
    """
    user = os.environ.get("CODEEVOLVER_AGENT_USER", "").strip()
    if not user:
        return {"configured": False, "ok": True, "detail": "CODEEVOLVER_AGENT_USER unset"}
    try:
        import pwd

        pwd.getpwnam(user)
        subprocess.run(
            ["id"], user=user, capture_output=True, check=True, timeout=15,
        )
        return {"configured": True, "ok": True, "detail": f"setuid({user}) works"}
    except Exception as exc:  # noqa: BLE001 - any failure means the fence is unusable
        return {
            "configured": True,
            "ok": False,
            "detail": f"cannot run as {user!r}: {type(exc).__name__}: {exc}",
        }


def _check_agent_git(workspace: str | None = None) -> dict[str, Any]:
    """Prove git works AS THE AGENT USER inside the workspace BEFORE any spend.

    The workspace is cloned by root and only chmod'ed open; git's
    `safe.directory` check is about ownership, not mode, so without
    `entrypoint.trust_workspace_for_agent` every agent-side git command dies
    with "dubious ownership" — the architect cannot commit, and the v3 close
    protocol depends on the agent's git. `git rev-parse HEAD` as the agent
    user is the exact command shape the agent runs.
    """
    user = os.environ.get("CODEEVOLVER_AGENT_USER", "").strip()
    if not user:
        return {"configured": False, "ok": True, "detail": "CODEEVOLVER_AGENT_USER unset"}
    workspace = workspace or os.environ.get("CE_WORKSPACE", "/workspace")
    if not os.path.isdir(os.path.join(workspace, ".git")):
        return {"configured": True, "ok": True,
                "detail": f"{workspace} is not a git repository — nothing to prove"}
    try:
        proc = subprocess.run(
            ["git", "-C", workspace, "rev-parse", "HEAD"],
            user=user, capture_output=True, text=True, timeout=15,
        )
    except Exception as exc:  # noqa: BLE001
        return {"configured": True, "ok": False,
                "detail": f"git as {user!r} raised: {type(exc).__name__}: {exc}"}
    if proc.returncode != 0:
        return {"configured": True, "ok": False,
                "detail": f"git as {user!r} failed in {workspace}: {proc.stderr.strip()}"}
    return {"configured": True, "ok": True,
            "detail": f"git as {user!r} sees HEAD {proc.stdout.strip()[:12]}"}


def _check_ripgrep() -> dict[str, Any]:
    """Prove `rg` is on PATH BEFORE any spend.

    OpenCode's grep / glob / skill tools shell out to ripgrep. It resolves the
    binary as PATH -> `$HOME/.cache/opencode/bin/rg` -> download from
    github.com; the egress fence terminates github.com by design, so an image
    without `rg` on PATH loses all three tools for every agent in the run
    (architect, inspect_traces, run-integrity, meta-reflection) while the run
    itself reports success. Only PATH counts here: the cache is a bind mount
    that is empty on a fresh run.
    """
    path = shutil.which("rg")
    if not path:
        return {"ok": False, "detail": "rg not on PATH (runner/Dockerfile.base must install ripgrep)"}
    try:
        out = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=15,
        ).stdout.strip().splitlines()
        return {"ok": True, "detail": f"{path}: {out[0] if out else 'ok'}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": f"{path} failed: {type(exc).__name__}: {exc}"}


def check() -> dict[str, Any]:
    """Run every probe and return a structured report with an `ok` verdict."""
    denied = [_probe(h) for h in DENIED_HOSTS]
    allowed = [_probe(h) for h in allowed_hosts()]
    node_denied = [r for r in (_probe_node(h) for h in DENIED_HOSTS[:2]) if r]
    agent_user = _check_agent_user()
    agent_git = _check_agent_git()
    ripgrep = _check_ripgrep()

    leaks = [r["host"] for r in denied if r["reachable"]]
    leaks += [f"{r['host']} (node)" for r in node_denied if r["reachable"]]
    unreachable = [r["host"] for r in allowed if not r["reachable"]]

    return {
        "ok": (
            not leaks and not unreachable
            and agent_user["ok"] and agent_git["ok"] and ripgrep["ok"]
        ),
        "leaked_hosts": leaks,
        "unreachable_allowed_hosts": unreachable,
        "agent_user": agent_user,
        "agent_git": agent_git,
        "ripgrep": ripgrep,
        "bench": os.environ.get("CE_BENCH", ""),
        "proxy": os.environ.get("HTTPS_PROXY", ""),
        "denied_probes": denied,
        "allowed_probes": allowed,
        "node_probes": node_denied,
    }


def main() -> int:
    report = check()
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/artifacts/egress.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)

    if report["leaked_hosts"]:
        print(f"EGRESS FENCE OPEN — reachable: {report['leaked_hosts']}", file=sys.stderr)
    if report["unreachable_allowed_hosts"]:
        print(
            f"REQUIRED HOSTS UNREACHABLE: {report['unreachable_allowed_hosts']}",
            file=sys.stderr,
        )
    if not report["agent_user"]["ok"]:
        print(f"AGENT USER UNUSABLE: {report['agent_user']['detail']}", file=sys.stderr)
    if not report["agent_git"]["ok"]:
        print(f"AGENT GIT UNUSABLE: {report['agent_git']['detail']}", file=sys.stderr)
    if not report["ripgrep"]["ok"]:
        print(f"RIPGREP MISSING: {report['ripgrep']['detail']}", file=sys.stderr)
    print(f"egress preflight: {'PASS' if report['ok'] else 'FAIL'} -> {out_path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
