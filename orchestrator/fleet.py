#!/usr/bin/env python3
"""Copyright © 2026 440 Labs LLC

Droplet lifecycle for one fleet matrix: create the hosts at launch, destroy
them once every run has been collected.

    python3 orchestrator/fleet.py up [--provision]
    python3 orchestrator/fleet.py status
    python3 orchestrator/fleet.py down [--force]

Everything about the hosts -- region, size, image, SSH key, project, and the
tag that groups them -- comes from the matrix's `fleet:` block (CE_MATRIX), so
the droplets a matrix needs are declared next to the runs it dispatches to
them. The DigitalOcean token is DIGITALOCEAN_API_KEY. No doctl and no SDK: the
handful of calls involved are plain HTTPS, and nothing is installed anywhere.

`down` is the dangerous one, so it is gated. It refuses while any matrix run is
not terminal, any run's artifacts are not on this machine, or any ce-*
container is still up on a host: an LM budget already spent is the one thing
tearing down a host can lose, and /data/runs goes with the droplet. --force is
for a host you have given up on.

Idempotent: `up` skips hosts that already exist under the tag, and refuses to
touch a droplet that has a matrix host's name but not the tag (a colleague's).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from controller import (
    HERE, MATRIX_PATH, TERMINAL, build_runs, load_matrix, load_state, log,
    running_runs,
)
from fleet_config import runs_dir

API = "https://api.digitalocean.com/v2"
REQUIRED = ("tag", "region", "size", "image", "ssh_key", "project")


# --------------------------------------------------------------------------
# DigitalOcean API
# --------------------------------------------------------------------------

def _token() -> str:
    tok = os.environ.get("DIGITALOCEAN_API_KEY")
    if not tok:
        raise SystemExit("DIGITALOCEAN_API_KEY is not set (source env/bin/activate)")
    return tok


def api(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {_token()}",
                 "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise SystemExit(f"DigitalOcean {method} {path} -> {e.code}: {detail[:300]}")


def paged(path: str, key: str) -> list[dict]:
    out: list[dict] = []
    page = 1
    while True:
        sep = "&" if "?" in path else "?"
        d = api("GET", f"{path}{sep}per_page=200&page={page}")
        out.extend(d.get(key, []))
        if not d.get("links", {}).get("pages", {}).get("next"):
            return out
        page += 1


def tagged_droplets(tag: str) -> dict[str, dict]:
    return {d["name"]: d for d in paged(f"/droplets?tag_name={tag}", "droplets")}


def public_ip(droplet: dict) -> str | None:
    for net in droplet.get("networks", {}).get("v4", []):
        if net.get("type") == "public":
            return net["ip_address"]
    return None


def ssh_key_id(name: str) -> int:
    for k in paged("/account/keys", "ssh_keys"):
        if k["name"] == name:
            return k["id"]
    raise SystemExit(f"no SSH key named {name!r} on the DigitalOcean account")


def project_id(name: str) -> str:
    for p in paged("/projects", "projects"):
        if p["name"] == name:
            return p["id"]
    raise SystemExit(f"no DigitalOcean project named {name!r}")


def check_size_in_region(region: str, size: str) -> None:
    """Fail before creating anything: a size missing from a region is a 422
    per droplet, and the 64 GB tiers are only offered in some regions."""
    for r in paged("/regions", "regions"):
        if r["slug"] != region:
            continue
        if not r["available"]:
            raise SystemExit(f"region {region} is not accepting new droplets")
        if size not in r["sizes"]:
            similar = [s for s in r["sizes"] if s.endswith(size.rsplit("-", 1)[-1])]
            raise SystemExit(f"size {size} is not offered in {region}; "
                             f"same-memory sizes there: {similar or 'none'}")
        return
    raise SystemExit(f"unknown region {region!r}")


# --------------------------------------------------------------------------
# matrix + local state
# --------------------------------------------------------------------------

def fleet_cfg(matrix: dict) -> dict:
    cfg = matrix.get("fleet")
    if not cfg:
        raise SystemExit(f"{MATRIX_PATH}: no `fleet:` block — this matrix does "
                         f"not declare its droplets (see matrix.main.yaml)")
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise SystemExit(f"{MATRIX_PATH}: fleet block is missing {missing}")
    return cfg


def state_file(cfg: dict) -> Path:
    return HERE / f"fleet.{cfg['tag']}.json"


# --------------------------------------------------------------------------
# waits
# --------------------------------------------------------------------------

def wait_active(hosts: list[str], tag: str, timeout: float = 600) -> dict[str, dict]:
    """Until every host is `active` with a public IPv4 (DO takes ~30-60s)."""
    deadline = time.time() + timeout
    while True:
        droplets = tagged_droplets(tag)
        ready = {h: droplets[h] for h in hosts
                 if h in droplets and droplets[h]["status"] == "active"
                 and public_ip(droplets[h])}
        if len(ready) == len(hosts):
            return ready
        if time.time() > deadline:
            pending = [h for h in hosts if h not in ready]
            raise SystemExit(f"timed out waiting for droplets: {pending}")
        time.sleep(10)


def wait_ssh(host: str, ip: str, timeout: float = 300) -> None:
    """Until sshd answers with our key. cloud-init installs the key a little
    after the droplet reports active, so the first attempts usually fail."""
    deadline = time.time() + timeout
    while True:
        res = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             "-o", "StrictHostKeyChecking=accept-new", f"root@{ip}", "true"],
            capture_output=True, text=True,
        )
        if res.returncode == 0:
            log(f"{host}: ssh ready at {ip}")
            return
        if time.time() > deadline:
            raise SystemExit(f"{host}: ssh never came up at {ip}: {res.stderr.strip()[-200:]}")
        time.sleep(10)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_up(args) -> int:
    matrix = load_matrix()
    cfg = fleet_cfg(matrix)
    hosts = list(matrix["hosts"])
    if args.hosts:
        unknown = [h for h in args.hosts if h not in hosts]
        if unknown:
            raise SystemExit(f"not in {MATRIX_PATH.name}'s hosts: {unknown}")
        hosts = [h for h in hosts if h in args.hosts]
    check_size_in_region(cfg["region"], cfg["size"])
    key = ssh_key_id(cfg["ssh_key"])
    proj = project_id(cfg["project"])

    existing = tagged_droplets(cfg["tag"])
    everyone = {d["name"] for d in paged("/droplets", "droplets")}
    created: list[int] = []
    for host in hosts:
        if host in existing:
            log(f"{host}: already exists ({existing[host]['status']}, id {existing[host]['id']})")
            continue
        if host in everyone:
            raise SystemExit(f"{host}: a droplet with this name exists but is not "
                             f"tagged {cfg['tag']!r} — not touching it")
        d = api("POST", "/droplets", {
            "name": host, "region": cfg["region"], "size": cfg["size"],
            "image": cfg["image"], "ssh_keys": [key], "tags": [cfg["tag"]],
            "monitoring": True, "ipv6": False,
        })["droplet"]
        created.append(d["id"])
        log(f"{host}: creating {cfg['size']} in {cfg['region']} (id {d['id']})")

    if created:
        # Create does not take a project; the default project gets it otherwise.
        api("POST", f"/projects/{proj}/resources",
            {"resources": [f"do:droplet:{i}" for i in created]})
        log(f"assigned {len(created)} droplet(s) to project {cfg['project']!r}")

    ready = wait_active(hosts, cfg["tag"])
    record = {h: {"id": d["id"], "ip": public_ip(d)} for h, d in ready.items()}
    for host, info in record.items():
        wait_ssh(host, info["ip"])
    state_file(cfg).write_text(json.dumps(record, indent=2) + "\n")
    log(f"wrote {state_file(cfg)}")

    print("\nprovision with:")
    for host, info in record.items():
        print(f"  ./orchestrator/provision.sh {host} {info['ip']}")
    if args.provision:
        for host, info in record.items():
            log(f"provisioning {host}")
            subprocess.run([str(HERE / "provision.sh"), host, info["ip"]], check=True)
    return 0


def cmd_status(args) -> int:  # noqa: ARG001
    matrix = load_matrix()
    cfg = fleet_cfg(matrix)
    droplets = tagged_droplets(cfg["tag"])
    price = float(cfg.get("price_hourly", 0) or 0)
    total = 0.0
    print(f"{'host':12} {'id':>10} {'ip':16} {'status':8} {'hours':>6} {'cost':>8}")
    for host in matrix["hosts"]:
        d = droplets.get(host)
        if not d:
            print(f"{host:12} {'-':>10} {'-':16} {'absent':8}")
            continue
        created = time.strptime(d["created_at"], "%Y-%m-%dT%H:%M:%SZ")
        hours = (time.time() - time.mktime(created) + time.timezone) / 3600
        cost = hours * price
        total += cost
        print(f"{host:12} {d['id']:>10} {public_ip(d) or '-':16} {d['status']:8} "
              f"{hours:6.1f} {'$' + format(cost, '.2f'):>8}")
    if price:
        print(f"{'':12} {'':>10} {'':16} {'':8} {'':>6} {'$' + format(total, '.2f'):>8}  "
              f"(at ${price}/hr each)")
    return 0


def cmd_down(args) -> int:
    matrix = load_matrix()
    cfg = fleet_cfg(matrix)
    droplets = tagged_droplets(cfg["tag"])
    targets = {h: droplets[h] for h in matrix["hosts"] if h in droplets}
    if not targets:
        log(f"no droplets tagged {cfg['tag']!r} match this matrix — nothing to do")
        return 0

    if not args.force:
        problems: list[str] = []
        state = load_state()
        for run in build_runs(matrix):
            status = state["runs"].get(run.run_id, {}).get("status")
            if status not in TERMINAL:
                problems.append(f"{run.run_id}: {status or 'never dispatched'}")
            elif not (runs_dir() / run.run_id / "status.json").exists():
                problems.append(f"{run.run_id}: artifacts not collected to {runs_dir()}")
        for host in targets:
            live = running_runs(host)
            if live:
                problems.append(f"{host}: still running {', '.join(live)}")
        if problems:
            for p in problems:
                log(f"  {p}")
            raise SystemExit("refusing to destroy hosts with uncollected work "
                             "(--force to override)")

    for host, d in targets.items():
        # Best-effort: the host-level squid access.log is per-campaign egress
        # evidence and lives nowhere else once the droplet is gone. A failure
        # must never block the teardown.
        subprocess.run(
            ["scp", "-o", "ConnectTimeout=10",
             f"root@{public_ip(d)}:/data/gateway-logs/access.log",
             str(runs_dir() / f"gateway-{host}.access.log")],
            capture_output=True, check=False,
        )
        api("DELETE", f"/droplets/{d['id']}")
        subprocess.run(["docker", "context", "rm", "-f", host], capture_output=True)
        log(f"{host}: destroyed (id {d['id']}, was {public_ip(d)})")
    state_file(cfg).unlink(missing_ok=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="fleet", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("up", help="create the matrix's droplets (skips existing)")
    up.add_argument("--provision", action="store_true",
                    help="run provision.sh on each host once it is reachable")
    up.add_argument("--hosts", nargs="+", metavar="HOST",
                    help="only these matrix hosts (e.g. one build host first)")
    up.set_defaults(func=cmd_up)
    st = sub.add_parser("status", help="the matrix's droplets, uptime and cost")
    st.set_defaults(func=cmd_status)
    dn = sub.add_parser("down", help="destroy the droplets once everything is collected")
    dn.add_argument("--force", action="store_true",
                    help="destroy even with uncollected or running work")
    dn.set_defaults(func=cmd_down)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
