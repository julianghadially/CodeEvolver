#!/usr/bin/env bash
# Copyright © 2026 440 Labs LLC
# One-time host setup. This is the ENTIRE server-configuration burden: a
# droplet from DigitalOcean's Docker marketplace image plus this script. There
# is no CodeEvolver install on the worker, no Python, no per-server script to
# keep in sync -- the controller drives everything over SSH.
#
#   ./provision.sh ce-a 203.0.113.10 [--ragqa]
set -euo pipefail

HOST_NAME="${1:?usage: CE_BENCHMARKS=... provision.sh <host-name> <ip>}"
HOST_IP="${2:?usage: CE_BENCHMARKS=... provision.sh <host-name> <ip>}"
ADDR="root@${HOST_IP}"
# The container registry the images are pushed to, e.g.
# registry.digitalocean.com/<your-registry> (the same value as the controller's).
REGISTRY="${REGISTRY:?set REGISTRY to your container registry}"
TAG="${TAG:-latest}"

# This researcher's benchmark subset drives which images the host pulls and
# whether the retriever sidecar starts. A colleague not running ragqa never
# downloads its image and never starts its ~8GB retriever.
HERE="$(cd "$(dirname "$0")" && pwd)"
BENCHES="$(python3 "${HERE}/fleet_config.py" --slugs)"
NEEDS_RETRIEVER="$(python3 "${HERE}/fleet_config.py" --needs-retriever)"
NEEDS_GPU="$(python3 "${HERE}/fleet_config.py" --needs-gpu)"
echo "==> benchmarks for ${HOST_NAME}: ${BENCHES}  (retriever: ${NEEDS_RETRIEVER}, gpu: ${NEEDS_GPU})"

# A droplet reports `active` before cloud-init has installed the key, and a
# re-run can land while a previous attempt's apt run is still restarting sshd.
# Either way the first ssh fails with a refusal; wait rather than abort.
for _ in $(seq 1 30); do
  ssh -o ConnectTimeout=5 -o BatchMode=yes "${ADDR}" true 2>/dev/null && break
  sleep 5
done

echo "==> docker context ${HOST_NAME} -> ${ADDR}"
# Reuse one SSH connection across the controller's polls instead of a fresh
# handshake per docker call.
mkdir -p ~/.ssh/cm
# Per-HOST guard: every host needs its own block. Without multiplexing each
# ssh call opens a new TCP connection, and the marketplace image's `ufw limit
# 22/tcp` rejects a source after 6 new connections in 30 s, which reads as a
# persistent "connection refused" mid-provision.
grep -q "^Host ${HOST_IP}\$" ~/.ssh/config 2>/dev/null || cat >> ~/.ssh/config <<SSHCFG

Host ${HOST_IP}
    ControlMaster auto
    ControlPath ~/.ssh/cm/%r@%h:%p
    ControlPersist 10m
    ServerAliveInterval 30
SSHCFG
docker context rm -f "${HOST_NAME}" 2>/dev/null || true
docker context create "${HOST_NAME}" --docker "host=ssh://${ADDR}"

# Pulls (and, for a build droplet, pushes) are done by the HOST's daemon, so
# the host needs its own registry login; the controller's credentials never
# travel with a `docker --context` call. DigitalOcean takes the API token as
# both username and password. The token goes over stdin and is read into a
# shell variable on the far side, so it never appears on a command line or in
# the remote process list.
echo "==> registry login on ${HOST_NAME}"
: "${DIGITALOCEAN_API_KEY:?DIGITALOCEAN_API_KEY is required (source env/bin/activate)}"
printf '%s\n' "${DIGITALOCEAN_API_KEY}" | ssh "${ADDR}" \
  "IFS= read -r T; printf '%s' \"\$T\" | docker login ${REGISTRY%%/*} -u \"\$T\" --password-stdin"

echo "==> host prerequisites"
ssh "${ADDR}" bash -s <<'REMOTE'
set -euo pipefail
mkdir -p /data/runs /data/gateway-logs /opt/ce
# Squid in the gateway container runs as its unprivileged 'proxy' user
# (uid:gid 13:13 in the ubuntu base, see runner/Dockerfile.gateway) and writes
# /var/log/squid, which is this host dir -- root-only perms crash-loop the
# gateway on first start. The access log doubles as research-integrity
# evidence (squid.conf), so grant the proxy user and nobody else: no
# world-writable bit for anything unprivileged that lands on the host later.
chown 13:13 /data/gateway-logs
chmod 750 /data/gateway-logs
# gVisor: a real syscall boundary rather than relying on runc. Enable it in
# the matrix (defaults.runtime: runsc) once this is in place.
if ! command -v runsc >/dev/null 2>&1; then
  # --batch --yes: a re-run after an interrupted attempt finds the keyring
  # already present, and gpg's overwrite prompt needs a tty ssh does not give.
  curl -fsSL https://gvisor.dev/archive.key | gpg --batch --yes --dearmor -o /usr/share/keyrings/gvisor.gpg
  echo "deb [signed-by=/usr/share/keyrings/gvisor.gpg] https://storage.googleapis.com/gvisor/releases release main" \
    > /etc/apt/sources.list.d/gvisor.list
  apt-get update -qq && apt-get install -y -qq runsc || echo "gVisor install skipped"
  runsc install 2>/dev/null || true
  systemctl restart docker || true
fi
REMOTE
# The apt run above triggers needrestart, which restarts sshd on a fresh
# droplet (and the docker restart briefly drops the daemon); the next ssh can
# land in that window and fail with "connection refused". Wait it out.
for _ in $(seq 1 30); do
  ssh -o ConnectTimeout=5 "${ADDR}" true 2>/dev/null && break
  sleep 5
done

echo "==> egress gateway"
ssh "${ADDR}" "mkdir -p /opt/ce/ssl"
scp "$(dirname "$0")/../runner/squid.conf" "${ADDR}:/opt/ce/squid.conf"
scp "$(dirname "$0")/../runner/compose.host.yml" "${ADDR}:/opt/ce/compose.host.yml"
# The gateway is BUILT on the host rather than pulled: the stock ubuntu/squid
# image is a GnuTLS build (no SslBump -> cannot read TLS SNI) and ships no
# iptables (cannot install the intercept rules). Building here keeps it off the
# registry entirely. See runner/Dockerfile.gateway.
scp "$(dirname "$0")/../runner/Dockerfile.gateway" "${ADDR}:/opt/ce/Dockerfile.gateway"
# Self-signed cert for the intercept listener. No client ever validates it --
# peek+splice reads the SNI and hands the connection through untouched, so
# nothing is decrypted and no CA has to be trusted anywhere.
ssh "${ADDR}" "test -f /opt/ce/ssl/squid.pem || openssl req -new -newkey rsa:2048 -days 3650 \
  -nodes -x509 -subj '/CN=ce-gateway' -keyout /opt/ce/ssl/squid.pem -out /opt/ce/ssl/squid.pem"

# GPU benchmarks (alfworld: AI2-THOR/Unity) render against a headless Xorg on
# the HOST, shared into run containers by dispatch() as DISPLAY=:99 plus the
# X socket and Xauthority mounts. This is the qualified lifecycle-v2 recipe
# from cloud/alfworld-docker/prepare_gpu_host.sh, made idempotent: a host that
# already has the display up is left alone. Requires a GPU droplet whose image
# ships the NVIDIA driver + container toolkit (nvidia-smi / nvidia-ctk).
# Hosts without a GPU are fine too -- containers then fall back to in-container
# Xvfb software rendering (~5x slower), so this block is best-effort only when
# the subset needs it.
if [[ "${NEEDS_GPU}" == "yes" ]]; then
  echo "==> GPU display (headless Xorg) for AI2-THOR benchmarks"
  # Non-fatal on purpose: `set -e` would otherwise let a GPU-host hiccup
  # abort the script BEFORE the egress gateway and the ragqa retriever are
  # brought up below, taking down every benchmark on the host over an
  # alfworld-only prerequisite. dispatch() probes the display at run time
  # (controller.py::_host_has_gpu_display) and degrades to in-container
  # Xvfb, so a host that fails here is slower, not broken.
  scp "${HERE}/xorg-headless.conf" "${ADDR}:/opt/ce/xorg-headless.conf" \
    || echo "    could not copy xorg-headless.conf — skipping Xorg" >&2
  ssh "${ADDR}" bash -s <<'REMOTE' \
    || echo "    GPU display setup failed — alfworld falls back to Xvfb (~5x slower)" >&2
set -euo pipefail
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "    no NVIDIA GPU on this host — skipping Xorg; runs fall back to Xvfb"
  exit 0
fi
apt-get update -qq && apt-get install -y -qq --no-install-recommends \
  xserver-xorg-core xauth x11-utils >/dev/null
command -v nvidia-ctk >/dev/null || {
  echo "nvidia-ctk missing: install the NVIDIA container toolkit on this host" >&2
  exit 3
}
if ! docker info 2>/dev/null | grep -qF nvidia; then
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

mkdir -p /opt/ce/x11 && chmod 700 /opt/ce/x11
auth_file=/opt/ce/x11/Xauthority
display=:99

# Substitute the GPU's PCI BusID into the Xorg config (Xorg wants decimal).
pci="$(nvidia-smi --id=0 --query-gpu=pci.bus_id --format=csv,noheader | tr -d '[:space:]')"
if [[ "$pci" =~ ^([0-9A-Fa-f]{8}):([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})\.([0-9A-Fa-f])$ ]]; then
  domain=$((16#${BASH_REMATCH[1]})); bus=$((16#${BASH_REMATCH[2]}))
  dev=$((16#${BASH_REMATCH[3]})); fn=$((16#${BASH_REMATCH[4]}))
  if (( domain == 0 )); then bus_id="PCI:${bus}:${dev}:${fn}"
  else bus_id="PCI:${bus}@${domain}:${dev}:${fn}"; fi
else
  echo "could not parse NVIDIA PCI address: $pci" >&2
  exit 4
fi
sed "s|__NVIDIA_BUS_ID__|${bus_id}|g" /opt/ce/xorg-headless.conf > /etc/X11/xorg.conf

if [[ -f /opt/ce/x11/xorg.pid ]] && kill -0 "$(cat /opt/ce/x11/xorg.pid)" 2>/dev/null \
   && DISPLAY="$display" XAUTHORITY="$auth_file" xdpyinfo >/dev/null 2>&1; then
  echo "    Xorg already up on ${display}"
  exit 0
fi

rm -f "$auth_file"; touch "$auth_file"; chmod 600 "$auth_file"
IFS= read -r cookie_uuid < /proc/sys/kernel/random/uuid
xauth -f "$auth_file" add "$display" MIT-MAGIC-COOKIE-1 "${cookie_uuid//-/}"
# Containers have a different hostname: a FamilyWild copy of the cookie lets
# their Unix-socket clients authenticate against the same display.
xauth -f "$auth_file" nlist "$display" | sed -e 's/^..../ffff/' \
  | xauth -f "$auth_file" nmerge -

rm -f /tmp/.X99-lock
Xorg "$display" -noreset -nolisten tcp \
  +extension GLX +extension RANDR +extension RENDER \
  -auth "$auth_file" -config /etc/X11/xorg.conf \
  > /opt/ce/x11/xorg-launch.log 2>&1 &
echo $! > /opt/ce/x11/xorg.pid
for _ in $(seq 1 120); do
  kill -0 "$(cat /opt/ce/x11/xorg.pid)" 2>/dev/null || {
    echo "Xorg exited during startup — see /opt/ce/x11/xorg-launch.log" >&2
    exit 5
  }
  DISPLAY="$display" XAUTHORITY="$auth_file" xdpyinfo >/dev/null 2>&1 && break
  sleep 0.25
done
DISPLAY="$display" XAUTHORITY="$auth_file" xdpyinfo >/dev/null
echo "    Xorg ready on ${display} (auth ${auth_file})"
REMOTE
fi

PROFILE=""
if [[ "${NEEDS_RETRIEVER}" == "yes" ]]; then
  PROFILE="--profile ragqa"
  # The sidecar mounts /data/ragqa read-only (compose.host.yml), so its own
  # first-boot download cannot run; the host fetches the two files itself,
  # straight from Hugging Face (both public, no token). ~4.5 GB per host, a
  # few minutes on the droplet's uplink. Sizes are pinned so a truncated or
  # silently-changed file fails here rather than as a wrong retrieval result
  # hours into a run.
  echo "==> ragqa retriever data (/data/ragqa; skipped when already present)"
  ssh "${ADDR}" bash -s <<'REMOTE'
set -euo pipefail
mkdir -p /data/ragqa
fetch() {  # url expected_bytes
  local url="$1" want="$2" f="/data/ragqa/$(basename "$1")"
  if [[ -f "$f" && "$(stat -c %s "$f")" == "$want" ]]; then
    echo "    $(basename "$f") present"; return
  fi
  echo "    downloading $(basename "$f") ($((want / 1000000)) MB)"
  curl -fsSL --retry 5 --retry-delay 10 -o "$f.part" "$url" && mv "$f.part" "$f"
  [[ "$(stat -c %s "$f")" == "$want" ]] || { echo "size mismatch: $f" >&2; exit 6; }
}
fetch https://huggingface.co/dspy/cache/resolve/main/index.pt 3923000402
fetch https://huggingface.co/datasets/colbertv2/lotte_passages/resolve/main/technology/test_collection.jsonl 584436206
REMOTE
  echo "==> ragqa retriever enabled on this host"
else
  echo "==> no retriever needed for this benchmark subset"
fi
ssh "${ADDR}" "cd /opt/ce && REGISTRY=${REGISTRY} OPENAI_API_KEY=${OPENAI_API_KEY:-} \
  docker compose -f compose.host.yml ${PROFILE} up -d --build"

echo "==> pulling runner images for the selected subset only"
for bench in ${BENCHES}; do
  docker --context "${HOST_NAME}" pull "${REGISTRY}/codeevolver-runner:${bench}-${TAG}" || \
    echo "    (skip ${bench} — not pushed yet?)"
done

echo "==> ${HOST_NAME} ready"
docker --context "${HOST_NAME}" ps
