#!/usr/bin/env bash
# Build the custom DeepSeek-V4 sm_80 vLLM image on a fresh instance (e.g. vast.ai)
# and push it to a container registry.
#
# NOT self-contained: copy the whole deepseek-v4-cmp170hx directory to the
# instance and run docker/vastai-build-push.sh from inside it. The script
# locates patches/ and docker/Dockerfile.fullbuild relative to itself.
#
# Instance requirements:
#   - RAM: >= 48 GB recommended (nvcc/cicc jobs take 2-4 GB each; MAX_JOBS
#     below). A 30 GB box OOM-killed this build even at MAX_JOBS=16.
#   - Disk: >= 250 GB NVMe (CUDA base image + torch + build tree + ~30 GB
#     output image + push staging).
#   - No GPU needed -- the build is CPU-only (TORCH_CUDA_ARCH_LIST=8.0).
#   - Docker working (privileged instance) OR podman as fallback (vfs driver,
#     works in unprivileged containers).
#
# Usage:
#   export REGISTRY_PASS=...                          # Docker Hub password/token
#   docker/vastai-build-push.sh                       # builds + pushes default tag
#   REGISTRY_IMAGE=ghcr.io/me/vllm-sm80:c3046d1 MAX_JOBS=32 docker/vastai-build-push.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/docker/Dockerfile.fullbuild" ] || {
  echo "FATAL: $REPO_ROOT does not look like the deepseek-v4-cmp170hx repo" >&2
  exit 1
}

# ---- configure ----------------------------------------------------------------
REGISTRY="${REGISTRY:-docker.io}"
REGISTRY_IMAGE="${REGISTRY_IMAGE:-$REGISTRY/zanooda/vllm-sm80:c3046d1}"
REGISTRY_USER="${REGISTRY_USER:-zanooda}"
# REGISTRY_PASS must come from the environment -- never hardcode it here;
# this repo is public on GitHub.
MAX_JOBS="${MAX_JOBS:-}"          # empty = keep the Dockerfile's value (8)
BASE_SHA="c3046d1ebd2dae9b94ad2ef5f966ea153632251e"
BASE_TARBALL="https://codeload.github.com/haosdent/vllm/tar.gz/$BASE_SHA"
# Expected `git write-tree` SHA of the pristine c3046d1 tree (patches/README.md).
EXPECTED_TREE="d13ae12b9a6621ef8d218f53741e59c6db2f68d2"
PATCHES="0002-speculative 0003-pp_utils 0004-model_runner 0005-dspark-utils 0005a-prefill-topk-torch-fallback 0006-logits-row-chunk 0007-dsml-malformed-wrapper-recovery 0008-rejection-sampler-nan-guard"
WORK="${WORK:-/root/dsv4-build}"
# --------------------------------------------------------------------------------

echo "==> target image: $REGISTRY_IMAGE"
echo "==> workdir: $WORK"

# ---- 0. base tooling -----------------------------------------------------------
if ! command -v git >/dev/null || ! command -v curl >/dev/null; then
  echo "==> installing git/curl"
  apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates
fi

# ---- 1. container builder: docker, else podman --------------------------------
BUILDER=""
if command -v docker >/dev/null 2>&1; then
  if ! docker info >/dev/null 2>&1; then
    echo "==> starting dockerd"
    dockerd >/var/log/dockerd.log 2>&1 &
    for i in $(seq 1 30); do docker info >/dev/null 2>&1 && break; sleep 2; done
  fi
  docker info >/dev/null 2>&1 && BUILDER="docker"
fi
if [ -z "$BUILDER" ]; then
  echo "==> docker unavailable, falling back to podman (vfs)"
  apt-get update -qq && apt-get install -y -qq --no-install-recommends podman
  BUILDER="podman --storage-driver=vfs"
fi
echo "==> builder: $BUILDER"

# ---- 2. fetch sources ----------------------------------------------------------
mkdir -p "$WORK" && cd "$WORK"

if [ ! -d vllm-src ]; then
  echo "==> downloading c3046d1 tarball"
  curl -sL -o c3046d1.tar.gz "$BASE_TARBALL"
  mkdir vllm-src && tar xzf c3046d1.tar.gz -C vllm-src --strip-components=1
fi

# ---- 3. verify the tree is byte-identical to upstream c3046d1 ------------------
# Tree SHA depends only on content, so a scratch repo is enough -- no clone
# history needed. `add -Af` (not -A): upstream tracks files that match
# .gitignore patterns, and without -f they are silently skipped -> wrong SHA.
echo "==> verifying tree SHA"
git init -q "$WORK/verify"
git --git-dir="$WORK/verify/.git" --work-tree="$WORK/vllm-src" add -Af
TREE=$(git --git-dir="$WORK/verify/.git" write-tree)
rm -rf "$WORK/verify"
if [ "$TREE" != "$EXPECTED_TREE" ]; then
  echo "FATAL: tree SHA mismatch: got $TREE, expected $EXPECTED_TREE" >&2
  echo "       do NOT build from an unverified tree" >&2
  exit 1
fi
echo "==> tree verified: $TREE"

# ---- 4. apply patches (0001 dropped: gate is upstream in c3046d1) --------------
cd "$WORK/vllm-src"
for p in $PATCHES; do
  echo "==> patch $p"
  patch -p1 --no-backup-if-mismatch -s < "$REPO_ROOT/patches/$p.patch"
done
echo "==> all patches applied, zero rejects"

# ---- 5. build ------------------------------------------------------------------
DOCKERFILE="$REPO_ROOT/docker/Dockerfile.fullbuild"
if [ -n "$MAX_JOBS" ]; then
  echo "==> overriding MAX_JOBS=$MAX_JOBS"
  sed -i "s/^ENV MAX_JOBS=.*/ENV MAX_JOBS=$MAX_JOBS/" "$DOCKERFILE"
fi

echo "==> building (this takes hours; log: $WORK/build.log)"
$BUILDER build -f "$DOCKERFILE" -t "$REGISTRY_IMAGE" "$WORK/vllm-src" 2>&1 | tee "$WORK/build.log"

# ---- 6. push -------------------------------------------------------------------
if [ -z "${REGISTRY_PASS:-}" ]; then
  echo "FATAL: REGISTRY_PASS is not set -- export it first (see header)." >&2
  echo "       the image remains built locally: $REGISTRY_IMAGE" >&2
  exit 1
fi
echo "==> logging into $REGISTRY as $REGISTRY_USER"
echo "$REGISTRY_PASS" | $BUILDER login "$REGISTRY" -u "$REGISTRY_USER" --password-stdin
echo "==> pushing $REGISTRY_IMAGE"
$BUILDER push "$REGISTRY_IMAGE"

echo "==> DONE: $REGISTRY_IMAGE"
