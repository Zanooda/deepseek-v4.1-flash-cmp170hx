#!/usr/bin/env bash
# Build the DeepSeek-V4.1 sm_80 vLLM image (branch "v41-sm80": vLLM PR #56201
# head + patches/v41/) on a fresh instance (e.g. vast.ai) and push it.
#
# NOT self-contained: copy the whole deepseek-v4-cmp170hx directory to the
# instance and run docker/vastai-build-push-v41.sh from inside it.
#
# Instance requirements are the same as vastai-build-push.sh: >= 48 GB RAM
# (this build OOM-kills a 30 GB box), >= 250 GB NVMe, no GPU needed
# (TORCH_CUDA_ARCH_LIST=8.0), docker or podman.
#
# Base: upstream vLLM PR #56201 ("[Model] Support DeepSeek-V4.1-Flash"), head
# commit 79a7108d9aea27ddab99ce1779290d300b17fc23, tree
# a731238e73f0023b0a43f6d91c07c466295224e6. It is an OPEN PR and may be
# force-pushed; if the SHA becomes unreachable, GitHub still serves it by
# tarball (codeload.github.com/vllm-project/vllm/tar.gz/<sha>) -- see
# patches/README.md for the tree-SHA verification that makes that safe.
#
# Usage:
#   export REGISTRY_PASS=...
#   docker/vastai-build-push-v41.sh
#   REGISTRY_IMAGE=ghcr.io/me/vllm-sm80:v41-sm80 MAX_JOBS=32 docker/vastai-build-push-v41.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO_ROOT/docker/Dockerfile.fullbuild" ] || {
  echo "FATAL: $REPO_ROOT does not look like the deepseek-v4-cmp170hx repo" >&2
  exit 1
}

REGISTRY="${REGISTRY:-docker.io}"
REGISTRY_IMAGE="${REGISTRY_IMAGE:-$REGISTRY/zanooda/vllm-sm80-ds41f:v41-sm80}"
REGISTRY_USER="${REGISTRY_USER:-zanooda}"
MAX_JOBS="${MAX_JOBS:-}"
BASE_SHA="79a7108d9aea27ddab99ce1779290d300b17fc23"
EXPECTED_TREE="a731238e73f0023b0a43f6d91c07c466295224e6"
# Byte-exact tree the series produces on top of the base (patches 0001-0009;
# the published image zanooda/vllm-sm80-ds41f:v41-sm80 was built from the
# 0001-0008 tree 70aaaced9d and runs 0009 via bind mounts). git am is checked
# against it below so a silently fuzzed apply cannot go unnoticed).
EXPECTED_PATCHED_TREE="${EXPECTED_PATCHED_TREE:-96c3fc9bc69948672cffe8f2a7bb4277735ee4b5}"
WORK="${WORK:-/root/dsv41-build}"

echo "==> target image: $REGISTRY_IMAGE"
echo "==> workdir: $WORK"

if ! command -v git >/dev/null || ! command -v curl >/dev/null; then
  apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates
fi

BUILDER=""
if command -v docker >/dev/null 2>&1; then
  if ! docker info >/dev/null 2>&1; then
    dockerd >/var/log/dockerd.log 2>&1 &
    for i in $(seq 1 30); do docker info >/dev/null 2>&1 && break; sleep 2; done
  fi
  docker info >/dev/null 2>&1 && BUILDER="docker"
fi
if [ -z "$BUILDER" ]; then
  apt-get update -qq && apt-get install -y -qq --no-install-recommends podman
  BUILDER="podman --storage-driver=vfs"
fi
echo "==> builder: $BUILDER"

mkdir -p "$WORK" && cd "$WORK"
if [ ! -d vllm-src/.git ]; then
  echo "==> cloning vllm and fetching the PR head"
  git clone -q https://github.com/vllm-project/vllm.git vllm-src
  cd vllm-src
  # Try the PR ref first (works while the PR is open), then the SHA directly.
  git fetch -q origin "refs/pull/56201/head" || true
  git fetch -q origin "$BASE_SHA" || true
  if ! git cat-file -e "$BASE_SHA^{commit}" 2>/dev/null; then
    echo "==> $BASE_SHA unreachable via git; reconstructing from the tarball"
    curl -sL -o ../base.tar.gz "https://codeload.github.com/vllm-project/vllm/tar.gz/$BASE_SHA"
    mkdir -p ../base-src && tar xzf ../base.tar.gz -C ../base-src --strip-components=1
    export GIT_INDEX_FILE="$WORK/base.index"
    git read-tree --empty && git --work-tree=../base-src add -Af
    TREE=$(git write-tree)
    [ "$TREE" = "$EXPECTED_TREE" ] || { echo "FATAL: tree $TREE != $EXPECTED_TREE" >&2; exit 1; }
    git tag base-recon "$(git commit-tree "$TREE" -m 'PR #56201 head reconstructed from tarball')"
    unset GIT_INDEX_FILE
    git checkout -q -B v41-sm80 base-recon
  else
    git checkout -q -B v41-sm80 "$BASE_SHA"
  fi
  cd ..
fi
cd "$WORK/vllm-src"
TREE=$(git rev-parse "HEAD^{tree}")
[ "$TREE" = "$EXPECTED_TREE" ] || { echo "FATAL: base tree $TREE != $EXPECTED_TREE" >&2; exit 1; }
echo "==> base tree verified: $TREE"

echo "==> applying patches/v41 (git am)"
git -c user.name=build -c user.email=build@localhost am -q "$REPO_ROOT"/patches/v41/*.patch
TREE=$(git rev-parse "HEAD^{tree}")
[ "$TREE" = "$EXPECTED_PATCHED_TREE" ] || { echo "FATAL: patched tree $TREE != $EXPECTED_PATCHED_TREE" >&2; exit 1; }
echo "==> patched tree verified: $TREE"

DOCKERFILE="$REPO_ROOT/docker/Dockerfile.fullbuild"
if [ -n "$MAX_JOBS" ]; then
  sed -i "s/^ENV MAX_JOBS=.*/ENV MAX_JOBS=$MAX_JOBS/" "$DOCKERFILE"
fi
echo "==> building (hours; log: $WORK/build.log)"
$BUILDER build -f "$DOCKERFILE" -t "$REGISTRY_IMAGE" "$WORK/vllm-src" 2>&1 | tee "$WORK/build.log"

if [ -z "${REGISTRY_PASS:-}" ]; then
  echo "FATAL: REGISTRY_PASS is not set; image stays local: $REGISTRY_IMAGE" >&2
  exit 1
fi
echo "$REGISTRY_PASS" | $BUILDER login "$REGISTRY" -u "$REGISTRY_USER" --password-stdin
$BUILDER push "$REGISTRY_IMAGE"
echo "==> DONE: $REGISTRY_IMAGE"
