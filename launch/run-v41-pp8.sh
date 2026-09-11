#!/bin/bash
# DeepSeek-V4.1-Flash on 8x CMP 170HX (GA100, sm_80, 64 GB each), PP=8.
#
# STATUS: RUNNING since 2026-09-10 (RESULTS.md#deepseek-v41-flash-on-8-cmp-170hx).
# Image zanooda/vllm-sm80-ds41f:v41-sm80 (tree 70aaaced9d) plus patches 0009-0013
# bind-mounted from a v41-sm80 checkout (DSV41_VLLM_SRC); rebuild to bake them in.
#
# Usage: run-v41-pp8.sh [--plain] [--maxlen N] [--engram gpu|cpu|disk]
#
# Why these settings (details in SETTINGS.md#deepseek-v41-flash):
#   PP=8, no TP        PCIe Gen2 x4: TP loses 6.6x on prefill (measured on V4). With the
#                      cmpunlocker BAR1 P2P enabled, NCCL_P2P_LEVEL=SYS (passed through)
#                      routes the hops GPU-to-GPU; measured: no speed change, link-bound.
#   engram cpu         The two Engram tables (98 GiB fp8 + 3 GiB scales each) are
#                      pinned in host RAM on the ranks owning layers 1 and 14 and
#                      gathered by the GPU over UVA. Patch 0013 makes that fit:
#                      exact-size cudaHostRegister (torch's pinned allocator would
#                      round each table to 128 GiB) and a direct chunked fill from
#                      the shard (the generic loader materialised a second 98 GB
#                      copy). Steady state ~217 GB used of 278. Measured vs NVMe:
#                      prefill +25-38 %, decode +10 %. DSV41_ENGRAM=disk keeps the
#                      NVMe gather (with launch/warm-engram-cache.sh for the
#                      page-cache variant) for hosts with less RAM.
#   row chunk 64       DSV4_LOGITS_ROW_CHUNK: the indexer's [rows, N] fp32 logits
#                      transient is 8 GB at 1M tokens un-chunked. Same fix as V4.
#   kv fp8             The main KV is fp8_ds_mla (584 B/state); no FP4 KV on sm_80
#                      and none in the V4.1 checkpoint either (FP4 = experts only).
#   partition 5x8      40 backbone layers. The last rank also carries lm_head and,
#                      with DSpark, the 3-layer drafter (~7.8 GiB). Measured 2026-09-11:
#                      5,5,5,5,5,5,6,4 doubles the KV pool but costs 10 % concurrent
#                      decode (any 6-layer rank is the slowest stage); a last rank
#                      with < 4 layers cannot start (drafter aux states come from
#                      layers 36-38). Keep 5x8.
#   memutil 0.95       The KV pool is set by the last rank (weights + drafter + lm_head
#                      leave it ~3 GiB at 0.90 -> 3.15M tokens). 0.95 gives it ~6.5 GiB:
#                      6,171,394 tokens measured, decode unchanged.
#   usage details      --enable-prompt-tokens-details: cached_tokens in usage for the
#                      proxy's cache-read accounting.
#   images             --limit-mm-per-prompt image=8: V4.1's vision encoder is loaded
#                      (verified: colour/layout of test images through the proxy).
#   concurrency        With PP the scheduler puts every runnable request into ONE
#                      micro-batch. Requests that join one at a time get their own
#                      in-flight batch and overlap across the 8 stages (128k c=8:
#                      532 tok/s aggregate); 8 requests that become runnable in the
#                      same step walk the stages as one batch (274 tok/s). Batches
#                      only merge, never split, so long-running load drifts toward
#                      the lower figure. A scheduler-side fix is a candidate patch.
#   gpus 0,1,2,3,4,5,7,6  Device order = rank order. On the reference box GPU 6 (PCI
#                      c3:00.0) fell off the bus (Xid 79) under sustained load at stock
#                      250 W and runs capped (a systemd unit applies the cap at boot); it
#                      takes the last rank, which draws the least power.
#   max-model-len 1M   The model's max_position_embeddings is 1,048,576: no single
#                      request can be 2M. "2M context" = KV POOL of >= 2M tokens
#                      (e.g. 2 x 1M concurrent), which the fp8 KV (~1.8 kB/token
#                      spread over 8 ranks) gives with room to spare.
# ---- configure -------------------------------------------------------------------
IMG="${DSV41_IMAGE:-zanooda/vllm-sm80-ds41f:v41-sm80}"
MODEL="${DSV41_MODEL:-/models/DeepSeek-V4.1-Flash}"
MAXLEN="${DSV41_MAXLEN:-1048576}"
ROW_CHUNK="${DSV4_ROW_CHUNK:-64}"
ENGRAM="${DSV41_ENGRAM:-cpu}"           # cpu (both tables pinned in host RAM, 2 x 101 GB; needs >= ~240 GB, the box has 278) | disk (+ page-cache warm) | gpu
WARM_ENGRAM="${DSV41_WARM_ENGRAM:-1}"    # after health, read both Engram shards into the page cache (launch/warm-engram-cache.sh)
ENGRAM_THREADS="${DSV41_ENGRAM_THREADS:-32}"
PARTITION="${VLLM_PP_LAYER_PARTITION:-5,5,5,5,5,5,5,5}"
GPUS="${DSV41_GPUS:-0,1,2,3,4,5,7,6}"
PORT="${DSV41_PORT:-8099}"
MEMUTIL="${DSV41_MEMUTIL:-0.95}"
MAXSEQS="${DSV41_MAX_SEQS:-8}"     # max RUNNING requests in total (not per micro-batch); 1 serialises users
# DSpark: block size 5 for this checkpoint (dspark_block_size=5), so 5 tokens
# exactly, as on V4 (7 was slower there; below 5 is rejected). Adds ~7.8 GiB to
# the last rank -- see the partition note in README.
SPEC='--speculative-config {"method":"dspark","num_speculative_tokens":5}'
# Breakable (PIECEWISE) CUDA graphs must be ON for V4.1: the model is not
# torch-compiled (its attention runs in an eager break), so without them the
# runner has no piecewise graphs at all and refuses FULL_AND_PIECEWISE. The
# V4 stack ran with them off; the capture-time indexer bug that motivated
# that (PR #52492) is upstream in this tree. The disk-backed Engram gather
# runs in the eager model-state hook, outside any capture.
BREAKABLE="${DSV4_BREAKABLE_CUDAGRAPH:-1}"

while [ $# -gt 0 ]; do
  case "$1" in
    --plain)  SPEC=""; shift ;;
    --maxlen) MAXLEN="$2"; shift 2 ;;
    --engram) ENGRAM="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

if [ ! -f "$MODEL/model.safetensors.index.json" ]; then
  echo "ERROR: $MODEL is not a local DeepSeek-V4.1-Flash checkpoint." >&2
  echo "       The checkpoint is 510 GB (48 shards; shards 47/48 are the two" >&2
  echo "       101 GB Engram tables). storage=disk needs it on local NVMe." >&2
  exit 1
fi
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  echo "ERROR: image $IMG not found. Pull it (docker pull $IMG) or build it" >&2
  echo "       (docker/vastai-build-push-v41.sh: a full CUDA source build, >= 48 GB RAM)." >&2
  exit 1
fi

docker stop -t 60 dsv41 >/dev/null 2>&1
docker rm dsv41 >/dev/null 2>&1

# The compiled code is baked into the image. To iterate on Python files without
# a rebuild, point DSV41_VLLM_SRC at the vllm/ directory of a v41-sm80 checkout:
# every Python file that differs from the image's commit (IMG_COMMIT) is
# bind-mounted over its /vllm/vllm/... twin (the image installs vLLM with
# `pip install -e .`, so those are live). Only files are mounted, never
# directories, so the compiled .so files in /vllm/vllm stay visible.
# Local commit whose tree (70aaaced9d) is what the image was built from.
IMG_COMMIT="${DSV41_IMG_COMMIT:-68de681be3}"
MOUNTS=""
if [ -n "${DSV41_VLLM_SRC:-}" ]; then
  SRC_ROOT="$(cd "$DSV41_VLLM_SRC/.." && pwd)"
  for f in $(git -C "$SRC_ROOT" diff --name-only "$IMG_COMMIT" -- 'vllm/*.py'; git -C "$SRC_ROOT" ls-files --others --exclude-standard -- 'vllm/*.py'); do
    [ -f "$SRC_ROOT/$f" ] || continue   # deleted in the checkout: cannot unmount a file, skip
    MOUNTS="$MOUNTS -v $SRC_ROOT/$f:/vllm/$f:ro"
  done
  n=$(echo "$MOUNTS" | grep -o ' -v ' | wc -l)
  echo "bind-mounting $n changed python file(s) from $SRC_ROOT over the image"
else
  echo "WARNING: DSV41_VLLM_SRC is not set, so nothing is bind-mounted over the image." >&2
  echo "         The published image contains patches v41/0001-0008 only; 0009-0013 are" >&2
  echo "         required (0009 to start at all). Point DSV41_VLLM_SRC at the vllm/ dir of a" >&2
  echo "         checkout with all patches applied, or use an image rebuilt from the full" >&2
  echo "         series and set DSV41_IMG_COMMIT to its commit. See README, V4.1 quick start." >&2
fi
EXTRA="${DSV41_EXTRA_ARGS:-}"   # e.g. --no-enable-prefix-caching for benchmarks

# shellcheck disable=SC2086
# Pass through any NCCL_* variables set in the caller's environment (e.g.
# NCCL_P2P_LEVEL=SYS once the cmpunlocker BAR1 P2P is enabled, NCCL_DEBUG=INFO
# to see which transport each pipeline hop takes).
NCCL_ENV=""
for _v in $(compgen -e | grep '^NCCL_'); do NCCL_ENV="$NCCL_ENV -e $_v=${!_v}"; done
docker run -d --name dsv41 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$GPUS" $NCCL_ENV \
  -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e DSV4_LOGITS_ROW_CHUNK="$ROW_CHUNK" \
  -e VLLM_PP_LAYER_PARTITION="$PARTITION" \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH="$BREAKABLE" \
  -e PYTHONUNBUFFERED=1 \
  -v "$MODEL":/model \
  $MOUNTS \
  --shm-size=16g -p "$PORT":8000 \
  "$IMG" vllm serve /model --served-model-name dsv41 \
  --pipeline-parallel-size 8 --kv-cache-dtype fp8 --block-size "${DSV41_BLOCK_SIZE:-128}" \
  --max-model-len "$MAXLEN" --max-num-batched-tokens 4096 --trust-remote-code \
  --gpu-memory-utilization "$MEMUTIL" --max-num-seqs "$MAXSEQS" \
  --engram-config "{\"storage\":\"$ENGRAM\",\"disk_threads\":$ENGRAM_THREADS}" \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 \
  --enable-prompt-tokens-details --limit-mm-per-prompt "${DSV41_MM_LIMIT:-{\"image\":8\}}" \
  --no-enable-flashinfer-autotune \
  $SPEC $EXTRA >/dev/null
if [ "$ENGRAM" = "disk" ] && [ "$WARM_ENGRAM" = "1" ]; then
  nohup "$(dirname "$0")/warm-engram-cache.sh" "$MODEL" > /tmp/dsv41-warm-engram.log 2>&1 &
fi
echo "launched dsv41 on :$PORT  (maxlen $MAXLEN, engram $ENGRAM, partition $PARTITION, spec: ${SPEC:-none}, extra: ${EXTRA:-none})"
