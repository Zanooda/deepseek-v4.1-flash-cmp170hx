#!/bin/bash
# Pull the two Engram shards (101.5 GB each) into the Linux page cache so the
# storage=disk gather reads them from RAM. Run AFTER the engine is healthy:
# loading the other 46 shards streams ~270 GB through the cache and would evict
# them. Needs ~205 GB of free page cache (the box has 247 GB, the engine's
# processes take ~25). The cache is evictable, so this can never OOM; under
# memory pressure it silently degrades back to NVMe reads.
#
# The read is throttled (64 MiB chunks with a short pause; idle I/O class).
# An unthrottled `cat` saturates NFS, the per-step gathers starve, and after
# ~60 s the engine dies with "No available shared memory broadcast block".
# Gate external traffic on DSV41_WARM_MARKER if the arrival is unpredictable.
# usage: warm-engram-cache.sh [model-dir]   (waits for :8099/health first)
set -u
MODEL="${1:-/models/DeepSeek-V4.1-Flash-1017}"
PORT="${DSV41_PORT:-8099}"
CHUNK_MB="${DSV41_WARM_CHUNK_MB:-64}"
SLEEP="${DSV41_WARM_SLEEP:-0.05}"
MARKER="${DSV41_WARM_MARKER:-/tmp/dsv41-engram-warmed}"
IO_PRIO=""
command -v ionice >/dev/null 2>&1 && IO_PRIO="ionice -c 3"

for i in $(seq 1 240); do curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 5; done
SHARDS=$(grep -lE '"(layers\.1|layers\.14)\.engram\.embed\.weight"' "$MODEL"/model-*.safetensors 2>/dev/null)
[ -z "$SHARDS" ] && SHARDS="$MODEL/model-00047-of-00048.safetensors $MODEL/model-00048-of-00048.safetensors"
rm -f "$MARKER"
t0=$(date +%s)
for f in $SHARDS; do
  size=$(stat -c%s "$f")
  # $IO_PRIO is empty or "ionice -c 3"; the unquoted expansion is intended.
  for ((off = 0; off < size; off += CHUNK_MB * 1024 * 1024)); do
    # shellcheck disable=SC2086
    $IO_PRIO dd if="$f" of=/dev/null bs="${CHUNK_MB}M" count=1 \
      skip=$((off / CHUNK_MB / 1024 / 1024)) iflag=fullblock status=none
    sleep "$SLEEP"
  done
done
for f in $SHARDS; do fincore "$f" 2>/dev/null | tail -1; done
: > "$MARKER"
echo "engram shards warmed in $(( $(date +%s) - t0 )) s; MemAvailable $(awk '/MemAvailable/ {printf "%.0f", $2/1048576}' /proc/meminfo) GiB; marker $MARKER"
