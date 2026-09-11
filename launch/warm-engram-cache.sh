#!/bin/bash
# Pull the two Engram shards (101.5 GB each) into the Linux page cache so the
# storage=disk gather reads them from RAM. Run AFTER the engine is healthy:
# loading the other 46 shards streams ~270 GB through the cache and would evict
# them. Needs ~205 GB of free page cache (the box has 247 GB, the engine's
# processes take ~25). The cache is evictable, so this can never OOM; under
# memory pressure it silently degrades back to NVMe reads.
# usage: warm-engram-cache.sh [model-dir]   (waits for :8099/health first)
set -u
MODEL="${1:-/models/DeepSeek-V4.1-Flash}"
PORT="${DSV41_PORT:-8099}"
for i in $(seq 1 240); do curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 5; done
SHARDS=$(grep -lE '"(layers\.1|layers\.14)\.engram\.embed\.weight"' "$MODEL"/model-*.safetensors 2>/dev/null)
[ -z "$SHARDS" ] && SHARDS="$MODEL/model-00047-of-00048.safetensors $MODEL/model-00048-of-00048.safetensors"
t0=$(date +%s)
for f in $SHARDS; do cat "$f" > /dev/null; done
for f in $SHARDS; do fincore "$f" 2>/dev/null | tail -1; done
echo "engram shards warmed in $(( $(date +%s) - t0 )) s; MemAvailable $(awk '/MemAvailable/ {printf "%.0f", $2/1048576}' /proc/meminfo) GiB"
