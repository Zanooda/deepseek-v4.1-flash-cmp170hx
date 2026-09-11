#!/bin/bash
# Restart the V4.1 engine with a layer partition (and optional device order),
# wait for health, print the KV pool and per-rank memory, then run the 128k
# c=1 matrix cell and the 128k c=8 prefill-free decode window.
# usage: v41_partition_trial.sh <partition> <tag> [gpus]
set -u
P="$1"; TAG="$2"; GPUS="${3:-0,1,2,3,4,5,7,6}"
cd "$(dirname "$0")/.."
docker stop -t 30 dsv41 >/dev/null 2>&1
sleep 3
VLLM_PP_LAYER_PARTITION="$P" DSV41_GPUS="$GPUS" DSV41_MAX_SEQS="${DSV41_MAX_SEQS:-8}" DSV41_VLLM_SRC="${DSV41_VLLM_SRC:-$HOME/vllm-v41/vllm}" bash launch/run-v41-pp8.sh 2>&1 | tail -1
for i in $(seq 1 200); do
  curl -sf -m 3 http://127.0.0.1:8099/health >/dev/null 2>&1 && break
  docker ps --format '{{.Names}}' | grep -q '^dsv41$' || { echo "CONTAINER_DIED"; docker logs dsv41 2>&1 | grep -E "Error|Traceback" | grep -v "amdsmi\|_rocm_C\|vllm._C'" | tail -5; exit 1; }
  sleep 6
done
curl -sf -m 3 http://127.0.0.1:8099/health >/dev/null 2>&1 || { echo "TIMEOUT waiting for health"; exit 1; }
echo "=== partition $P gpus $GPUS ready"
docker logs dsv41 2>&1 | grep -E "GPU KV cache size" | tail -1 | sed 's/.*GPU KV/GPU KV/'
docker logs dsv41 2>&1 | grep "gpu_worker.py:867" | sed -E 's/.*\(Worker_(PP[0-9]+).*Actual usage is ([0-9.]+) GiB.*in use is ([0-9.]+) GiB.*/\1 consumed \2 GiB, kv \3 GiB/' | sort -V
PWLOG=$(mktemp)
nvidia-smi --query-gpu=index,power.draw --format=csv,noheader,nounits -l 1 > "$PWLOG" 2>/dev/null &
PWPID=$!
python3 bench/bench_v41_matrix.py --ctx 128000 --conc 1 --out bench/bench_v41_matrix_v3.jsonl 2>&1 | tail -1
python3 bench/bench_v41_decode_window.py --ctx 128000 --conc 8 --gen 3000 --tag "$TAG"
kill $PWPID 2>/dev/null
echo "power draw per GPU during the benches (W): max / mean"
awk -F', ' '{ if ($2>mx[$1]) mx[$1]=$2; sum[$1]+=$2; n[$1]++ } END { for (g in mx) printf "  GPU %s: %.0f / %.0f\n", g, mx[g], sum[g]/n[g] }' "$PWLOG" | sort -V
rm -f "$PWLOG"
