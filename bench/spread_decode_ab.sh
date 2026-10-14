#!/bin/bash
# A/B the VLLM_DSV41_SPREAD_DECODE scheduler knob (patches/v41/0019) on one box.
#
# Arms: 0 (stock) | 1 | 2 | 4  -- the max number of decode requests admitted per
# scheduler step. For each arm the engine is restarted, then the prefill-free
# decode window is measured three times:
#
#   cold c=8   fresh seeds, engine freshly started -> prefills finish at spread
#              times, requests join the ring one at a time (the good case)
#   warm c=8   the same seeds again -> prefix-cache hits, all 8 become runnable
#              in one step (the case the knob is meant to split)
#   warm c=16  same, with more in-flight requests than the 8 PP ring slots
#
# The harness bins its samples by the number of actively generating streams, so
# one run yields the whole n=1..8 (or 1..16) curve, including the single-stream
# control. Compare the `warm c=8` rows: stock measured 274 tok/s aggregate there
# and 531 in the cold rows (RESULTS.md section 4).
#
# Usage: bench/spread_decode_ab.sh [arm ...]      (default: 0 1 2 4)
# Cost: ~10 min engine start + ~6 min of measurement per arm.
set -u
cd "$(dirname "$0")/.."
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(0 1 2 4)
OUT=bench/bench_v41_decode_window.jsonl

start_engine() {   # $1 = spread value
  docker stop -t 30 dsv41 >/dev/null 2>&1
  sleep 5
  DSV41_SPREAD_DECODE="$1" DSV41_VLLM_SRC="${DSV41_VLLM_SRC:-$HOME/vllm-v41/vllm}" \
    bash launch/run-v41-pp8.sh >/dev/null 2>&1 || return 1
  for _ in $(seq 1 240); do
    curl -sf -m 3 http://127.0.0.1:8099/health >/dev/null 2>&1 && { echo "   up (spread=$1)"; return 0; }
    docker ps --format '{{.Names}}' | grep -q '^dsv41$' || { echo "   CONTAINER_DIED (spread=$1)"; return 1; }
    sleep 6
  done
  echo "   TIMEOUT waiting for health (spread=$1)"; return 1
}

for arm in "${ARMS[@]}"; do
  echo "=== arm spread=$arm  $(date -u +%H:%M:%S)"
  start_engine "$arm" || continue
  seed=$((7000 + arm * 1000))
  python3 bench/bench_v41_decode_window.py --ctx 128000 --conc 8  --gen 3000 \
      --seed "$seed" --tag "spread${arm}-cold-c8"
  python3 bench/bench_v41_decode_window.py --ctx 128000 --conc 8  --gen 3000 \
      --seed "$seed" --tag "spread${arm}-warm-c8"
  python3 bench/bench_v41_decode_window.py --ctx 128000 --conc 16 --gen 3000 \
      --seed "$seed" --tag "spread${arm}-cold-c16"
  python3 bench/bench_v41_decode_window.py --ctx 128000 --conc 16 --gen 3000 \
      --seed "$seed" --tag "spread${arm}-warm-c16"
done

echo
echo "=== aggregate / per-stream tok/s per arm (n = active streams, warm runs)"
python3 - "$OUT" <<'EOF'
import json, re, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rows = [r for r in rows if r.get("tag", "").startswith("spread")]
def cell(r, n):
    d = (r.get("by_active") or {}).get(str(n))
    return f"{d['aggregate']:6.1f} /{d['per_request']:5.1f}" if d else "      - /    -"
def sort_key(t):
    m = re.match(r"spread(\d+)-(cold|warm)-c(\d+)", t)
    return (int(m.group(1)), m.group(2) != "warm", int(m.group(3)))
tags = sorted({r["tag"] for r in rows}, key=sort_key)
ns = [1, 4, 8, 16]
print("tag".ljust(24) + "".join(f"  n={n:<12}" for n in ns))
for t in tags:
    row = [r for r in rows if r["tag"] == t][-1]
    print(t.ljust(24) + "".join(" " + cell(row, n) for n in ns))
print("\n(aggregate / per-stream; - = that many streams were never active together)")
EOF
