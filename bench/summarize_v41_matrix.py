#!/usr/bin/env python3
"""Turn bench_v41_matrix.jsonl into the markdown tables used in RESULTS.md."""
import json
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "bench_v41_matrix.jsonl"
cells = [json.loads(line) for line in open(path) if line.strip()]
by = defaultdict(dict)
for c in cells:
    by[c["ctx"]][c["conc"]] = c
concs = sorted({c["conc"] for c in cells})


def fmt(v, nd=0):
    if v is None:
        return "—"
    return f"{v:,.{nd}f}"


print("### Prefill (tok/s; per request = prompt tokens / TTFT, aggregate = all prompts / max TTFT)\n")
print("| real prompt tok | " + " | ".join(f"c={c} per-req" for c in concs) + " | " + " | ".join(f"c={c} aggregate" for c in concs) + " |")
print("|---|" + "---|" * (2 * len(concs)))
for ctx in sorted(by):
    row = by[ctx]
    ref = next(iter(row.values()))
    ptok = ref.get("prompt_tokens_each", [None])[0]
    per = [fmt(row[c]["prefill_tps_per_req"]) if c in row and "error" not in row[c] else "fail" for c in concs]
    agg = [fmt(row[c]["prefill_tps_aggregate"]) if c in row and "error" not in row[c] else "fail" for c in concs]
    print(f"| {fmt(ptok)} | " + " | ".join(per) + " | " + " | ".join(agg) + " |")

print("\n### Time to first token (s, worst request in the cell)\n")
print("| real prompt tok | " + " | ".join(f"c={c}" for c in concs) + " |")
print("|---|" + "---|" * len(concs))
for ctx in sorted(by):
    row = by[ctx]
    ptok = next(iter(row.values())).get("prompt_tokens_each", [None])[0]
    print(f"| {fmt(ptok)} | " + " | ".join(fmt(row[c]["ttft_max"], 1) if c in row and "error" not in row[c] else "fail" for c in concs) + " |")

print("\n### Decode (tok/s; per request = generated tokens / (last − first token), aggregate = sum over requests)\n")
print("| real prompt tok | " + " | ".join(f"c={c} per-req" for c in concs) + " | " + " | ".join(f"c={c} aggregate" for c in concs) + " |")
print("|---|" + "---|" * (2 * len(concs)))
for ctx in sorted(by):
    row = by[ctx]
    ptok = next(iter(row.values())).get("prompt_tokens_each", [None])[0]
    per = [fmt(row[c]["decode_tps_per_req"], 1) if c in row and "error" not in row[c] else "fail" for c in concs]
    agg = [fmt(row[c]["decode_tps_aggregate"], 1) if c in row and "error" not in row[c] else "fail" for c in concs]
    print(f"| {fmt(ptok)} | " + " | ".join(per) + " | " + " | ".join(agg) + " |")
