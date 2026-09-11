#!/usr/bin/env python3
"""Steady-state concurrent decode at long context, free of prefill interference.

bench_v41_matrix.py's "decode" at c>1 and >=128k is mostly measured while the
other requests' 4096-token prefill chunks share every step (each request's
prefill takes ~20 s at 128k), so it does not describe c requests decoding
together. This fires c long prompts with a long generation, records the
arrival time of every stream chunk, and then bins time into 1 s slots: a slot
counts only if no request is still waiting for its first token (no prefill in
flight) and exactly n requests are mid-generation. It reports aggregate and
per-request tok/s for every n that occurred (n = c while all are alive, then
c-1, ... as requests finish). Chunk counts are scaled to real tokens with the
request's usage total (DSpark emits several tokens per chunk).

Usage: bench_v41_decode_window.py [--ctx 128000] [--conc 8] [--gen 3000]
"""
import argparse
import json
import random
import threading
import time
import urllib.request
from collections import defaultdict

WORDS = (
    "system kernel memory buffer thread process socket packet register cache "
    "pointer allocate schedule interrupt virtual physical address translate "
    "compile execute branch predict pipeline vector matrix tensor gradient "
    "cluster network storage device driver module segment offset boundary "
    "harbor lantern meadow copper violin orchard glacier saffron thimble walnut"
).split()


def make_prompt(approx_tokens, seed):
    rng = random.Random(seed)
    body = " ".join(rng.choice(WORDS) for _ in range(max(8, int(approx_tokens / 1.3))))
    return f"[run {seed}] Notes: {body}\nWrite a very long, detailed summary of the notes above."


def stream(url, model, prompt, gen, timeout, out):
    payload = {"model": model, "prompt": prompt, "max_tokens": gen, "temperature": 0,
               "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    stamps, usage = [], None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            if any(ch.get("text") for ch in obj.get("choices", [])):
                stamps.append(time.perf_counter())
    out.update({"stamps": stamps, "usage": usage or {}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8099/v1/completions")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--ctx", type=int, default=128000)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--gen", type=int, default=3000)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--out", default="bench/bench_v41_decode_window.jsonl")
    ap.add_argument("--tag", default="")
    ap.add_argument("--stagger", type=float, default=0.0, help="seconds between request starts")
    args = ap.parse_args()

    outs = [{} for _ in range(args.conc)]
    ths = [threading.Thread(target=stream, args=(args.url, args.model, make_prompt(args.ctx, 7000 + i),
                                                  args.gen, args.timeout, outs[i])) for i in range(args.conc)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
        if args.stagger:
            time.sleep(args.stagger)
    [t.join() for t in ths]
    wall = time.perf_counter() - t0

    # per request: first/last stamps, tokens per chunk
    reqs = []
    for o in outs:
        st = o.get("stamps") or []
        if not st:
            continue
        ctoks = o["usage"].get("completion_tokens") or len(st)
        reqs.append({"first": st[0], "last": st[-1], "stamps": st, "tok_per_chunk": ctoks / len(st),
                     "prompt_tokens": o["usage"].get("prompt_tokens"), "completion_tokens": ctoks})
    last_first = max(r["first"] for r in reqs)
    # 1 s slots from the last first-token onward
    slot_tokens = defaultdict(float)
    for r in reqs:
        for s in r["stamps"]:
            if s > last_first:
                slot_tokens[int(s - last_first)] += r["tok_per_chunk"]
    by_n = defaultdict(lambda: [0.0, 0])  # n active -> [tokens, slots]
    max_slot = int(max(r["last"] for r in reqs) - last_first)
    for k in range(0, max_slot):
        t_mid = last_first + k + 0.5
        n = sum(1 for r in reqs if r["first"] <= t_mid <= r["last"])
        if n == 0:
            continue
        by_n[n][0] += slot_tokens.get(k, 0.0)
        by_n[n][1] += 1
    print(f"ctx~{args.ctx} c={args.conc} gen={args.gen}: prompt tokens {reqs[0]['prompt_tokens']}, "
          f"all prefills done at {last_first - t0:.0f} s, wall {wall:.0f} s")
    print(f"{'active n':>8} {'seconds':>8} {'aggregate tok/s':>15} {'per-request tok/s':>17}")
    rows = {}
    for n in sorted(by_n, reverse=True):
        toks, slots = by_n[n]
        if slots < 5:
            continue
        agg = toks / slots
        rows[n] = {"seconds": slots, "aggregate": round(agg, 1), "per_request": round(agg / n, 1)}
        print(f"{n:>8} {slots:>8} {agg:>15.1f} {agg / n:>17.1f}")
    with open(args.out, "a") as f:
        f.write(json.dumps({"ctx": args.ctx, "conc": args.conc, "gen": args.gen, "tag": args.tag, "stagger": args.stagger,
                            "prompt_tokens": reqs[0]["prompt_tokens"], "by_active": rows,
                            "ts": time.time()}) + "\n")


if __name__ == "__main__":
    main()
