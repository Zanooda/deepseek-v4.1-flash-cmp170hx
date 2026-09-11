#!/usr/bin/env python3
"""Prefill/decode matrix for DeepSeek-V4.1 on the sm_80 stack: context x concurrency.

For every (context length, concurrency) cell it fires `c` simultaneous streaming
completions with UNIQUE prompts of ~ctx tokens, and reports per request:

  prefill  = server-reported prompt_tokens / time-to-first-token
  decode   = server-reported completion_tokens / (last token - first token)

and aggregate (sum of tokens / wall) for the cell. Uses the pitfalls list in
bench/README.md: token counts come from `usage` with `include_usage`, never from
SSE chunk counts; `ignore_eos` keeps generations the same length; a warm-up
request is discarded; run the server with --no-enable-prefix-caching.

Usage: bench_v41_matrix.py [--url URL] [--model M] [--ctx 1000,8000,...]
                           [--conc 1,2,4,8] [--gen 128] [--out results.jsonl]
"""
import argparse
import json
import random
import sys
import threading
import time
import urllib.request

WORDS = (
    "system kernel memory buffer thread process socket packet register cache "
    "pointer allocate schedule interrupt virtual physical address translate "
    "compile execute branch predict pipeline vector matrix tensor gradient "
    "cluster network storage device driver module segment offset boundary "
    "harbor lantern meadow copper violin orchard glacier saffron thimble walnut"
).split()


def make_prompt(approx_tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    n_words = max(8, int(approx_tokens / 1.3))
    body = " ".join(rng.choice(WORDS) for _ in range(n_words))
    return f"[run {seed}] Notes: {body}\nWrite a short summary of the notes above."


def one_request(url, model, prompt, gen, timeout):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": gen,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    t_first = t_last = None
    usage = None
    text = []
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
            for ch in obj.get("choices", []):
                piece = ch.get("text", "")
                if piece:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    text.append(piece)
    t_end = time.perf_counter()
    if t_first is None:
        t_first = t_end
    if t_last is None:
        t_last = t_first
    return {
        "ttft": t_first - t0,
        "decode_span": t_last - t_first,
        "wall": t_end - t0,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "text": "".join(text)[:160],
    }


def run_cell(url, model, ctx, conc, gen, seed_base, timeout):
    prompts = [make_prompt(ctx, seed_base + i) for i in range(conc)]
    results = [None] * conc

    def work(i):
        try:
            results[i] = one_request(url, model, prompts[i], gen, timeout)
        except Exception as e:  # noqa: BLE001
            results[i] = {"error": str(e)[:200]}

    threads = [threading.Thread(target=work, args=(i,)) for i in range(conc)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    ok = [r for r in results if r and "error" not in r and r.get("prompt_tokens")]
    errs = [r["error"] for r in results if r and "error" in r]
    if not ok:
        return {"ctx": ctx, "conc": conc, "error": errs[:1], "wall": wall}
    ptok = sum(r["prompt_tokens"] for r in ok)
    gtok = sum(r["completion_tokens"] or 0 for r in ok)
    per_prefill = [r["prompt_tokens"] / r["ttft"] for r in ok if r["ttft"] > 0]
    per_decode = [
        (r["completion_tokens"] - 1) / r["decode_span"]
        for r in ok
        if r["decode_span"] > 0 and (r["completion_tokens"] or 0) > 1
    ]
    return {
        "ctx": ctx,
        "conc": conc,
        "n_ok": len(ok),
        "errors": errs,
        "wall": wall,
        "prompt_tokens_each": [r["prompt_tokens"] for r in ok],
        "ttft_each": [round(r["ttft"], 3) for r in ok],
        "ttft_max": max(r["ttft"] for r in ok),
        "prefill_tps_per_req": round(sum(per_prefill) / len(per_prefill), 1) if per_prefill else None,
        "prefill_tps_aggregate": round(ptok / max(r["ttft"] for r in ok), 1),
        "decode_tps_per_req": round(sum(per_decode) / len(per_decode), 2) if per_decode else None,
        "decode_tps_aggregate": round(sum(per_decode), 2) if per_decode else None,
        "gen_tokens_each": [r["completion_tokens"] for r in ok],
        "sample": ok[0]["text"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8099/v1/completions")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--ctx", default="1000,8000,32000,128000,256000,512000,1000000")
    ap.add_argument("--conc", default="1,2,4,8")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--out", default="bench_v41_matrix.jsonl")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()
    ctxs = [int(x) for x in args.ctx.split(",")]
    concs = [int(x) for x in args.conc.split(",")]

    if not args.no_warmup:
        w = one_request(args.url, args.model, make_prompt(2000, 1), 16, args.timeout)
        print(f"warm-up discarded: ttft {w['ttft']:.2f}s, {w['completion_tokens']} tok")

    print(f"{'ctx':>8} {'c':>2} {'ok':>2} {'prompt tok':>10} {'TTFT max s':>10} "
          f"{'prefill/req':>11} {'prefill agg':>11} {'decode/req':>10} {'decode agg':>10}")
    with open(args.out, "a") as out:
        for ctx in ctxs:
            for c in concs:
                cell = run_cell(args.url, args.model, ctx, c, args.gen,
                                seed_base=ctx * 10 + c * 1000, timeout=args.timeout)
                cell["ts"] = time.time()
                out.write(json.dumps(cell) + "\n")
                out.flush()
                if "error" in cell:
                    print(f"{ctx:>8} {c:>2}  FAILED: {cell['error']}")
                    continue
                print(f"{ctx:>8} {c:>2} {cell['n_ok']:>2} {cell['prompt_tokens_each'][0]:>10} "
                      f"{cell['ttft_max']:>10.2f} {cell['prefill_tps_per_req']:>11} "
                      f"{cell['prefill_tps_aggregate']:>11} {cell['decode_tps_per_req']:>10} "
                      f"{cell['decode_tps_aggregate']:>10}")
                sys.stdout.flush()


if __name__ == "__main__":
    main()
