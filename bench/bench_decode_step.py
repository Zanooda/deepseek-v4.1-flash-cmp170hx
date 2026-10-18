#!/usr/bin/env python3
"""Decode step cadence vs context, from streaming chunk intervals.

Under DSpark vLLM emits one SSE chunk per engine step and each chunk carries
the accepted tokens, so the interval between text chunks is the step period.
This is the measurement RESULTS §3 quotes: 40-56 ms from 4k to 115k prompt
tokens, flat. It also prints the tokens each step carried, which is the part
that changes with the text (acceptance); comparing tok/s across prompts
without it attributes acceptance changes to the step.

The engine may be serving other requests. `running` is sampled alongside;
with `VLLM_DSV41_SPREAD_DECODE=2`, a request shares every step while at most
two requests are decoding, so the interval stays the step period. A prefill
from another request can inflate a sample (the 82k row in RESULTS §3).

Usage: bench/bench_decode_step.py [--ctx 5000,20000,40000,70000]
"""
import argparse
import json
import random
import statistics
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
    body = " ".join(rng.choice(WORDS) for _ in range(max(8, int(approx_tokens / 1.3))))
    return f"[run {seed}] Notes: {body}\nWrite a very long, detailed summary of the notes above."


def running(url: str) -> int:
    with urllib.request.urlopen(url + "/metrics", timeout=10) as resp:
        for line in resp.read().decode().splitlines():
            if line.startswith("vllm:num_requests_running{"):
                return int(float(line.rsplit(" ", 1)[1]))
    return -1


def probe(url: str, model: str, approx_ctx: int, seed: int, gen: int) -> dict:
    payload = {
        "model": model,
        "prompt": make_prompt(approx_ctx, seed),
        "max_tokens": gen,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        url + "/v1/completions",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    stamps: list[float] = []
    usage: dict = {}
    conc: list[int] = []
    stop = False

    def sample() -> None:
        while not stop:
            try:
                conc.append(running(url))
            except OSError:
                pass
            time.sleep(1.0)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=7200) as resp:
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
            if any(choice.get("text") for choice in obj.get("choices", [])):
                stamps.append(time.perf_counter())
    stop = True

    gaps = sorted(b - a for a, b in zip(stamps, stamps[1:]))
    ctoks = usage.get("completion_tokens", len(stamps))
    n = max(len(stamps), 1)
    return {
        "ctx": approx_ctx,
        "prompt_tokens": usage.get("prompt_tokens"),
        "tokens": ctoks,
        "tok_per_step": round(ctoks / n, 1),
        "ttft_s": round(stamps[0] - t0, 2) if stamps else None,
        "step_ms": round(statistics.median(gaps) * 1e3, 1) if gaps else None,
        "p10_ms": round(gaps[len(gaps) // 10] * 1e3, 1) if gaps else None,
        "p90_ms": round(gaps[9 * len(gaps) // 10] * 1e3, 1) if gaps else None,
        "running": f"{min(conc, default=-1)}-{max(conc, default=-1)}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--ctx", default="5000,20000,40000,70000")
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--seed", type=int, default=9100)
    args = ap.parse_args()

    contexts = [int(x) for x in args.ctx.split(",")]
    print(f"{'ctx req':>8} {'prompt':>8} {'tok/step':>8} {'step ms':>8} {'p10-p90':>13} {'running':>8}")
    for i, ctx in enumerate(contexts):
        r = probe(args.url, args.model, ctx, args.seed + i, args.gen)
        spread = f"{r['p10_ms']}-{r['p90_ms']}" if r["step_ms"] is not None else "-"
        print(
            f"{r['ctx']:>8} {str(r['prompt_tokens']):>8} {r['tok_per_step']:>8} "
            f"{str(r['step_ms']):>8} {spread:>13} {r['running']:>8}"
        )


if __name__ == "__main__":
    main()
