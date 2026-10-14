#!/usr/bin/env python3
"""Read a vLLM engine's /metrics twice and report what happened in between.

Read-only: no requests are sent to the engine, so this is safe to run while it
serves. It exists because the interesting decode symptoms are only visible as
*deltas* — the lifetime counters hide them:

  * step composition. How many tokens each scheduler iteration carried (the
    `iteration_tokens` histogram) separates decode steps (~1-8 tokens with
    DSpark) from prefill-chunk steps (hundreds to 4096) from near-empty steps.
  * per-token gaps. `inter_token_latency` diffed over the window gives the
    arrival pattern a streaming client actually feels: a median of tens of ms
    with a tail of seconds means stutter, not low throughput.

Usage: bench/metrics_window.py [--url http://127.0.0.1:8099/metrics] [--window 20]
"""
import argparse
import re
import time
import urllib.request

LINE = re.compile(r"(vllm:[\w:]+)(\{[^}]*\})? ([\d.eE+-]+)$")


def snap(url: str) -> dict[str, float]:
    with urllib.request.urlopen(url, timeout=10) as resp:
        text = resp.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = LINE.match(line.strip())
        if m:
            out[m.group(1) + (m.group(2) or "")] = float(m.group(3))
    return out


def single(metrics: dict[str, float], name: str) -> float:
    for key, value in metrics.items():
        if key.startswith(name + "{"):
            return value
    return 0.0


def buckets(before: dict[str, float], after: dict[str, float], name: str) -> list[tuple[float, float]]:
    rows = []
    for key, value in after.items():
        if not key.startswith(name + "{"):
            continue
        m = re.search(r'le="([^"]+)"', key)
        if not m or float(m.group(1)) == float("inf"):
            continue
        rows.append((float(m.group(1)), max(0.0, value - before.get(key, 0.0))))
    return sorted(rows)


def quantile(hist: list[tuple[float, float]], q: float) -> float | None:
    total = hist[-1][1] if hist else 0.0
    if not total:
        return None
    for bound, cum in hist:
        if cum >= q * total:
            return bound
    return hist[-1][0]


def report(title: str, hist: list[tuple[float, float]]) -> None:
    # Each row is the *cumulative* bucket value, so the last row's delta is the
    # sample count in the window (it equals the counter's own count delta).
    total = hist[-1][1] if hist else 0.0
    if not total:
        print(f"  {title}: no samples")
        return
    parts = [
        f"p50<={quantile(hist, .5):.3g}s",
        f"p90<={quantile(hist, .9):.3g}s",
        f"p99<={quantile(hist, .99):.3g}s",
    ]
    print(f"  {title}: n={total:.0f}  " + "  ".join(parts))
    prev = 0.0
    for bound, cum in hist:
        share = cum - prev
        if share:
            print(f"      <= {bound:8.3f}s: {share:7.0f} ({100 * share / total:5.1f}%)")
        prev = cum


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8099/metrics")
    ap.add_argument("--window", type=float, default=20.0, help="seconds between the two reads")
    args = ap.parse_args()

    before = snap(args.url)
    time.sleep(args.window)
    after = snap(args.url)
    dt = args.window

    steps = single(after, "vllm:iteration_tokens_total_count") - single(before, "vllm:iteration_tokens_total_count")
    toks = single(after, "vllm:iteration_tokens_total_sum") - single(before, "vllm:iteration_tokens_total_sum")
    gen = single(after, "vllm:generation_tokens_total") - single(before, "vllm:generation_tokens_total")
    prompt = single(after, "vllm:prompt_tokens_total") - single(before, "vllm:prompt_tokens_total")
    accepted = single(after, "vllm:spec_decode_num_accepted_tokens_total") - single(before, "vllm:spec_decode_num_accepted_tokens_total")
    drafted = single(after, "vllm:spec_decode_num_draft_tokens_total") - single(before, "vllm:spec_decode_num_draft_tokens_total")

    print(f"--- last {dt:.0f}s at {args.url}")
    print(
        f"  running {single(after, 'vllm:num_requests_running'):.0f}  "
        f"waiting {single(after, 'vllm:num_requests_waiting'):.0f}  "
        f"kv {100 * single(after, 'vllm:kv_cache_usage_perc'):.1f}%  "
        f"preempted(ever) {single(after, 'vllm:num_preemptions_total'):.0f}"
    )
    print(f"  iterations {steps:.0f} ({steps / dt:.1f}/s), {toks:.0f} tokens scheduled, mean {toks / max(steps, 1):.1f} tok/step")
    print(f"  generated {gen:.0f} tok ({gen / dt:.1f}/s)   prompt-logged {prompt:.0f} tok "
          f"(counting cache hits whole; see RESULTS section 9)")
    if drafted:
        print(f"  spec decode: {accepted:.0f} accepted / {drafted:.0f} drafted ({100 * accepted / drafted:.1f}% per-draft)")

    print("  step composition (scheduler iterations by tokens carried):")
    hist = buckets(before, after, "vllm:iteration_tokens_total_bucket")
    prev = 0.0
    for bound, cum in hist:
        share = cum - prev
        if share:
            print(f"      <= {bound:7.0f} tok: {share:6.0f} steps ({100 * share / max(steps, 1):5.1f}%)")
        prev = cum

    print("  per-token gaps as the engine measured them:")
    report("inter_token_latency", buckets(before, after, "vllm:inter_token_latency_seconds_bucket"))
    report("time_per_output_token", buckets(before, after, "vllm:time_per_output_token_seconds_bucket"))


if __name__ == "__main__":
    main()
