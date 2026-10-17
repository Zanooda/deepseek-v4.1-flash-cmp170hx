# Results: DeepSeek-V4.1-Flash on 8× CMP 170HX

Everything here was measured on one box between 2026-09-10 and 2026-09-11: eight CMP 170HX
(GA100, sm_80, 64 GB, Gen2 x4), one EPYC 7402P, 278 GB DDR4 (247 GB and 32 GB during the
earlier runs, noted where it matters), the checkpoint on local NVMe (the pre-1017
`deepseek-ai/DeepSeek-V4.1-Flash` weights — the 2026-10-17 `-1017` refresh is a weights-only
update and was not re-measured here), image
`zanooda/vllm-sm80-ds41f:v41-sm80` plus the bind-mounted patches. Unless stated otherwise:
PP=8 with `5,5,5,5,5,5,5,5`, `--max-model-len 1048576`, DSpark with 5 draft tokens, greedy
decoding, unique word-salad prompts, prefix cache cold. Harnesses are in `bench/`, raw data
in `bench/*.jsonl`.

## 1. Correctness

| check | result |
|---|---|
| chat coherence (factual, arithmetic, code, multi-turn memory, tool call) | 5/5 |
| needle-in-haystack, 4k / 32k / 128k / 512k / 1M (820k real tokens), depths 10/50/90 % | 19/19 |
| 4 concurrent 32k needles, distinct passphrases | 4/4, no cross-request bleed |
| needles re-run after patch 0011 (32k/128k/512k, 3 depths, +4 concurrent) | 13/13 |
| needles re-run after patch 0013 (128k, 3 depths, Engram from pinned RAM) | 3/3 |
| automatic tool choice, `deepseek_v41` parser, through a litellm proxy | correct `get_weather({"city": "Lisbon"})`, `finish_reason: tool_calls` |
| images (the checkpoint's vision encoder), 1 and 3 images per request | colours and layout described correctly |
| `prompt_tokens_details.cached_tokens` on a repeated 8,042-token prompt | 7,936 |

Patch 0011 (candidate-only indexer scoring) was verified bit-exact against the full path on
one card: identical scores (max abs diff 0.0) and identical top-k sets on 96 prefill rows and
48 decode rows at two cache block sizes.

## 2. Prefill, single stream

Time to first token of one request, prompt tokens / TTFT. Three stages of the stack:

| prompt tokens | Engram gather in Python (first start) | C gather from NVMe (patch 0010) | tables pinned in RAM (patch 0013) |
|---|---|---|---|
| 850 | 1.6 s · 542 tok/s | | |
| 6,586 | 5.5 s · 1,188 | | |
| 26,208 | 13.0 s · 2,023 | 8.1 s · 3,255 | **5.8 s · 4,486** |
| 104,881 | 42.6 s · 2,462 | 21.3–23.5 s · 4,455–4,930 | **17.3 s · 6,066** |
| 209,738 | 83.9 s · 2,500 | | |
| 419,430 | 169.5 s · 2,475 | 88–106 s · 3,965–4,780 | **78.7 s · 5,328** |
| 1,007,820 | 468.5 s · 2,151 | not re-run | not re-run |

The disk-mode variant with both Engram shards resident in the page cache
(`launch/warm-engram-cache.sh`) measured 6.65 s at 26k and 80.3 s at 420k: ~80 % of the
pinned-RAM gain without pinning.

**Concurrent prefill is compute-bound, not serialised.** Eight simultaneous 105k prompts
finished in 122 s (about 6,900 tok/s aggregate, arrival order preserved) with every card at
92–96 % utilisation (69 % on the last rank). The requests' chunks pipeline across the eight
stages; the cards are simply busy. At 4096-token chunks this is roughly 17 % of the cards'
bf16 peak, which puts the remaining prefill headroom in the Marlin W4A16 GEMMs at large M.

## 3. Decode, single stream

| context | Engram from NVMe | tables pinned in RAM |
|---|---|---|
| 26k | 93 tok/s | 42–51 tok/s (a different prompt; see acceptance) |
| 105k | 104–109 tok/s | **117 tok/s** |
| 420k | 74–86 tok/s | **96 tok/s** |

The step rate is constant at about 16 steps/s (60 ms per step, 8 stages plus the drafter).
Tokens per step follow DSpark acceptance, which follows the text: about 5.9 tokens per step
on repetitive output (98–117 tok/s), about 3.9 on prose-like output (61 tok/s). Early
per-position acceptance on prose was 77 / 55 / 42 / 34 / 31 %. A 3,000-token generation
holds a flat 97–101 tok/s from the first to the last 500 tokens, so decode does not slow with
generated length.

## 4. Decode, several streams together

`bench/bench_v41_decode_window.py`: long generations with per-chunk timestamps; only 1 s
slots in which every stream is generating and no prefill is in flight count.

| prompt tokens | streams | aggregate | per stream | run |
|---|---|---|---|---|
| 26k | 8 | 452 tok/s | 56.5 | NVMe Engram |
| 105k | 8 | **532 / 531 tok/s** | 66 | NVMe Engram, two runs |
| 105k | 8 | 514 tok/s | 64 | + P2P |
| 105k | 8 | 460 / 464 tok/s | 58 | tables in RAM / page cache |
| 420k | 4 | 411 tok/s | 103 | NVMe Engram |

Every row here is the **stock scheduler** (patch 0019 not in the engine, 2026-09-11) and all
of them are cold-prompt arrivals; the same runs with warm prompts were 274/280. Section 11
is what changed that, and the note below it.

Throughput is flat in context length. The 450–532 scatter at 105k × 8 is not the
configuration but request arrival: with pipeline parallelism vLLM's scheduler puts every
runnable request into one micro-batch. Eight requests that become runnable in the same step
(prefix-cache hits, all prefills done at once) travel the eight stages as a single batch:
**274 and 280 tok/s** in two runs; 8 s staggered starts: 389; requests joining one at a time
(cold prompts, 21 s prefills): 514–532. Batches only merge and never split, so sustained
load drifts toward the low figure. A scheduler patch that spreads runnable requests over the
in-flight micro-batches is **section 11** (patch 0019: the same 8 streams then run 560 tok/s
warm and 571 cold). `--max-num-seqs` is not that knob: it bounds the
total number of running requests (1 gives "Running: 1, Waiting: 7").

## 5. Memory and KV pool

| configuration | KV pool | concurrent 1M requests |
|---|---|---|
| utilisation 0.90, before patch 0011 | 2,827,499 tokens | 2.7 |
| utilisation 0.90, patch 0011 | 3,151,289 tokens | 3.0 |
| utilisation 0.95 (launch default) | **6,171,394 tokens** | 5.9 |
| utilisation 0.95, partition `5,5,5,5,5,5,6,4` | 6,859,213 tokens | 6.5 |
| utilisation 0.95, 2026-10-13 tree (patches 0014–0019) | 5,928,408 tokens | 5.65 |

The pool is decided by the last rank (52 GB of weights, drafter and `lm_head` plus non-torch
memory leave it 3.3 GB at 0.90 and 6.5 GB at 0.95) and by vLLM's uniform block pool: every
rank gets the same block count and a block costs each rank the page of its largest cache
group. The last row is the live engine's own `/metrics` (`GPU KV cache size`) on the same
utility 0.95; the 4 % drop against the 2026-09-11 figure is not attributed to a specific
change. V4.1's own KV is about 2.3 kB per token over the whole model. Weights per rank:
35.9 GB on ranks 1–6, 37.6 on rank 0, 46.3 on rank 7. Startup about 10 minutes (204 s of
weight loading per rank, 89–98 s per Engram table into pinned RAM, then profiling and
graph capture).

## 6. Partition trials

| partition | KV pool | 105k c=1 decode | 105k c=8 aggregate |
|---|---|---|---|
| `5,5,5,5,5,5,5,5` | 3,151,289 | 104 tok/s | 532 tok/s |
| `5,5,5,5,5,5,6,4` | 6,859,213 | 106 tok/s | 479 tok/s |
| `5,5,5,5,5,6,6,3` | does not start | | |

Single-stream decode does not depend on the partition (a step is the sum of the stages).
The 479 versus 532 difference is inside the arrival scatter of section 4, so the layout
question is only the pool size; 0.95 utilisation gives most of that pool without moving
layers. A last rank with fewer than four layers fails to start: the DSpark drafter takes its
auxiliary states from layers 36–38, captured on the rank that runs them.

## 7. Engram placement

| mode | host RAM | 26k TTFT | 420k TTFT | note |
|---|---|---|---|---|
| `disk`, Python gather | 0 | 13.0 s | 169.5 s | first start; 125k `preadv` calls/s was the ceiling |
| `disk`, C gather (0010) | 0 | 8.1 s | 88–106 s | 32 GB host |
| `disk` + shards in page cache | ~205 GB evictable | 6.65 s | 80.3 s | `warm-engram-cache.sh` after health |
| `cpu`, pinned (0013) | 203 GB pinned | **5.8 s** | **78.7 s** | 278 GB host; steady 217 GB used |

`cpu` mode as shipped OOM-killed the table-owning workers three times: torch's pinned
allocator rounds a 98.3 GB table to a 128 GiB block (264 GiB for the pair) and the generic
loader materialises a second 98 GB copy before `copy_()`. Patch 0013 pins at exact size via
`cudaHostRegister` and fills the buffers straight from the shard in 256 MB chunks.

## 8. P2P

With the cmpunlocker BAR1 patches (64 GiB BAR1 per card): peer copies verified correct at
1.55 GB/s for same-switch and cross-switch pairs; with `NCCL_P2P_LEVEL=SYS` every pipeline
hop uses `P2P/CUMEM`. Effect: 105k c=1 TTFT 23.5 s (was 23.6–24.5), decode 104 tok/s (was
104–106), c=8 514 tok/s (inside the scatter). The hops are bound by the receiving card's
Gen2 x4 link, which P2P does not widen.

## 9. Hardware notes

- One card (PCI `c3:00.0`) fell off the bus (Xid 79) three times under sustained load at the
  stock 250 W limit: during a 6-layer weight load, during c=8 decode, and 5 s into a memory
  test at 914 GB/s. Memory verified clean (45 iterations × 58 GB). Stable at a 180 W cap;
  the cap is re-applied at boot by a systemd unit and the card takes the last pipeline rank,
  which draws the least. Caveat: `nvidia-smi` power readings exceed the software limits on
  every card during prefill (up to 333 W against 250 W), so whether the firmware enforces
  the cap or only reports inflated draw is unresolved; the memory test is the evidence.
- Six cards link at Gen2 x4, two at Gen2 x16 (different cabling); no measured difference.
- The vLLM logger credits a request's whole prompt to the 10-second window in which its
  first token appears, so "Avg prompt throughput: 10,501 tok/s" for a 105k prompt is an
  artifact; real prefill is TTFT-based.

## 10. Not done

- No in-engine A/B of patch 0011 (`VLLM_DSV41_CAND_LOGITS=0`); the kernel-level numbers
  (2.4 → 0.32 ms per index layer at 48 rows × 131k) put it at a few percent of a step.
- The 1M prefill was not re-run after patches 0010/0011/0013; 468 s is the first-start value.
- `--max-num-batched-tokens 8192` untested (another V4.1-on-170HX setup reports no gain).
- No per-rank profile of the decode step; the 4 ms stages against a 1.5 ms bandwidth floor
  are the largest remaining single-stream lever.

## 11. Decode burst spreading (patch 0019, `VLLM_DSV41_SPREAD_DECODE`)

Section 4's collapse, attacked in the scheduler. The engine keeps `pp_size` (= 8 here)
scheduler steps in flight (`max_concurrent_batches`, `vllm/config/vllm.py`), so the pipeline
is only full when those slots carry *independent* batches; the stock scheduler instead fills
every slot with every runnable request, and a burst therefore rides as one fused batch that
walks the eight stages stage by stage. Patch 0019 (`patches/v41/0019`) admits at most N
decodes per step and re-arms each admitted request `ceil(C/N)` steps later (C = running
decodes), which spreads a burst over the ring.

Measured 2026-10-14, same box and engine as everything else (PP=8, `5,5,5,5,5,5,5,5`, Engram
pinned, DSpark 5, util 0.95), one engine restart per arm, `bench/spread_decode_ab.sh`, the
prefill-free decode window at ~105k-token prompts (raw data:
`bench/bench_v41_decode_window.jsonl`, tags `spread*`). Aggregate / per-stream tok/s at each
run's peak bin:

| arm | c=8 cold | c=8 warm | c=16 cold | c=16 warm |
|---|---|---|---|---|
| stock (patch absent, 2026-09-11 rows) | 531 / 66 | 274 / 34 | not measured | not measured |
| N=1 | 498 / 62 | 446 / 56 | 449 / 75 — only 6 streams ever co-active | 167 / 83 — 2 streams |
| **N=2** | **571 / 71** | **560 / 70** | **813 / 51** | **806 / 50** |
| N=4 | 552 / 69 | 420 / 53 | 736 / 46 | 732 / 46 |

N=2 is the best of the three and it is the one that matters: it **removes the cold/warm
gap** (571 vs 560, where stock is 531 vs 274), i.e. the fused-batch penalty this knob targets
is gone. N=1 starves the ring at c=16 — period 16 against 8 ring slots, so only 6 (cold) or 2
(warm) streams were ever generating at once. N=4 overfills it (period 4 < ring depth) and
loses to N=2. The n=1 bins stayed 54–83 tok/s in every arm, so single-stream is unaffected.

Caveats: arm 0 was not re-run in this session, so the stock column is the 2026-09-11 rows
with an equivalent but not identical prompt set (105.1k vs 104.8–105.0k tokens) — the same
arm's own cold-vs-warm pair is the internal control. c=32 is unmeasured: the engine now runs
`--max-num-seqs 32` with N=2, which makes the period 16, so per-stream latency doubles while
aggregate should stay near the 813 of c=16 (the ring is already full at 8 slots × 2). The
"period ≈ ring depth" rule that explains c=16 (N=2 → period 8) points at N=4 for 32
concurrent; that arm is worth running. And `--max-num-seqs 32` is only a bound: the KV pool
is 5.93M tokens, so 32 requests averaging above ~185k tokens will preempt (watch
`num_preemptions` in `/metrics`).
