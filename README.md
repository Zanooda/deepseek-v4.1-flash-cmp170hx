# DeepSeek-V4 and V4.1 Flash on CMP 170HX mining cards

vLLM patches, container builds, launch scripts and benchmark harnesses for
**DeepSeek-V4.1-Flash** on eight and **DeepSeek-V4-Flash-0731** on four 64 GB
**CMP 170HX** (sm_80, PCIe Gen2 x4). Every number below was measured on that hardware.

| | DeepSeek-V4.1-Flash, 8 cards | DeepSeek-V4-Flash-0731, 4 cards |
|---|---|---|
| decode, one stream | **117 tok/s** at 128k, 96 at 512k (text-dependent, see below) | **98 tok/s** |
| decode, 8 streams together | **532 tok/s** aggregate at 128k (66 per stream) | 713 tok/s at 64 streams |
| prefill, one stream | **6,066 tok/s** at 105k tokens (time to first token 17 s) | ~5,300 tok/s |
| longest verified prompt | 1,007,820 tokens (the model's full 1M) | 1,047,736 tokens |
| KV pool | 6.17M tokens, 5.9 concurrent 1M requests | |
| speculative decoding | DSpark, up to 6 tokens per step | DSpark, 1.93× |

This repository is based on the work in
[allover326/deepseek-v4-cmp170hx](https://github.com/allover326/deepseek-v4-cmp170hx),
which produced the DeepSeek-V4 stack, the V4 patch series and the V4 measurements below;
the DeepSeek-V4.1 series and its results were added on top. Both stacks build on
[haosdent/vllm@dsv4-flash-a100](https://github.com/haosdent/vllm/tree/dsv4-flash-a100),
the sm_80 DeepSeek-V4 backend (see [vllm#50576](https://github.com/vllm-project/vllm/issues/50576)).
Deep dives: [SETTINGS.md](SETTINGS.md) (every flag and why), [RESULTS.md](RESULTS.md)
(all measurements, including what does not work), [patches/README.md](patches/README.md)
(patch-by-patch notes and the base-commit recovery procedures).

---

## DeepSeek-V4.1-Flash on 8 cards

vLLM support for V4.1 is [PR #56201](https://github.com/vllm-project/vllm/pull/56201);
this repo ports the PR onto the sm_80 fork. On sm_80 the FP4 experts and FP8 dense layers
run as Marlin W4A16/W8A16, attention and indexer scoring are Triton, fp8 conversions are
software. Weights take ≈ 41 GB per card at PP=8 (46 GB on the last rank, which also holds
`lm_head` and the drafter). The two 98 GB Engram tables live in pinned host RAM (203 GB)
or, on smaller hosts, are gathered per step from the NVMe shards.

### Quick start (V4.1)

```bash
# 1. the patched vLLM tree: PR #56201 head + all 13 patches
git clone https://github.com/vllm-project/vllm.git ~/vllm-v41 && cd ~/vllm-v41
git fetch origin pull/56201/head && git checkout -b v41-sm80 79a7108d9a
git am ~/deepseek-v4-cmp170hx/patches/v41/*.patch

# 2. the image: sm_80-only SASS, built from the PR head + patches 0001-0008
docker pull zanooda/vllm-sm80-ds41f:v41-sm80

# 3. checkpoint deepseek-ai/DeepSeek-V4.1-Flash (510 GB, 48 shards) under /models, then
DSV41_VLLM_SRC=$HOME/vllm-v41/vllm ~/deepseek-v4-cmp170hx/launch/run-v41-pp8.sh
```

**The bind mounts are not optional.** The published image contains patches 0001–0008 only,
which is the part that needs a CUDA build. Patches 0009–0013 are Python-only and the launch
script mounts them over the image from the checkout you point `DSV41_VLLM_SRC` at: every
`vllm/*.py` file that differs from the image's commit. Without them the engine does not
start (0009 is what makes a PP=8 split of V4.1's shared caches possible at all), and
without 0010–0013 you get 3.4× slower prefill, full-context indexer scoring, no host-RAM
Engram and an incomplete tool-call parser. The step-1 checkout is therefore part of the
runtime, not a build convenience. To do without it, rebuild the image from the full series
with `docker/vastai-build-push-v41.sh` (any box with ≥ 48 GB RAM and a CUDA 13 toolkit; it
clones vLLM, checks out the PR head, applies `patches/v41/`, verifies both tree SHAs and
builds `Dockerfile.fullbuild` for sm_80) and set `DSV41_IMG_COMMIT` to that image's commit
so the mount list becomes empty.

Defaults in the launch script (all measured, see [SETTINGS.md](SETTINGS.md#deepseek-v41-flash-launchrun-v41-pp8sh)):
PP=8 with `VLLM_PP_LAYER_PARTITION=5,5,5,5,5,5,5,5`, `--max-model-len 1048576`,
`--gpu-memory-utilization 0.95`, `--block-size 128`, `--max-num-batched-tokens 4096`,
`--max-num-seqs 8`, fp8 KV, `--engram-config '{"storage":"cpu"}'`, DSpark with 5 draft
tokens, `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, `deepseek_v41` tool and reasoning parsers,
`--enable-prompt-tokens-details`, up to 8 images per request. Startup takes about 10 minutes
(204 s of weight loading per rank, ~95 s to read the Engram tables into RAM, then profiling
and graph capture).


### The V4.1 patch series

Thirteen patches on top of the PR head, in `patches/v41/`. Details and rationale per patch
in [patches/README.md](patches/README.md#-2026-09-10-deepseek-v41-flash-series--patchesv41-runs-on-8-cmp-170hx-pp8).

| # | what it does |
|---|---|
| 0001 | **The fork's sm_80 layer, rebased onto the PR head.** 72 files: the Ampere DeepSeek-V4 attention on the Triton kernels, `fp8_sm80.py` (software fp8), `mqa_logits_triton.py` (indexer scoring), Marlin/MoE/mHC/DSpark work, `topk.cu`. 49 files applied clean, 22 were hand-merged. |
| 0002 | **Our V4 patches, hand-ported:** DSpark under pipeline parallelism (draft `pipeline_parallel_size=1`, `broadcast_draft()`, draft-token scatter on non-last ranks, drafter embedding from the checkpoint), the prefill top-k torch fallback and `DSV4_LOGITS_ROW_CHUNK`, both extended to V4.1's candidate-block select/mask. |
| 0003–0006 | **DSML tool-call recovery**, vLLM PR #52645 commits 1–4: recover tool calls whose outer `<｜DSML｜ calls>` wrapper is missing or corrupted, hold them provisional until the invoke closes and the function name matches a declared tool. |
| 0007 | Merge fix-ups found by a static undefined-name pass over the hand-merge (ROCm ragged graph-buffer builder, three tilelang kernels, a test import). |
| 0008 | **What makes V4.1 run on sm_80.** An Ampere attention class over the PR's ROCm Triton sparse-MLA implementation (backend `TRITON_MLA_SPARSE_DSV41`); software fp8 in every new V4.1 kernel; dense MXFP8 linears through Marlin W8A16 (with a bf16 dequant exception for `wo_a`, which an einsum reads raw); and `EngramConfig.storage = cpu \| gpu \| disk`, where `disk` gathers only the rows a step needs from the safetensors shards. |
| 0009 | **What the first engine start needed** (eight launch iterations). `pp_share.py`: cross-rank replication of the KV-source and indexer caches, because V4.1 shares caches across layers that any PP partition splits across ranks, and upstream cannot do that at all. Also: input ids on every rank, Engram loader skips and a no-mmap safetensors reader, KV-layout intersection and per-worker KV tensor packing, a tile-scheduler stub for the FlashMLA-less SWA path, block size 128, breakable CUDA graphs on. |
| 0010 | **Engram disk gather in C.** The per-step row gather moved from a Python `preadv` thread pool (125k syscalls/s, the prefill ceiling) to a runtime-compiled pthread helper: prefill 3.4× faster. |
| 0011 | **Candidate-only indexer scoring.** Layers 24/28/32/36 score only the 2,048 candidate blocks layer 20 published instead of the whole context and then masking: bit-identical scores, identical top-k, 8× less indexer work at 128k, 64× at 1M, and a 10 % larger KV pool as a side effect (the full-width transient disappears from memory profiling). Kill switch `VLLM_DSV41_CAND_LOGITS=0`. |
| 0012 | Parser accepts the near-miss wrapper spellings `<｜DSML｜calls>` and `<｜DSML｜_calls>` the model writes near the context limit. |
| 0013 | **Engram tables in pinned host RAM without the load transient.** Torch's pinned allocator rounds a 98 GB table to a 128 GiB block and the generic loader materialises a second copy before `copy_()`, so `storage=cpu` needed ~330 GB and was OOM-killed three times. Exact-size `cudaHostRegister` plus a direct chunked fill from the shard bring it to 203 GB. Prefill +25–38 %, decode +10 % over the NVMe gather. |

### V4.1 benchmarks

All numbers: 8× CMP 170HX, PP=8 as launched above, word-salad prompts of unique content,
greedy decoding, prefix cache cold unless noted. Raw data in `bench/*.jsonl`; tables and
caveats in [RESULTS.md](RESULTS.md#deepseek-v41-flash-on-8-cmp-170hx).

**Prefill, one stream** (time to first token, prompt tokens / TTFT):

| prompt tokens | Engram from NVMe (patch 0010) | Engram in RAM (patch 0013) |
|---|---|---|
| 26,208 | 8.1 s · 3,255 tok/s | **5.8 s · 4,486 tok/s** |
| 104,881 | 21.3–23.5 s · 4,455–4,930 tok/s | **17.3 s · 6,066 tok/s** |
| 419,430 | 88–106 s · 3,965–4,780 tok/s | **78.7 s · 5,328 tok/s** |
| 1,007,820 | 468 s · 2,151 tok/s (measured before 0010/0011/0013; not re-run) | |

**Decode, one stream**: 117 tok/s at 128k, 96 tok/s at 512k (Engram in RAM). The step rate
is a constant ~16 steps/s; tokens per step depend on DSpark acceptance, which depends on
the text: ~5.9 tokens per step on repetitive output (98–117 tok/s), ~3.9 on prose
(61 tok/s). Read any decode figure against the prompt that produced it.

**Decode, several streams together** (only the window in which every stream is generating
and no prefill is in flight; `bench/bench_v41_decode_window.py`):

| prompt tokens | streams | aggregate | per stream |
|---|---|---|---|
| 26k | 8 | 452 tok/s | 56.5 |
| 105k | 8 | **532 tok/s** | 66 |
| 420k | 4 | 411 tok/s | 103 |

Decode throughput is essentially flat in context length. It does depend on request arrival:
with pipeline parallelism vLLM packs all runnable requests into one micro-batch, so eight
requests that become ready in the same step walk the eight stages as a single batch
(274 tok/s) while requests that join one at a time overlap (532). Batches merge and never
split; a scheduler patch to spread requests over the in-flight micro-batches is the open
item. The older `bench_v41_matrix.py` "c=4 / c=8 decode collapse" numbers in RESULTS.md were
this effect plus prefill interference, not decode scaling.

**Memory**: KV pool 6,171,394 tokens at utilisation 0.95 (5.9 concurrent 1M-token
requests). The pool is set by the last rank, which has 46 GB of weights; V4.1's own KV is
tiny (~2.3 kB per token over the whole model). `5,5,5,5,5,5,6,4` gives 6.86M tokens but
costs concurrent decode; a last rank with fewer than 4 layers cannot start (the drafter's
auxiliary states come from layers 36–38).

**Correctness**: chat coherence 5/5 (factual, arithmetic, code, multi-turn memory, tool
call); needles 19/19 at 4k, 32k, 128k, 512k and 1M (820k real tokens), depths 10/50/90 %,
plus 4 concurrent needles with distinct passphrases and no cross-request bleed; automatic
tool choice with the `deepseek_v41` parser; images (the checkpoint's vision encoder) verified
with multi-image prompts; `prompt_tokens_details.cached_tokens` reported for prefix-cache
hits.

### Operational findings

- **Power.** One card fell off the PCIe bus (Xid 79) three times under sustained load at the
  stock 250 W limit and is stable at 180 W (`nvidia-smi -pl 180`, persisted by a systemd unit).
  Memory tested clean; it is power delivery on that card. Put a capped card on the last pipeline rank,
  which draws the least (`DSV41_GPUS` sets the device order).
- **P2P (cmpunlocker BAR1 patches).** Works across all pairs at 1.55 GB/s and NCCL uses it
  for every pipeline hop (`NCCL_P2P_LEVEL=SYS`), but the hops are bound by the Gen2 x4 link
  either way: no measurable speed change. Tensor parallel stays 6.6× slower on prefill.
- **Engram placement.** `cpu` (pinned RAM, needs ~240 GB total) is the default; `disk` with
  `launch/warm-engram-cache.sh` (both shards in the evictable page cache) reaches ~80 % of the
  gain on 210–240 GB hosts; plain `disk` runs on 32 GB with NVMe latency per step.
- **Reasoning effort** is a numeric budget 1–100 in the prompt. `low`/`high`/`xhigh`/`max`
  map to 25/50/75/100 on this server, `none` disables thinking; `minimal` and `medium` are
  rejected.

---

## DeepSeek-V4-Flash-0731 on 4 cards

PP=4 with DSpark speculative decoding, verified to the model's full 1,047,736-token context.

### Quick start (V4)

```bash
docker pull zanooda/vllm-sm80:c3046d1          # patches 0002-0008 baked in
docker tag zanooda/vllm-sm80:c3046d1 dsv4-a100:devel
launch/run-pp-dspark.sh                         # sources default to ~/vllm/vllm
```

Building yourself means a full CUDA source build (2–4 h, ≥ 48 GB RAM) of the fork at
`c3046d1` plus `patches/0002`–`0009`; `c3046d1` was force-pushed away and is only recoverable
by tarball, procedure in [patches/README.md](patches/README.md#getting-c3046d1--it-is-unreachable-by-any-git-method).
`docker/vastai-build-push.sh` does the whole thing on a rented box. Three cards also work
with `VLLM_PP_LAYER_PARTITION=15,15,13`; two cannot hold the weights.

### The V4 patch series

| # | file | what it does |
|---|---|---|
| 0001 | `sparse_attn_indexer.py` | `has_device_capability(90)` gate on the persistent top-k (legacy `f8ea5bb` base only; upstream in `c3046d1`) |
| 0002 | `config/speculative.py` | the DSpark draft runs whole on the last rank: `draft_parallel_config.pipeline_parallel_size = 1` |
| 0003 | `v1/worker/gpu/pp_utils.py` | `broadcast_draft()` and its receive: relay draft tokens to the ranks that must verify them (vLLM PR #46994) |
| 0004 | `v1/worker/gpu/model_runner.py` | drop the "no DSpark under PP" guard, call `broadcast_draft()`, scatter relayed tokens |
| 0005 | `spec_decode/dspark/utils.py` | drop the `NotImplementedError`, load the draft's token embedding from the checkpoint (under PP the target's embedding lives on rank 0) |
| 0005a | `sparse_attn_indexer.py` | torch fallback for the prefill top-k on sm_8x: the CUDA kernel leaves indices uninitialised above ~128k and crashes with an illegal memory access |
| 0006 | `sparse_attn_indexer.py` | `DSV4_LOGITS_ROW_CHUNK`: row-chunk the indexer's `[M, N]` fp32 logits transient. **The context-ceiling fix: ~134k → 1,047,736 tokens.** |
| 0007 | `parser/deepseek_v4.py` + engine | DSML tool-call recovery for missing/corrupted wrappers (vLLM PR #52645, commits 1–4) |
| 0008 | `rejection_sampler_utils.py` | NaN block maxima → `-inf` before `argmax` (vLLM PR #50183) |
| 0009 | `models/deepseek_v4/**`, `csrc` | CUDA-graph/indexer corruption fixes (vLLM PRs #52492, #52836): the intermittent token-salad windows across all sessions |

Together, 0002–0005 are what enables **DSpark under pipeline parallelism**, which vLLM
refuses in three places. It is worth 1.93× on decode and, unlike on tensor parallel where
speculation turns negative above ~8 streams, keeps winning to 64 concurrent streams.

### V4 benchmarks

| | plain | **+ DSpark** |
|---|---|---|
| decode, single stream | 50.8 | **98.1 tok/s** |
| decode, 64 concurrent | 472.0 | **712.8 tok/s** |
| decode at 100k context | 38.8 | **90.0 tok/s** |
| prefill (25k–77k context) | ~5,300 | ~5,200 tok/s |
| time to first token at 100k | 14.6 s | 14.6 s |
| time to first token at 1M | | 9.2 min (decode 35.6 tok/s) |

Pipeline parallel beats tensor parallel by 6.6× on prefill here: TP performs 86 all-reduces
per forward over a Gen2 x4 link with no P2P and measures flat at ~800 tok/s. Full tables:
[RESULTS.md](RESULTS.md).

### V4 known limits

- **Conversations have a lower ceiling than one-shot prompts.** At `DSV4_LOGITS_ROW_CHUNK=128`
  a one-shot prefill reaches 1,047,736 tokens but a multi-turn chat dies at ~718–733k; at `64`
  a 405-turn chat reached 1,002,852 clean. Use 64 for conversational workloads.
- **Retrieval accuracy degrades with depth**: 100 % at 150k, 87 % at 300k, 60 % at 450k,
  50 % at 750k, 30 % at 900k. `index_topk` is a fixed 512 while the candidate pool grows;
  treat 1M as a large working set, not a database. Thinking recovers part of it
  (61 → 81 % at depth), the effort level does not.
- **DSpark output is not reproducible at temperature 0**, a property of DSpark itself.
- **Never use `--enforce-eager`**: 8–10 tok/s.

---

## Troubleshooting: cards running 4× slow (PWRBRK#)

Some boards (the ASUS Pro WS WRX80E-SAGE used for the V4 numbers) assert `PWRBRK#` on edge
pin **B30** and pin the card at ~88 W, 1140 MHz, a quarter of its fp16 and half its bandwidth.

```bash
nvidia-smi -q | grep -A1 "HW Power Brake Slowdown"      # "Active" on an idle card = braked
```

Fix: Kapton tape over pin B30 (B side, counted from the notch), a riser that does not route
B30, or a BIOS/BMC option if the board has one. Do nothing if it reads "Not Active".

---

## Repo layout

```
patches/            V4 series 0001-0009 (on haosdent/vllm c3046d1) and patches/README.md
patches/v41/        V4.1 series 0001-0013 (on vLLM PR #56201 head 79a7108d9a)
docker/             Dockerfile.fullbuild, Dockerfile.devel, vastai-build-push.sh (V4),
                    vastai-build-push-v41.sh (V4.1, verifies both tree SHAs)
launch/             run-pp-dspark.sh (V4), run-a100.sh (V4 TP variant),
                    run-v41-pp8.sh (V4.1 PP=8), warm-engram-cache.sh (page-cache Engram)
bench/              every harness behind RESULTS.md: bench_v41_check.py (coherence + needles),
                    bench_v41_matrix.py (context x concurrency), bench_v41_decode_window.py
                    (prefill-free concurrent decode), v41_partition_trial.sh, the V4 harnesses,
                    and the raw *.jsonl
SETTINGS.md         every flag and environment variable, and why it has that value
RESULTS.md          all measurements, correctness tests, limits and measurement pitfalls
```

## Clients

The engine speaks the OpenAI API (`/v1/chat/completions`, `/v1/completions`) with tool
calling, reasoning content, images and prefix-cache usage details. A litellm proxy in front
of it works unchanged; allow `reasoning_effort` through (`allowed_openai_params`) so clients
can set the thinking budget per request.

## License

Apache-2.0, matching vLLM. The patches are derivative of vLLM, of
haosdent/vllm@dsv4-flash-a100, and of
[allover326/deepseek-v4-cmp170hx](https://github.com/allover326/deepseek-v4-cmp170hx).
