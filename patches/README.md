# Patches

## ★ 2026-09-10: DeepSeek-V4.1-Flash series — `patches/v41/` (RUNS on 8× CMP 170HX, PP=8)

A second, independent patch series for **DeepSeek-V4.1-Flash** on sm_80. It does **not**
stack on the V4 series below: the base is different, and everything below this section
still describes the V4-Flash-0731 stack (`c3046d1` + `0002`–`0009`), which is unchanged.

**Base:** upstream vLLM **PR #56201** ("[Model] Support DeepSeek-V4.1-Flash", *open*,
unmerged), head commit `79a7108d9aea27ddab99ce1779290d300b17fc23`, tree
`a731238e73f0023b0a43f6d91c07c466295224e6`. The PR is 147 files / +20k lines against
September `main`, 1,429 upstream commits ahead of the V4 base. Forward-porting it onto
`c3046d1` was tried first (80 files clean, 37 conflicted, 68 of its modified files had
drifted upstream in between, and its new code imports a dozen upstream modules the old
base lacks) and abandoned; the series is the inverse port instead: **PR head + the fork's
sm_80 layer + our patches**, which is what `patches/v41/` contains.

```bash
git clone https://github.com/vllm-project/vllm.git && cd vllm
git fetch origin refs/pull/56201/head && git checkout -B v41-sm80 79a7108d9a
git rev-parse HEAD^{tree}     # a731238e73f0023b0a43f6d91c07c466295224e6
git am ../deepseek-v4-cmp170hx/patches/v41/*.patch
git rev-parse HEAD^{tree}     # 70aaaced9d8d2c2dce411b1aef19287c0134bf77
```

Verified 2026-09-10: `git am` of the eight patches onto a pristine `79a7108d9a` worktree
applies with zero conflicts and reproduces tree `70aaaced…` byte-for-byte. If the PR is
force-pushed and the SHA becomes unreachable, the tarball route from the V4 section
works for `vllm-project/vllm` too; [docker/vastai-build-push-v41.sh](../docker/vastai-build-push-v41.sh)
does both and checks both tree SHAs.

| # | patch | what | notes |
|---|---|---|---|
| v41/0001 | `sm_80-fork-delta` (72 files, +11.5k) | haosdent's whole sm_80 layer (`62195e9..c3046d1`: the Ampere DSv4 attention on the ROCm Triton kernels, `fp8_sm80.py`, `mqa_logits_triton.py`, Marlin/MoE/mHC/DSpark work, csrc `topk.cu`/`marlin.cu`/`custom_all_reduce.cuh`) rebased onto the PR head | 49 files clean, 22 hand-merged. Dropped: the fork's `KVBlockZeroer` rewrite (upstream already masks heterogeneous pages) and its `sparse_mla_triton_warmup.py` hunk (file removed upstream; the indexer warms its own Triton kernels). Marlin MoE `clamp_limit`/`gemm1_*` kwargs became upstream's `activation_config`. |
| v41/0002 | `patches-0002-0006` | our V4 patches **0002–0006** hand-ported (they carry no index blobs, so 3-way cannot apply them): DSpark-on-PP (draft `pipeline_parallel_size=1`, `broadcast_draft`, runner scatter, drafter embedding from checkpoint), the prefill top-k torch fallback (0005a) and `DSV4_LOGITS_ROW_CHUNK` (0006) | 0003 applied clean by itself. 0005a/0006 now also row-chunk V4.1's candidate-block select/mask, which sits between the logits and the row top-k. |
| v41/0003–0006 | PR **#52645** commits 1–4 | the DSML malformed-wrapper recovery, i.e. our **0007** — taken from the PR's own (rebased) commits instead of our old hand-port, because the PR's base `7ca49fbe4b` is an ancestor of the V4.1 base | commits 5–6 (reasoning-adapter delegation, orphan hardening) conflict and are skipped, matching the V4 decision. One trivial conflict in `streaming_parser_engine.py` (both sides add a field). |
| v41/0007 | merge fixups | the three defects a static undefined-name pass found in the hand-merge: ROCm ragged graph-buffer builder body/signature mismatch, the fork's three tilelang kernels referencing a removed module-level `pass_configs` (now `@tilelang_jit`), a test import | |
| v41/0008 | **DeepSeek V4.1 on SM8x** | the new work — see below | |
| v41/0009 | **PP=8 runtime fixes** | everything the first engine start needed, found in eight launch iterations — see below | not in the published image (`70aaaced9d`); `launch/run-v41-pp8.sh` bind-mounts these files from a checkout (`DSV41_VLLM_SRC`) until a rebuild (`bdde91029a`) |
| v41/0010 | **Engram disk gather in C** | the per-step row gather of `storage=disk` moves from a Python `preadv` thread pool (~125k syscalls/s, the prefill bottleneck) to a runtime-compiled C `pthread` helper (`engram_gather`, built with gcc into `$VLLM_CACHE_ROOT/engram_pread/`, Python pool as fallback) | prefill 32k c=1 1,6xx → 3,255 tok/s, 128k → 4,832 (3.4×). Python-only, bind-mounted like 0009 |
| v41/0011 | **candidate-only indexer logits** | index layers 24/28/32/36 score just the 2,048 candidate blocks layer 20 published instead of the whole context and then masking — see below | lossless (identical top-k set, bit-identical scores); Python-only, bind-mounted like 0009. `VLLM_DSV41_CAND_LOGITS=0` restores the full path |
| v41/0012 | **lenient DSML wrapper spelling** | the V4.1 parser also accepts `<｜DSML｜calls>` / `<｜DSML｜_calls>` (and the closing forms), which the model writes near the context limit; the tool call was already recovered through the invoke markers, this stops the misspelled closing tag leaking into the reply | mirrors Schaka/170hx-journey `ebaadb6`; the "tool call opens inside `<think>`" half of that commit was already covered by the PR #52645 recovery (0003–0006). 107 parser tests pass. Python-only, bind-mounted like 0009 |
| v41/0013 | **Engram storage=cpu without the load transient** | exact-size `cudaHostRegister` pinning (torch's pinned allocator rounds a 98 GB table to 128 GiB) and a direct chunked fill of the pinned tables from the shard (the loader materialised a 98 GB copy first) | needed for `storage=cpu` on the 278 GB box (OOM-killed three times before). Measured: prefill +25–38 %, decode +10 % vs the NVMe gather; both tables load in ~95 s. Python-only, bind-mounted like 0009 |
| v41/0014 | **heal hybrid XML/DSML tool calls** | in long agentic sessions the model sometimes opens a tool call in the generic Anthropic-XML idiom — a plain `<invoke name="...">` or, more often, the invoke fused into the first parameter as `<parameter name="TOOLNAME">`, with plain `<parameter name="...">...</parameter>` children — and converges back to DSML by the close. The opening is not DSML, so the whole call, close tags and all, leaked into the reply as content; the parser now recovers it (lenient opener terminals on the provisional-recovery path, plus a schema-aware converter for the mixed tags) | self-reinforcing without the fix: once one call leaks, the harness echoes that turn's raw content back and the model imitates its own broken format, so every later call in the conversation leaks too. Gated by the tool-name validator (an `<invoke>`/`<parameter>` in prose stays text) and skipped entirely for well-formed DSML. Verified against five real captured leaks, streaming and non-streaming; correct calls unchanged. Python-only, bind-mounted like 0009 |
| v41/0015 | **defer the grammar bitmask for placeholder spec tokens** | under the async PP batch queue a structured-output request that also runs DSpark was killed with "grammar rejected tokens" on ~1/3 of constrained requests: `num_output_placeholders > 0` misses the case where a request is in flight (counter already 0) but its scheduled spec tokens are still `-1` placeholders, so the bitmask is computed against them, the worker verifies the real drafts unmasked, and the grammar rejects the accepted tokens. Now also defers when any scheduled spec token is a placeholder | only affects `use_structured_output` requests; plain generation unchanged. Python-only, bind-mounted like 0009 |
| v41/0016 | **strict tool-call parameter schemas** | port of upstream [PR #56235](https://github.com/vllm-project/vllm/pull/56235): a tool sent with `strict: true` has its parameter values lowered to the tool's JSON schema (string enum/const/pattern/length, JSON-typed values, local `$ref` against the root `$defs`) instead of accepting any text | tools without `strict: true` keep the permissive path unchanged, so non-strict traffic is unaffected. Python-only, bind-mounted like 0009 |
| v41/0017 | **headless tool-call recovery** | the harder half of 0014: the model sometimes drops the invoke opener *and* the tool name too, emitting only the `<parameter …>` tags before `</｜DSML｜ invoke>`. With no name to key on, the tool is inferred from the parameters — the single declared tool whose properties cover every parsed parameter name — with an opt-in headless path in the shared `StreamingParserEngine` (reinterpret the buffered opener as the first parameter, defer the name, confirm a tool can be named before committing) | a tool always covers its own parameters, so a unique cover is the right tool and anything ambiguous stays text; the commit-time check restores an unidentifiable block as text rather than dropping it, streaming inference is safe (a common-only prefix like `{"i"}` defers until a distinctive parameter arrives), and it is suppressed under `tool_choice:"none"`. Both engine hooks default off, so no other parser is touched. Python-only, bind-mounted like 0009 |

**Not carried, because upstream has them:** 0008 (NaN guard: `rejection_sampler_utils.py:111`
upstream) and 0009 (`is_current_stream_capturing` guard + no `eager_scratch.py` upstream).

### v41/0008 — what makes V4.1 run on sm_80

- **Attention** — `models/deepseek_v4_1/ampere/ampere_sparse.py`: 40 lines. The PR ships a
  ROCm Triton implementation (`deepseek_v4_1/amd/rocm.py`, `DeepseekV41ROCMAiterMLAAttention`)
  that already handles everything V4.1 adds over V4 (ratio-0/1/2 layer types, KV sharing
  via `kv_source_layer_ids`, candidate-block filtering, indexer-K-from-latent); its
  aiter-only GEMMs self-disable off ROCm. Exactly as the fork did for V4, the Ampere layer
  is a subclass with backend name `TRITON_MLA_SPARSE_DSV41` and `capability.major == 8`.
  `nvidia/model.py::_select_dsv4_attn_cls` gets a `major == 8` branch (rejects the MXFP4
  indexer cache, which is SM100-only).
- **fp8 below SM89** — the PR's new kernels use native `tl.float8e4nv` converts, which
  Triton refuses on sm_80. Substituted with the fork's `fp8_sm80.py` helpers in
  `deepseek_v4_1/common/ops/{cache_utils,fused_compress_quant_cache,indexer_k_store}.py`
  and the Engram lookup kernel; `cache_utils` gains the fork's `is_cutedsl_supported()` gate.
- **Dense MXFP8 linears** (`[32,32]` blocks, UE8M0) — go through upstream's own
  `MarlinMxfp8LinearKernel` (sm_75+, W8A16) with no change. `wo_a` is the exception: the
  einsum o_proj reads the raw `[N,K]` weight, which a Marlin repack destroys, so the Ampere
  layer sets `wo_a.mxfp8_dequant_to_bf16` and `ModelOptLinearMethod` honours it by picking
  the emulation kernel (bf16 dequant at load; +1.3 GiB total). `_get_cached_wo_a_bf16`
  learns to accept an already-bf16 weight.
- **Engram** — `EngramConfig.storage = cpu | gpu | disk` (+ `disk_threads`). `disk`: the
  two 98 GiB tables are never loaded (their names are registered as skipped in the
  safetensors iterator, so `get_tensor` never materialises them and the "weights not
  initialized" check has nothing to miss); each step the batch's hash ids go to the host,
  are de-duplicated, and the unique rows (256 B fp8 + 8 B UE8M0) are `pread` from the two
  101 GB shards by a thread pool into pinned staging, copied to the device and dequantized
  by the ordinary lookup kernel. The gather runs in the V2 runner's `DeepseekV41ModelState`
  hook (eager, before the forward, never inside graph capture); the forward skips its own
  `prepare_embeddings` for disk tables. `cpu` (upstream default, ~203 GiB pinned) and `gpu`
  are unchanged apart from the byte-view fix.
- Experts (`expert_dtype=fp4`, I8-packed, E8M0 block-16 scales) take the same
  `Mxfp4MoEMethod` → Marlin path the fork already runs for V4-Flash. **Untested on
  V4.1's 384-expert / block-16 layout.**

### v41/0009 — what the first engine start needed

- **PP splits inside kv-sharing groups** (`deepseek_v4_1/pp_share.py`, the big one). V4.1
  layers with `compress_ratio > 0` read the compressed KV cache of the nearest
  `kv_source_layer_ids` entry below them (2, 8, 14, 20); non-owning index sources read that
  source's indexer K cache; layer 20 also publishes candidate blocks for every later indexer
  and each index source publishes top-k indices for the layers up to the next. Upstream
  refuses any partition that splits a group — and layer 20's group is the last 20 layers,
  so no 64 GB partition can satisfy it. The fix registers a same-spec **replica** of the
  source's compressed cache (and indexer K cache) on every rank hosting consumers, under
  the source's exact layer name, so the KV manager gives it the same block tables; each
  step the rows the source wrote (584 B per compressed state, 132 B per index key, gathered
  by slot mapping) plus the candidate-block and top-k rows ride downstream inside the
  pipeline's `IntermediateTensors`, are scattered into the replicas before the consumer
  layers run and re-gathered for the next hop. ~43 MB per hop per 4096-token chunk.
- `input_ids` on every PP rank: V4.1's MoE (image-sentinel routing) and the Engram hash on
  the rank owning layer 14 read token ids; the V2 runner and the CUDA-graph capture path
  cleared them on non-first ranks (`pp_requires_input_ids` model flag).
- Loader: every rank walks the whole checkpoint, so ranks that do not own an Engram layer
  must skip its 98 GiB table too; and `safe_open` of the two 101 GB shards maps them
  privately, which the kernel's overcommit heuristic refuses on a 32 GB host, so the four
  small tensors in those shards are read with `pread` instead.
- KV cache config: workers may support different layout sets (a rank without an indexer
  cache supports more), so intersect instead of asserting equality; and a rank owning no
  layer of a group (ratio-2 compressor ring buffers exist only on ranks 0–2) must not get
  tensors for the group's foreign layers.
- sm_80 SWA metadata builder that skips FlashMLA tile-scheduler planning (FlashMLA is not
  compiled for sm_80); `num_heads` for the fork's Triton indexer; pinned Engram staging
  allocated on the CPU under the CUDA device context; Triton constexprs.
- Launch: `--block-size 128` (256 does not split into the 128-token indexer kernel blocks
  on the layer-compact layout) and breakable CUDA graphs **on** (V4.1 is not
  torch-compiled; without them there are no piecewise graphs at all).

### v41/0011 — candidate-only indexer logits (2026-09-11)

V4.1's two-level selection: layer 20 scores the full context and publishes the 2,048
best 8-position blocks per token; layers 24/28/32/36 mask their own scores to those
blocks before the row top-k (512). The sm_80 Triton path still computed the full
`[rows, context]` logits on those four layers and threw away everything outside the
16,384 candidate positions. `vllm/v1/attention/ops/mqa_logits_candidates_triton.py`
scores only the candidates: one Triton kernel gathers K at `cand*8+o` (prefill from the
gathered K, decode straight from the paged indexer cache through the block table), writes
a compact `[rows, 16384]` matrix with `-inf` outside the causal bound, vLLM's CUDA
`top_k_per_row_decode` picks the 512, and a small kernel maps columns back to positions
(dropping `-inf` picks, which the CUDA kernel would otherwise return for rows with fewer
finite candidates — those would point past the row's causal bound). Taken only when the
row width exceeds 16,384 (below that every block is a candidate and the mask is a no-op).

Verified against `full logits → apply_candidate_mask → top-k` on one card: scores
bit-identical, identical top-k sets on prefill (incl. single-column rows) and decode at
two cache block sizes. Kernel time (64 heads × 128): prefill 64 rows × 131k 2.42 → 0.40 ms,
decode 48 rows × 131k 2.4 → 0.32 ms, 8 rows × 1M 3.0 → 0.16 ms. In the engine the KV pool
grew from 2,827,499 to 3,151,289 tokens (profiling no longer sees the full-width transient);
needles 13/13 after the change. Decode-step impact is small because the indexer was a
smaller slice of the step than estimated — numbers in [RESULTS](../RESULTS.md).

### What was verified, and what was not

Verified: `py_compile` of every changed file; a pyflakes pass (undefined names / syntax)
over the 100+ changed files is clean; the series re-applies byte-exact. **Built** on a
vast.ai box (122 cores, 792 GB RAM) with `docker/vastai-build-push-v41.sh`: both tree SHAs
verified, csrc compiled first try (663 s at MAX_JOBS=64), pushed as
`zanooda/vllm-sm80-ds41f:v41-sm80` (digest `sha256:283fc7b0…`, `_C` 60 and `_moe_C` 22
sm_80 kernels). **Running on the 8× CMP 170HX box since 2026-09-10 22:26 UTC** (`launch/run-v41-pp8.sh`,
Engram from disk, partition 5×8): chat coherence 5/5 (factual, arithmetic, code,
multi-turn memory, tool call), needles **19/19** — depths 10/50/90 % at 4k, 32k, 128k,
512k and 1M (820k real tokens), plus 4 concurrent 32k needles with distinct passphrases
and no cross-request bleed. Speeds in [RESULTS](../RESULTS.md).

---

## ★ 2026-08-13: recommended base is now `c3046d1` — patch 0001 is no longer needed

Upstream's 41-commit serving-optimization campaign (base
`f8ea5bb` → `c3046d1ebd2dae9b94ad2ef5f966ea153632251e`, 2026-08-04) is worth a measured
**+7% decode (p<0.001)** on this hardware with correctness intact — see
the V4 results (removed from RESULTS.md, which now covers V4.1 only), including why the "+30%" you may
have seen claimed for this range does not survive a paired A/B.

On `c3046d1`:

- **Drop `0001`** — its `has_device_capability(90)` gate is now upstream.
- **`0002 0003 0004 0005 0005a 0006` apply unchanged, zero rejects**, in glob order.
  (`0005a` must still precede `0006`; the glob order does that.)
- **`0007` and `0008` also apply unchanged, zero rejects** — the range touches neither
  `vllm/parser/` nor `rejection_sampler_utils.py`. Verified 2026-08-21: all eight patches
  (`0002`–`0008`) applied from the reconstructed tree reproduce the same patched files as
  the `f8ea5bb` series, except `sparse_attn_indexer.py`, which legitimately differs
  (c3046d1 carries the 0001 gate upstream).
- ⚠️ **The range touches `csrc/`** (`libtorch_stable/topk.cu` FilteredTopK decode routing —
  one of the real wins — plus `marlin.cu`, `custom_all_reduce.cuh`), so
  `VLLM_USE_PRECOMPILED=1` and the bind-mount method **cannot deliver the kernel changes**.
  Build from source with [docker/Dockerfile.fullbuild](../docker/Dockerfile.fullbuild)
  (sm_80-only, ~115 min) with the patches applied to the tree first.
- New env worth setting: `VLLM_MARLIN_FP8_DEQUANT_BF16=1` (upstream-adopted prefill win,
  −35 ms TTFT@8k on block-fp8 dense; inert but harmless on the INT4 repack).
  `VLLM_MARLIN_DENSE_OCCUPANCY` was refuted by its own authors (leave unset), and
  `VLLM_DSPARK_VOCAB_SHARD` has **zero consumers** at this commit.

### Getting `c3046d1` — it is unreachable by ANY git method

The branch has been force-pushed **again** (tip is now a single squashed commit with newer,
un-benchmarked work), and this time fetching all refs does not help: `c3046d1` is referenced
by nothing. GitHub still serves unreachable commits by SHA over HTTP, and the tarball
reconstructs the exact tree:

```bash
git clone https://github.com/haosdent/vllm.git && cd vllm
curl -sL -o /tmp/c3046d1.tar.gz \
  https://codeload.github.com/haosdent/vllm/tar.gz/c3046d1ebd2dae9b94ad2ef5f966ea153632251e
mkdir /tmp/c3046d1-src && tar xzf /tmp/c3046d1.tar.gz -C /tmp/c3046d1-src --strip-components=1
export GIT_INDEX_FILE=/tmp/c3046d1.index
git read-tree --empty && git --work-tree=/tmp/c3046d1-src add -Af
git write-tree   # MUST print d13ae12b9a6621ef8d218f53741e59c6db2f68d2 — the upstream tree SHA
git tag c3046d1-recon "$(git commit-tree d13ae12b9a6621ef8d218f53741e59c6db2f68d2 \
  -p f8ea5bb163c161ef38b401d055cc5fd4a934091a -m 'c3046d1 reconstructed from tarball')"
unset GIT_INDEX_FILE && git checkout -B rebase-c3046d1 c3046d1-recon
```

The `git write-tree` check is the whole safety story: if it prints the upstream tree SHA,
your working tree is byte-identical to `c3046d1`.

---

## Legacy base `f8ea5bb` (all nine patches)

Against [haosdent/vllm@dsv4-flash-a100](https://github.com/haosdent/vllm/tree/dsv4-flash-a100)
(commit `f8ea5bb`). Apply with `patch -p1` from the vLLM checkout root.

> ⚠️ **You must check out `f8ea5bb`, and a plain clone will not have it.** The branch was
> force-pushed after these patches were generated: `f8ea5bb` is no longer reachable from the
> tip, a `--depth` clone will not contain it, and the server **refuses fetch-by-SHA**.
> Fetching all refs is what makes it reachable:
>
> ```bash
> git clone --branch dsv4-flash-a100 --single-branch https://github.com/haosdent/vllm.git
> cd vllm
> git fetch origin '+refs/*:refs/remotes/all/*'
> git checkout f8ea5bb
> ```
>
> Verified end to end: from that checkout the nine patches apply in glob order with **zero
> rejects** and reproduce our live production tree byte-for-byte. (Reported in
> [#1](https://github.com/allover326/deepseek-v4-cmp170hx/issues/1).)

The container installs vLLM with `pip install -e .`, so `/vllm/vllm/...` inside the image is
live source. You can therefore apply these by **bind-mounting the patched files** instead of
rebuilding — which is what [`launch/run-pp-dspark.sh`](../launch/run-pp-dspark.sh) does.

| # | file | what | why |
|---|---|---|---|
| 0001 | `model_executor/layers/sparse_attn_indexer.py` | add the missing `has_device_capability(90)` gate to `use_persistent_topk` | ⚠️ **Precautionary — the failure it guards against does NOT reproduce on this branch.** The original report (another CMP 170HX owner, vllm#50576) was that sm_80 selects a radix top-k returning wrong indices when the candidate count falls between k and 2k — prompt length 2049–4096 at `index_topk=512`/`compress_ratio=4` — emitting fluent-looking degenerate text. **That reporter has since retracted it for `dsv4-flash-a100`, and we could not reproduce it either.** We kept the patch because it costs nothing; see the V4 results (removed from RESULTS.md, which now covers V4.1 only). |
| 0002 | `config/speculative.py` | `draft_parallel_config.pipeline_parallel_size = 1` for dspark | The DSpark draft is **not** pipelined — the model runner builds it on the last PP rank only and it runs there whole. Inheriting the target's PP size makes `verify_with_parallel_config` demand `SupportsPP` from the *draft* architecture, which it neither implements nor needs. |
| 0003 | `v1/worker/gpu/pp_utils.py` | add `broadcast_draft()`, the matching receive, and sampled-token padding | This is **vLLM PR #46994**, which is not in upstream main. Without it, non-last pipeline ranks verify against a zero-initialised `req_states.draft_tokens` — acceptance near zero and corrupt output. The padding matters too: the receiver always posts a `max_sample_len`-wide buffer, so an unpadded narrow send is an element-count mismatch that deadlocks. |
| 0004 | `v1/worker/gpu/model_runner.py` | drop the dspark PP guard; call `broadcast_draft()` after `propose()`; scatter relayed draft tokens on non-last ranks | The guard covered eagle3/dflash/dspark; only dspark is enabled here — the other two are untested and their aux layers are spread across ranks rather than landing on one. |
| 0005 | `v1/worker/gpu/spec_decode/dspark/utils.py` | drop `NotImplementedError("DSpark does not support pipeline parallelism.")`; add `_has_real_weight()`; load the draft's token embedding from the checkpoint | Under PP the target's `embed_tokens` is a `PPMissingLayer` on the drafter's rank — and **aliasing one is a silent no-op, not an error**, hence the explicit check. The embedding (~1 GB) is read straight from `embed.weight` in the checkpoint, which avoids adding a cross-rank collective to model load. |
| **0005a** | `model_executor/layers/sparse_attn_indexer.py` **(must precede 0006)** | add `_prefill_topk_needs_torch_fallback()`, `_top_k_per_row_prefill_torch()`, and the prefill `if fallback / else CUDA kernel` branch | ★ **Patches 0001-0006 as first published were INCOMPLETE in two ways.** (1) Those two functions were called at four sites and defined nowhere. (2) 0006 does not *add* the fallback branch — it *rewrites* one, turning `if _prefill_topk_needs_torch_fallback():` into an `elif` and carrying `_top_k_per_row_prefill_torch(` as unchanged context — while the base `f8ea5bb` has a bare unconditional `ops.top_k_per_row_prefill(...)`. So supplying only the definitions is **not** enough. Reported by @fouvy, diagnosed by @snoby in [#1](https://github.com/allover326/deepseek-v4-cmp170hx/issues/1). Named `0005a` so a plain `patches/*.patch` glob applies it before 0006. **The fallback is ACTIVE on sm_80 by design — see below.** |
| **0006** | `model_executor/layers/sparse_attn_indexer.py` **(stacks on 0001)** | row-chunk the `[M, N]` float32 logits transient, gated by `DSV4_LOGITS_ROW_CHUNK` | ★ **The context-ceiling fix — ~134k → 1,047,736 tokens.** `fp8_mqa_logits_triton` allocates `logits = torch.empty((M, N), float32)` (`M` = prefill-chunk tokens, `N = seq_len / compress_ratio`) and hands the whole buffer to the top-k; it grows with context and is the largest allocation on the Triton fallback path. **Each row's top-k reads only its own `[ks, ke)`, so rows are independent and blocking them is exact, not approximate.** Default-OFF (`0` reproduces upstream byte-for-byte) because it is the same file as 0001 and you may want to bisect them. `256` reaches ~957,600; `128` reaches the full 1M. Costs nothing measurable — prefill 1,456 vs 1,448 tok/s at 4k, and the change is inside `if has_prefill:` so decode cannot be affected. |
| **0007** | `parser/deepseek_v4.py`, `parser/engine/parser_engine_config.py`, `parser/engine/streaming_parser_engine.py` (+ the PR's tests) | recover tool calls from `<｜DSML｜invoke>` blocks emitted with a missing or corrupted outer `<｜DSML｜tool_calls>` wrapper | **[vLLM PR #52645](https://github.com/vllm-project/vllm/pull/52645)**, backported to `f8ea5bb` — see the note below. Without it, one malformed wrapper feeds DSML markup back into agent context and sessions degrade into token soup. Python-only: bind-mountable, no rebuild. |
| **0008** | `v1/worker/gpu/spec_decode/rejection_sampler_utils.py` (+ the PR's regression test) | map NaN block maxima to `-inf` before `tl.argmax` in `_compute_global_target_argmax` and `_insert_resampled_kernel` | **[vLLM PR #50183](https://github.com/vllm-project/vllm/pull/50183)** verbatim (upstream `47a4e410b`, 2026-08-06); applies clean to `f8ea5bb`. Proposed as "0007" in [#10](https://github.com/allover326/deepseek-v4-cmp170hx/issues/10) before our 0007 existed — hence the number shift. **The DSpark corruption fix.** An all-NaN target-logits row makes `tl.argmax` return an out-of-range block index → OOB read → an *arbitrary token committed as if verified*. On sm_80 + fp8 KV the NaN rows are real, and the damage is invisible in prose while it shreds structured output (DSML tool calls, digit runs). Two branchless `tl.where` lines on already-loaded values; Triton/Python-only, bind-mountable, no rebuild. Verified: from pristine `f8ea5bb`, `0001 → 0008` apply in glob order with zero rejects; `tests/v1/spec_decode/test_rejection_sampler_utils.py` 30/31 on sm_80 — the one failure (`test_block_verification_accepts_at_least_as_many[5]`) fails identically on the pristine tree, so it is pre-existing and not caused by the guard. |
| **0009** | `models/deepseek_v4/**` (9 files + deletes `eager_scratch.py`), `csrc/libtorch_stable/*` (3 files), 3 kernel test files | **CUDA-graph/indexer corruption fixes**: (a) never take the indexer short-context shortcut while `torch.cuda.is_current_stream_capturing()`; (b) revert the DSv4 eager scratch pool (model-wide workspace reused across layers/streams without allocator lifetime tracking) | **Upstream [PR #52492](https://github.com/vllm-project/vllm/pull/52492) + [PR #52836](https://github.com/vllm-project/vllm/pull/52836)**, backported — root cause of the **intermittent multi-session corruption windows** (token salad / bad tool calls across ALL sessions at once, spec acceptance dipping ~30 s before visible corruption): capture-time CUDA-graph state disagreeing with runtime sparse-attention metadata, so attention reads wrong context slices. Matches [this DGX Spark analysis](https://forums.developer.nvidia.com/t/deepseek-v4-flash-on-2x-dgx-spark-intermittent-token-corruption-with-mtp-cuda-graphs/380889); our server auto-enables `VLLM_USE_BREAKABLE_CUDAGRAPH=1` (PIECEWISE capture) — exactly the configuration (a) fixes, and the eager scratch pool is created unconditionally on our config, which (b) removes. Hand-ported where the fork's sm_80 edits shifted context (`attention.py`, `fused_indexer_q.py`, `nvidia/model.py`, one test file). The `csrc/` part only *removes* the now-unused `_out` fused op — the Python falls back to the allocating variant already in the image, so **no rebuild needed**. PR #51318 from the same report is deliberately **not** carried: it touches `sparse_mla.py` (FlashMLA), which the sm_80 path (`DeepseekV4AmpereMLAAttention`) never uses. Verified: pristine `c3046d1` + `0002 → 0009` reproduce the live tree byte-for-byte; PR kernel tests **197 passed / 0 failed** on sm_80. Requires a whole-directory mount of `models/deepseek_v4` (the patch deletes a file). |

## Patch 0007 (PR #52645 DSML recovery) — partial backport, commits 1–4 only

0007 carries the PR **through `c848ab5`** (commits 1–4). The final upstream commit
`4f2aae2` (reasoning-adapter delegation) is **excluded**: its own
`TestDelegatingMalformedWrapperRecovery` regression tests fail when applied here —
`ParserEngine.finish_streaming` drops `_deferred_reasoning`, so a rolled-back
candidate loses its preceding newline (`"Still thinking.\n<invoke…"` comes back as
`"Still thinking.<invoke…"`). That is output corruption in exchange for pre-`</think>`
bare-invoke recovery, which is not the bug we have. The post-`</think>` corruption
cascade — the failure this patch exists to stop — is fully covered.

This matches the independent port referenced in the PR thread (randomvariable's), which
verified the same test failures on a clean worktree of `4f2aae2` against pure upstream.

Differences from the upstream diff, all forced by `f8ea5bb`'s older parser engine:

- `streaming_parser_engine.py` was hand-ported: the base predates the engine's
  `token_count` plumbing, `_in_skipped_tool_span`, and `_tool_exit_terminals`, so the
  recovery-hold machinery (`_begin/_advance/_abort/_clear_recovery_hold`,
  `_emit_for_state_now`, the `_apply_transition`/`_run_transition` split) is applied
  without token counting, and `_tool_exit_terminals` is added alongside.
- `abstract_parser.py` / `adapters.py` are **not** touched (those hunks are `4f2aae2`).
- The delegating test class is dropped from the test hunk (it tests `4f2aae2`).

**Verified:** from pristine `f8ea5bb`, `0001 → 0007` apply in glob order with zero
rejects and reproduce the live tree byte-for-byte. Tests (in `dsv4-a100:devel`):
**73/73** `tests/parser/engine/test_deepseek_v4.py`, **3746/3746** `tests/parser/engine/`.
Generated against `f8ea5bb`; applicability to the `c3046d1` base is untested.



`_prefill_topk_needs_torch_fallback()` returns **True on sm_80 deliberately**, and patch 0005a
must not be reduced to `return False`.

`ops.top_k_per_row_prefill`'s histogram path (taken by rows with more than `topk_tokens`
candidates) can leave part of its dynamic-shared-memory output uninitialised and copy it out
as indices; downstream, `compute_global_topk_indices_and_lens` treats any index `>= 0` as
valid and dereferences it into the KV block table. Upstream added this torch fallback for
SM12x in [vllm#49897](https://github.com/vllm-project/vllm/pull/49897); we enable it for SM8x
too because on 4x CMP 170HX it reproduces as **`Xid 31 MMU Fault ... ACCESS_TYPE_VIRT_WRITE`
killing a worker on prefills above roughly 128k tokens** (123k passes). Disabling it will look
fine until you go deep, then kill a worker with no obvious cause.

**This is unrelated to patch 0001.** 0001's `persistent_topk` gate is precautionary and its
originating report was retracted; **that retraction does not apply to 0005a**, which fixes a
different bug in a different kernel that we did reproduce on our own cards.

Overrides: `VLLM_DSV4_PREFILL_TOPK_TORCH=0` forces the CUDA kernel back on, `=1` forces the
fallback on any architecture.

**Verified end to end:** from a pristine checkout of `f8ea5bb`, applying `0001 → 0005a → 0006`
produces **zero rejects** and reproduces our live in-production file byte-for-byte
(md5 `96380027dd74c5913b6c4aeca6b25b02`). That check is what should have run before the first
release, and it is the only reason the second attempt at this fix was caught as incomplete.

**Ordering contract:** the downstream sparse-attention kernels iterate selected KV positions
in **ascending position order**, whereas `torch.topk` returns *score* order.
`_top_k_per_row_prefill_torch` sorts accordingly and pads short rows with `-1` at the tail. A
naive mask-then-`torch.topk` reimplementation compiles and runs but feeds wrongly ordered
indices downstream.

## Why DSpark-on-PP works at all

The model runner **already** builds the speculator on the last pipeline rank only:

```python
if self.is_last_pp_rank:
    self.speculator = init_speculator(self.vllm_config, self.device)
```

And the layer arithmetic cooperates. DeepSeek-V4-Flash has 43 layers; over PP4 they split:

```
rank 0: layers  0..10      rank 2: layers 22..32
rank 1: layers 11..21      rank 3: layers 33..42
```

DSpark taps `dspark_target_layer_ids = [40, 41, 42]` for its auxiliary hidden states, and
`lm_head` also lives on the last rank. **All three land on rank 3, together with the
drafter.** Only the token embedding (rank 0) is stranded, which patch 0005 handles.

That alignment is what makes the DSpark-on-PP work five small patches rather than a rewrite
(0006 is independent of it — it fixes the long-context ceiling). It is specific to
this model and this PP degree — a different layer count or a different `dspark_target_layer_ids`
could put the aux taps on a rank that has neither the drafter nor `lm_head`, and then the
auxiliary hidden states would need relaying across ranks too.

## Three guards, not one

Worth knowing if you port this further — they were found one at a time, and the third is the
one that costs an afternoon because its message blames the wrong model:

1. `v1/worker/gpu/spec_decode/dspark/utils.py` — `NotImplementedError: DSpark does not support pipeline parallelism.`
2. `v1/worker/gpu/model_runner.py` — `ValueError: {method} with pipeline parallel is not supported.`
3. `config/model.py` — `NotImplementedError: Pipeline parallelism is not supported for this model. Supported models implement the SupportsPP interface.` This fires at **config** time, on the **draft** architecture, and reads like a problem with the target model.
