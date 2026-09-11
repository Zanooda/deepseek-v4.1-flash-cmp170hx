# Every setting, and why it has that value

The full command the numbers in [RESULTS.md](RESULTS.md) were produced with:

```bash
docker run -d --name dsv4 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -v /path/to/DeepSeek-V4-Flash-0731:/model \
  -v $R/config/speculative.py:/vllm/vllm/config/speculative.py:ro \
  -v $R/v1/worker/gpu/pp_utils.py:/vllm/vllm/v1/worker/gpu/pp_utils.py:ro \
  -v $R/v1/worker/gpu/model_runner.py:/vllm/vllm/v1/worker/gpu/model_runner.py:ro \
  -v $R/v1/worker/gpu/spec_decode/dspark/utils.py:/vllm/vllm/v1/worker/gpu/spec_decode/dspark/utils.py:ro \
  -v $R/model_executor/layers/sparse_attn_indexer.py:/vllm/vllm/model_executor/layers/sparse_attn_indexer.py:ro \
  -v $R/parser/deepseek_v4.py:/vllm/vllm/parser/deepseek_v4.py:ro \
  -v $R/parser/engine/parser_engine_config.py:/vllm/vllm/parser/engine/parser_engine_config.py:ro \
  -v $R/parser/engine/streaming_parser_engine.py:/vllm/vllm/parser/engine/streaming_parser_engine.py:ro \
  -v $R/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:ro \
  --shm-size=16g -p 8098:8000 \
  dsv4-a100:devel vllm serve /model --served-model-name dsv4s \
  --pipeline-parallel-size 4 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-model-len 32768 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 8 \
  --trust-remote-code \
  --no-enable-flashinfer-autotune \
  --tokenizer-mode deepseek_v4 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}'
```

`$R` is the `vllm/` directory of your patched checkout. The image installs vLLM with
`pip install -e .`, so `/vllm/vllm/...` is live code — bind-mounting the nine patched
files applies them with **no rebuild**.

---

## The settings that matter most

### `--pipeline-parallel-size 4`  ← not tensor parallel
The single highest-impact choice. On these cards PP beats TP by **6.6× on prefill**
(5,321 vs 801 tok/s at 77k context) and roughly 2× on aggregate decode.

Why: tensor parallel performs **2 all-reduces per layer × 43 layers = 86 collectives**
per forward, each moving `tokens × hidden` bytes. Double the tokens and you double both
compute and communication, so a communication-bound setup stays communication-bound at
*every* sequence length — which is exactly why TP prefill measures flat at ~800 tok/s
from 1.5k to 77k tokens. Pipeline parallel moves the same payload only **3 times** (one
hand-off per stage boundary) and overlaps it with compute: **~28× less data on the wire**.

The CMP 170HX is PCIe **Gen2 x4 with no P2P** (~1.0 GB/s measured bus bandwidth), which
is close to the worst case for TP. On a machine with NVLink the trade-off would look
completely different.

### `--speculative-config '{"method":"dspark","num_speculative_tokens":5}'`
- **`dspark`, not `mtp`.** DeepSeek-V4-Flash-0731 replaced the single MTP layer with a
  3-layer DSpark stack (`mtp.{0,1,2}` + `markov_head` + `confidence_head`).
  `method:"mtp"` KeyErrors on `mtp_block.main_norm`.
- **5, and 5 exactly.** vLLM enforces `num_speculative_tokens >= dspark_block_size`
  (5 for this checkpoint) — below that the block/Markov machinery gets an unsupported
  layout and produces *garbled output*, not merely lower acceptance. Above is worse in
  practice: **7 measured 60.3 tok/s vs 5's 98.1**, because acceptance never extends past
  ~3 tokens so the extra drafts are pure waste.
- Worth **+1.93×** decode over the same config without it, and unlike on TP it keeps
  winning under load (see RESULTS).

### `--gpu-memory-utilization 0.85`
Not higher. Raising it takes headroom away from activations and CUDA-graph capture; at
**0.90 with the DSpark draft resident, capture OOMs**. The KV pool you would gain is
memory you cannot spend — at the ~128k usable ceiling a single context costs only about
6 GiB, and the pool at 0.85 already reports 1.8M tokens at `max-model-len 131072`.

### `--max-num-batched-tokens 2048`
Do **not** lower it: without it, activation memory during profiling eats the KV pool and
the engine dies with "No available memory for the cache blocks".

Raising it is pointless — measured twice, on two different stacks. 2048 → 8192 moved
prefill by ~4% (890 → 924 tok/s) and **halved the KV pool** (58,538 → 29,867 tokens).
Prefill is not chunk-size bound.

### `--kv-cache-dtype fp8`
Effectively required. DeepSeek-V4 asserts `fp8_ds_mla layout only supports fp8 kv-cache`.
FP8 KV does work correctly on sm_80 — the conversion is software; only FP8 *math* needs
sm_89+.

### `--no-enable-flashinfer-autotune`
FlashInfer's JIT runs at engine init and is a **hard requirement** (removing the package
gives `ModuleNotFoundError`), so the container must be able to compile it. See
[docker/Dockerfile.devel](docker/Dockerfile.devel) for why the base image needs a real
CUDA toolkit rather than pip CUDA wheels.

### `--tokenizer-mode deepseek_v4`
From the working launch command in
[vllm#50576](https://github.com/vllm-project/vllm/issues/50576).

### `--max-model-len` — ★ set it to what you need, up to the model maximum
**Previous versions of this file said a larger value gives you LESS usable context. That was a
symptom of the [logits-buffer bug](RESULTS.md#-context-ceiling--solved) and is withdrawn.**
With `DSV4_LOGITS_ROW_CHUNK` set, the full 1,048,576 works:

| `--max-model-len` | highest verified prompt | `DSV4_LOGITS_ROW_CHUNK` |
|---|---|---|
| 393,216 | 388,505 (one-shot) | 256 |
| **1,048,576** (model max) | **1,047,736** (one-shot) | **128** |
| **1,048,576** | **1,002,852** (**405-turn conversation**) | **64** |

⚠️ **Those first two rows are one-shot prefills.** A long *conversation* on `128` dies at
~718–733k with a CUDA illegal memory access — see
[accumulated vs one-shot](RESULTS.md#accumulated-conversation--one-shot-prefill). Chunk size costs
almost nothing (TTFT 7.48 s at `64` vs 7.08 s at `128`, measured at 750k), so **if in doubt use
64**.

The only reason not to set the maximum is **time**, not capacity: TTFT at 1M is **9.2 minutes**
and decode drops to ~40 tok/s. Pick the profile that matches your workload —

- everyday, ≤388k, one-shot documents → `--max-model-len 393216`, `DSV4_LOGITS_ROW_CHUNK=256`
- full 1M, one-shot documents → `--max-model-len 1048576`, `DSV4_LOGITS_ROW_CHUNK=128`
- ★ **long conversations / agents (any max-model-len) → `DSV4_LOGITS_ROW_CHUNK=64`**

⚠️ **None of these settings change retrieval accuracy**, which degrades with depth regardless
(~100% at 150k → ~30% at 900k). That is architectural, not a tuning problem. See
[RESULTS](RESULTS.md#retrieval-accuracy-vs-depth--the-window-is-reachable-not-uniformly-usable).

### `--reasoning-parser deepseek_v4` — ★ set it even if you never enable thinking

`thinking=False` is the default on 0731, so most people never see this. But the moment thinking is
on, the `<think>` delimiters are **special tokens** and get stripped on decode — so **without this
flag the reasoning text arrives inside `content` with nothing marking it as reasoning**, and replies
literally begin *"We need answer classic. Need be careful. User asks…"*. With the flag it lands in
its own field and `content` stays a clean answer. The flag is inert when thinking is off, so there
is no reason not to set it.

⚠️ On this build the returned field is **`reasoning`**, not `reasoning_content`. A client reading
only the latter sees zero thinking and cannot distinguish a thinking run from a non-thinking one.

**Enabling thinking** (worth it — [+19 points of long-context recall](RESULTS.md#thinking-recovers-a-lot-of-the-lost-long-context-recall--the-effort-level-does-not)):
per request via `"chat_template_kwargs": {"thinking": true, "reasoning_effort": "high"}`, or
server-side with `--default-chat-template-kwargs`. **Use `high`, not `max`** — `max` thinks 2.7×
harder for identical recall. A top-level `reasoning_effort` body param is the ambiguous path; it
changes behaviour but `/tokenize` cannot see it.

### `--max-num-seqs 8`
Raise for serving. 128 was used for the concurrency sweeps and behaves well; DSpark keeps
winning all the way to 64 concurrent requests on PP.

### `--block-size 256`, `--trust-remote-code`
Standard for this model.

---

## Environment variables

| var | value | why |
|---|---|---|
| `NVIDIA_VISIBLE_DEVICES` | `0,1,2,3` | Four cards. **Three also works** (e.g. `1,2,3`) with `VLLM_PP_LAYER_PARTITION=15,15,13` — see [RESULTS](RESULTS.md#three-or-four-cards). An earlier version of this table said three does not work; that was wrong. |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | Required; fork deadlocks with CUDA already initialised. |
| `HF_HUB_OFFLINE` | `1` | Local weights only; avoids a hub call on every start. |
| `CUDA_HOME` | set in image | The devel image sets `/usr/local/cuda`. |
| **`DSV4_LOGITS_ROW_CHUNK`** | **`64`** for conversations; `256`/`128` for one-shot | ★ **The context-ceiling fix** ([patch 0006](patches/0006-logits-row-chunk.patch)). Row-chunks the sparse indexer's `[M, N]` float32 logits transient. `0` = original upstream path, which dies at ~134k. **One-shot prefill:** `256` reaches ~957,600, `128` reaches 1,047,736. **Accumulating conversation:** `128` dies at ~718–733k (reproduced twice); **`64` reached 1,002,852 over 405 turns.** Costs almost nothing (TTFT 7.48 s vs 7.08 s at 750k). Affects only whether it crashes — **not** retrieval accuracy. |
| **`VLLM_PP_LAYER_PARTITION`** | `12,12,12,7` (4 cards) · `15,15,13` (3 cards) | Rebalances the 43 layers off pipeline rank 3, which uniquely carries `lm_head` **and** the DSpark drafter. **Does not affect the context ceiling** — but it removes an 8.7 GiB rank imbalance and grows the KV pool **~85%** (798,660 → 1,476,563 at `max-model-len 163840`). Must have one entry per pipeline rank, summing to 43. **On 3 cards this is required, not optional** — the default `[15,14,14]` fails during the Marlin FP4 expert repack because the last rank also carries `lm_head` and the DSpark drafter. |

⚠️ **Do not bother with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — it is a hard
failure at model load on these cards: `expandable_segments: memory mapping failed with OOM on
device 3 while trying to map 20971520 bytes (free: 28626452480)`, i.e. it cannot map 20 MB
with 28.6 GiB free. CUDA VMM appears broken on GA100 CMP parts.

`--shm-size=16g` on the container — the default 64 MB is not enough for multiprocess workers.

---

## Benchmarking-only flags

Not for production, but required to reproduce the numbers:

- **`--no-enable-prefix-caching`** — without it, repeated benchmark prompts hit the prefix
  cache and prefill numbers become fiction (a 100k-token prompt appearing to prefill in
  0.5 s was how this was caught).
- **`ignore_eos: true`** in the request — otherwise the speculative and non-speculative
  configs generate *different numbers of tokens* and the comparison is meaningless.
- **`stream_options: {include_usage: true}`** — under speculative decoding a single SSE
  chunk carries several tokens (roughly the acceptance length), so a harness that counts
  chunks under-reports by that factor. Rate against the server's own `completion_tokens`.
- **Discard the first request after boot** — it carries Triton JIT compilation and reads
  roughly 4× low.

---

## Things deliberately NOT set

| flag | why not |
|---|---|
| `--enforce-eager` | **Never.** 8–10 tok/s, i.e. worse than no speculative decoding at all. CUDA graphs are worth ~12×. |
| `--tensor-parallel-size` | PP wins on this hardware; see above. TP3 is also arithmetically impossible (64 heads, 256 experts don't divide by 3). |
| `--quantization` | The checkpoint's format is auto-detected. Forcing it breaks MoE scale loading. |
| `VLLM_SM86_SPLITK=0` | Helped ~8% on the *older* overlay stack at low TP. Untested on this branch; the kernel it targets may not be in this path. |


---

## DeepSeek-V4.1-Flash (`launch/run-v41-pp8.sh`)

The V4.1 launch is the V4 one with these differences. Every value below was measured on the
2026-09-10/11 runs unless marked otherwise.

| flag / env | value | why |
|---|---|---|
| `--pipeline-parallel-size 8` | 8 cards | ~327 GB of resident weights need eight 64 GB cards; PP for the same PCIe reasons as V4 |
| `--engram-config '{"storage":"cpu"}'` (`DSV41_ENGRAM`) | **cpu** since the 278 GB RAM upgrade (2026-09-11) | the two Engram tables (98 GB fp8 + 3 GB scales each) pinned in host RAM on the ranks owning layers 1 and 14, gathered by the GPU over UVA. Needs patch 0013 to fit (see patches/README); steady ~217 GB used. Measured vs `disk`: prefill +25–38 % (128k TTFT 21–23 s → 17.3 s), decode +10 %. `disk` remains for small-RAM hosts: rows are read from the shards per step (`disk_threads` = NVMe queue depth), and `launch/warm-engram-cache.sh` pulls the shards into the page cache when RAM allows (roughly the `cpu` speed, evictable) |
| `VLLM_PP_LAYER_PARTITION` | `5,5,5,5,5,5,5,5` | 40 backbone layers; the last rank also holds lm_head and the 3-layer DSpark drafter (~9 GB). Measured 2026-09-11: `5,5,5,5,5,5,6,4` gives a 2.2× KV pool (6.86M tokens) but −10 % concurrent decode; a last rank with < 4 layers cannot start (drafter aux states come from layers 36–38) |
| `NCCL_P2P_LEVEL=SYS` | set when P2P is enabled | with the cmpunlocker BAR1 P2P the pipeline hops go `P2P/CUMEM` instead of through host memory; measured no speed change (link-bound), but it is the path that stays valid across the four root ports. Any `NCCL_*` variable in the launcher's environment is passed into the container |
| `DSV41_GPUS` | `0,1,2,3,4,5,7,6` on this host | device order = rank order; GPU 6 (`c3:00.0`) is power-capped to 180 W (falls off the bus at stock), so it takes the last rank, which draws the least power |
| `--max-model-len 1048576` | model maximum | `max_position_embeddings`; "2M" is pool capacity, not a request length (README) |
| `--max-num-batched-tokens 4096` | 4096 | V4 showed prefill is not chunk-bound; 4096 halves the number of disk-Engram gathers per prompt vs 2048 |
| `DSV4_LOGITS_ROW_CHUNK` | `64` | same transient as V4, now on the ratio-1 layers with N = 1M; also chunks the candidate-block select/mask |
| `--speculative-config '{"method":"dspark","num_speculative_tokens":5}'` | 5 | `dspark_block_size=5` in the V4.1 config; V4.1 has no plain MTP method |
| `--tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41` | | the PR's V4.1 parsers; the tokenizer mode auto-selects `deepseek_v41` |
| `--kv-cache-dtype fp8` | fp8 | fp8_ds_mla layout (584 B/state); no FP4 KV exists in the checkpoint or on sm_80 |
| `--block-size 128` | 128, not 256 | the indexer's kernel block is 128 on non-SM90 and the layer-compact layout cannot split a 256-token manager block into two; 256 fails at KV allocation with a clear message |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | `1` | V4.1 is not torch-compiled (attention runs in an eager break); without breakable graphs there are no piecewise graphs at all and capture refuses `FULL_AND_PIECEWISE`. The V4 stack ran with them off. |
| `DSV41_VLLM_SRC` | path to a `v41-sm80` checkout's `vllm/` dir | until the image is rebuilt with patch 0009, the launch script bind-mounts every Python file changed since the image's commit (`DSV41_IMG_COMMIT`, default `68de681be3`) |
| `VLLM_DSV41_CAND_LOGITS` | unset (= on) | patch 0011: index layers 24/28/32/36 score only the candidate blocks. `0` restores full-context scoring + mask (identical output, slower, ~10 % smaller KV pool) |
| `--gpu-memory-utilization 0.95` | 0.95 (was 0.90) | the KV pool is decided by the last rank (52 GiB of weights + drafter + lm_head + non-torch); 0.90 left it 3.3 GiB → 3,151,289 tokens, 0.95 leaves 6.5 GiB → **6,171,394 tokens** (5.9 × 1M), measured 2026-09-11 with decode unchanged. Capture and serving fit; ~3 GB physical headroom remains on that card |
| `--enable-prompt-tokens-details` | on | adds `prompt_tokens_details.cached_tokens` to `usage`; without it clients such as litellm never see prefix-cache hits and their cache-read price never applies |
| `--limit-mm-per-prompt '{"image":8}'` | 8 (`DSV41_MM_LIMIT`) | the checkpoint's V4.1 vision encoder (32 layers, patch 14) is loaded and works through the OpenAI image_url content type; the vLLM default of one image per request is too few for agent screenshots |
| `--max-num-seqs 8` | 8 (`DSV41_MAX_SEQS`) | bounds the **total** running requests, not requests per micro-batch: `1` serialises users (measured: Running 1, Waiting 7). Keep ≥ the concurrency you serve |
| `--attention-backend` | unset | SM8x auto-routes to `TRITON_MLA_SPARSE_DSV41`; anything else is rejected with a clear error |
