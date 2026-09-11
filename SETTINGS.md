# Settings: `launch/run-v41-pp8.sh`

Every flag and variable the launch script sets, and the measurement behind it. Values are
the launch defaults; the environment variable in the second column overrides each.

## Parallelism and layout

| setting | override | why |
|---|---|---|
| `--pipeline-parallel-size 8` | | Eight cards, no P2P worth using: tensor parallel over Gen2 x4 measured 6.6× slower on prefill (V4), and with P2P enabled the pipeline hops did not get faster either (link-bound). |
| `VLLM_PP_LAYER_PARTITION=5,5,5,5,5,5,5,5` | env | 40 layers, five per rank. `5,5,5,5,5,5,6,4` doubles the KV pool at 0.90 but is inside the concurrent-decode scatter; a last rank with < 4 layers cannot start (drafter aux states come from layers 36–38). |
| device order `0,1,2,3,4,5,7,6` | `DSV41_GPUS` | Device order = rank order. The power-capped card takes the last rank, which draws the least. |
| `NCCL_*` passthrough | any `NCCL_*` in the environment | `NCCL_P2P_LEVEL=SYS` routes the hops GPU-to-GPU when the cmpunlocker P2P is enabled (no measured gain, but the path stays valid across the four root ports); `NCCL_DEBUG=INFO` shows the transport per hop. |

## Memory

| setting | override | why |
|---|---|---|
| `--gpu-memory-utilization 0.95` | `DSV41_MEMUTIL` | The KV pool is set by the last rank (52 GB weights + non-torch); 0.90 left it 3.3 GB → 3,151,289 tokens, 0.95 leaves 6.5 GB → **6,171,394 tokens** (5.9 × 1M). Capture and serving fit; ~3 GB physical headroom stays on that card. |
| `--max-model-len 1048576` | `DSV41_MAXLEN` | The model's `max_position_embeddings`. "2M context" means the KV pool (≥ 2M tokens), not one request. |
| `--block-size 128` | `DSV41_BLOCK_SIZE` | The indexer kernel block on non-SM90 is 128 and the layer-compact layout cannot split a 256-token manager block; 256 fails at KV allocation. |
| `--kv-cache-dtype fp8` | | `fp8_ds_mla`, 584 B per state. No FP4 KV on sm_80 and none in the checkpoint. |
| `--engram-config '{"storage":"cpu"}'` | `DSV41_ENGRAM` = `cpu` / `disk` / `gpu` | Both 98 GB tables pinned in host RAM (203 GB, needs patch 0013 to fit in 278 GB). Prefill +25–38 % and decode +10 % over `disk`. `disk` gathers the rows a step needs from the shards (`DSV41_ENGRAM_THREADS`, default 32, = NVMe queue depth) and runs on 32 GB; with `DSV41_WARM_ENGRAM=1` (default) `launch/warm-engram-cache.sh` reads both shards into the page cache after health, which recovers ~80 % of the `cpu` gain on hosts with 210–240 GB. `gpu` would need a 98 GB table next to a card's layers: impossible here. |
| `DSV4_LOGITS_ROW_CHUNK=64` | `DSV4_ROW_CHUNK` | Row-chunks the indexer's `[rows, N]` fp32 logits on the four full-context index layers (8 GB per chunk at N = 1M un-chunked). With patch 0011 the other four layers no longer need it. |
| `--shm-size=16g` | | Multiprocess workers. |

## Batching and scheduling

| setting | override | why |
|---|---|---|
| `--max-num-batched-tokens 4096` | | Prefill chunk. 4096 halves the number of Engram gathers per prompt versus 2048; 8192 untested. Concurrent prefill already saturates the cards (92–96 %) at this value. |
| `--max-num-seqs 8` | `DSV41_MAX_SEQS` | Bounds the **total** number of running requests, not requests per micro-batch (1 serialises users: "Running: 1, Waiting: 7"). Keep ≥ the concurrency you serve. |
| `--speculative-config '{"method":"dspark","num_speculative_tokens":5}'` | `--plain` disables | `dspark_block_size = 5` in the checkpoint; up to 6 tokens per step. Acceptance 3.9–5.9 tokens per step depending on the text; step rate ~16/s. |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=1` | `DSV4_BREAKABLE_CUDAGRAPH` | V4.1 is not torch-compiled (attention runs in an eager break); without breakable graphs there are no piecewise graphs and capture refuses `FULL_AND_PIECEWISE`. Two capture rounds per rank: target and drafter. |
| `VLLM_DSV41_CAND_LOGITS` | env, default on | Kill switch for patch 0011 (candidate-only indexer scoring on layers 24/28/32/36). `0` restores full-context scoring and masking: identical output, slower, ~10 % smaller KV pool. |
| `--no-enable-flashinfer-autotune` | | No FlashInfer on sm_80. |
| `--attention-backend` | unset | sm_80 auto-selects `TRITON_MLA_SPARSE_DSV41`; anything else is rejected. |

## API surface

| setting | why |
|---|---|
| `--enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41` | The PR's V4.1 parsers plus patches 0003–0006 and 0012 (DSML recovery, lenient wrapper spelling). Verified through a proxy. |
| `--enable-prompt-tokens-details` | `prompt_tokens_details.cached_tokens` in `usage`; without it clients never see prefix-cache hits. |
| `--limit-mm-per-prompt '{"image":8}'` (`DSV41_MM_LIMIT`) | The checkpoint's vision encoder is loaded and works through the OpenAI `image_url` content type; vLLM's default of one image per request is too few for agent screenshots. |
| `--served-model-name dsv41`, port 8099 (`DSV41_PORT`) | |
| `reasoning_effort` per request | A numeric budget 1–100 rendered into the system prompt. This server maps `low`/`high`/`xhigh`/`max` to 25/50/75/100 (the checkpoint's own encoding maps `low`/`high`/`max` to 50/75/100), `none` disables thinking, `minimal` and `medium` are rejected (HTTP 400). Default when omitted: `high`. Integers are the precise form. |

## Bind mounts and image

| setting | why |
|---|---|
| `DSV41_VLLM_SRC` | The `vllm/` directory of a checkout with all `patches/v41` applied. The image holds patches 0001–0008 only; every `vllm/*.py` that differs from `DSV41_IMG_COMMIT` (default `68de681be3`, the image's commit) is bind-mounted over the image. **Required**: without 0009 the engine does not start. The script warns when unset. |
| `DSV41_IMAGE` | Default `zanooda/vllm-sm80-ds41f:v41-sm80` (sm_80-only SASS). Rebuild from the full series with `docker/vastai-build-push-v41.sh` to make the mounts unnecessary. |
| `HF_HUB_OFFLINE=1`, `VLLM_WORKER_MULTIPROC_METHOD=spawn` | Local checkpoint only; spawn for the CUDA workers. |

## Deliberately not set

- `--enforce-eager`: 8–10 tok/s on V4; never.
- Tensor parallel in any form.
- `--max-num-partial-prefills` / `long_prefill_token_threshold`: not needed, concurrent
  prefill pipelines and saturates the cards as is.
- A software power cap check in the script: the capped card is a property of this box, not
  of the stack (see RESULTS.md, hardware notes).
