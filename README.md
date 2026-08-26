# DeepSeek-V4-Flash on 4× CMP 170HX

Running **DeepSeek-V4-Flash-0731** (~284B total / ~13B active) on four **NVIDIA CMP 170HX**
mining cards — GA100 silicon, sm_80, VRAM-unlocked to 64 GB, PCIe Gen2 x4, no P2P.

**98 tok/s decode · ~5,300 tok/s prefill · 1,047,736 verified context — the model's full 1M.**

> ⚠️ **If you run long *conversations*, set `DSV4_LOGITS_ROW_CHUNK=64`, not 128.** The 1M figure
> above is a **single one-shot prefill**. A real accumulating chat on the same config dies at
> **~718–733k** with a CUDA illegal memory access (reproduced twice). At `64` a 405-turn
> conversation reached **1,002,852 tokens** clean. See
> [accumulated vs one-shot](RESULTS.md#accumulated-conversation--one-shot-prefill).
> Note also that **retrieval accuracy degrades well before any crash** — ~100% at 150k, **~30% at
> 900k**. The window loads and runs; it is not uniformly usable.

For scale: a single DGX Spark does ~14 tok/s on this class of model, and a dual-Spark setup
with speculative decoding reports 55–67.

The 170HX is an ex-mining card with no display output and a fused-down PCIe link, but it is
GA100 silicon with 64 GB of HBM2e at ~1.6 TB/s once unlocked. Pricing has moved a lot as
people found LLM uses for them — around **$1,100–1,200 per card** as of August 2026, so
budget for four accordingly.

This repo contains the patches, container build, launch scripts and benchmark harnesses —
plus [every setting and why it has that value](SETTINGS.md), and the
[full measured results](RESULTS.md) including the things that *don't* work.

---

## What is actually new here

**DSpark speculative decoding running under pipeline parallelism.** vLLM refuses this
combination in three separate places. Enabling it is worth **1.93×** and — unlike on tensor
parallel, where speculation goes *negative* above ~8 concurrent requests — it keeps winning
all the way to 64 concurrent on PP.

Everything else in this repo is packaging: a working container build, the launch settings,
and honest benchmarks.

Credit where it's due: the SM8x DeepSeek-V4 backend is
[haosdent/vllm@dsv4-flash-a100](https://github.com/haosdent/vllm/tree/dsv4-flash-a100),
built on work discussed in [vllm#50576](https://github.com/vllm-project/vllm/issues/50576).
This repo sits on top of that branch.

> **2026-08-13 — recommended base moved from `f8ea5bb` to `c3046d1`** (+7% decode measured,
> p<0.001; patch 0001 is now upstream; the range touches `csrc/` so it needs a
> [full source build](docker/Dockerfile.fullbuild), and the base commit is only recoverable by
> [tarball](patches/README.md#getting-c3046d1--it-is-unreachable-by-any-git-method) after
> upstream's second force-push). Details and the honest numbers — including why a circulated
> "+30%" claim does not reproduce — in [RESULTS](RESULTS.md#rebase-to-c3046d1-2026-08-13).

---

## Requirements

- **3 or 4 CMP 170HX** (or other 64 GB sm_80 cards). Most numbers on this page were taken on 4.
  **3 cards also works** — you must set `VLLM_PP_LAYER_PARTITION=15,15,13`, because the last
  pipeline rank also carries `lm_head` and the DSpark drafter. ⚠️ An earlier version of this
  page said 3 cards could not work; that was **wrong** and is retracted in
  [RESULTS](RESULTS.md#three-or-four-cards). 2 cards cannot hold ~155 GiB of weights.
- Cards must be **VRAM-unlocked** — `nvidia-smi` should report 65,536 MiB, not 8,192 MiB.
- Check the power-brake diagnostic below before benchmarking anything. It is
  motherboard-specific and most people will not hit it, but it costs ~4× if you do.
- The original `deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint (**~155.4 GiB / 166.9 GB** on disk; `du` under-reports it as ~140 GB). The
  INT4/compressed-tensors repack is **not** needed; this branch reads the native
  MXFP4+FP8 weights.
- Docker with the NVIDIA runtime.

## Quick start

```bash
# Easiest: pull the prebuilt c3046d1 image (patches 0002-0008 baked in)
docker pull zanooda/vllm-sm80:c3046d1
docker tag zanooda/vllm-sm80:c3046d1 dsv4-a100:devel

deepseek-v4-cmp170hx/launch/run-pp-dspark.sh   # sources default to ~/vllm/vllm
```

Building it yourself instead — note the base is `c3046d1`, which touches `csrc/`, so this is a
**full CUDA source build (~2-4 h, needs >= 48 GB RAM)**; the old 10-minute `Dockerfile.devel`
flow cannot deliver the kernel changes. There is also a self-contained remote-build script
([docker/vastai-build-push.sh](docker/vastai-build-push.sh)) that does all of the below and
pushes to a registry:

```bash
git clone https://github.com/haosdent/vllm.git && cd vllm

# c3046d1 is NOT reachable by any git method (force-pushed, referenced by nothing).
# Reconstruct it from the tarball and verify the tree SHA -- full procedure and
# the safety story in patches/README.md:
curl -sL -o /tmp/c3046d1.tar.gz \
  https://codeload.github.com/haosdent/vllm/tar.gz/c3046d1ebd2dae9b94ad2ef5f966ea153632251e
# ... reconstruct + verify per patches/README.md, then:
git checkout -B rebase-c3046d1 c3046d1-recon

# patches 0002 -> 0008 (0001 is dropped -- its gate is upstream in c3046d1)
for p in ../deepseek-v4-cmp170hx/patches/000{2,3,4,5,5a,6,7,8}*.patch; do patch -p1 < "$p"; done

docker build -f ../deepseek-v4-cmp170hx/docker/Dockerfile.fullbuild -t dsv4-a100:devel .

../deepseek-v4-cmp170hx/launch/run-pp-dspark.sh
```

Two build traps worth knowing before you start:

- **Build from inside the vLLM checkout.** If your model weights live in a parent directory,
  a build rooted there will try to ship hundreds of GB to the Docker daemon as build context.
- **The base image needs a real CUDA toolkit.** `python:3.12-slim` plus pip CUDA wheels gives
  `nvcc` 13.3 against FlashInfer's bundled headers for 13.0, and FlashInfer's JIT is a hard
  requirement at engine init. Both Dockerfiles use `nvidia/cuda:13.0.2-cudnn-devel` with a
  venv, which is why they work.
- **RAM.** The full build OOMs on 30 GB even at `MAX_JOBS=16`; it is capped at 8 for a
  reason. Give it >= 48 GB or add swap before raising it.

On the *old* `f8ea5bb` base the SM8x work was Python/Triton-only, so `Dockerfile.devel`
(`VLLM_USE_PRECOMPILED=1`) turned the build into a ~10-minute download. That no longer
holds on `c3046d1`: the base itself changes `csrc/`, and precompiled wheels physically
cannot contain those kernels.

---

## Results at a glance

| | plain | **+ DSpark** |
|---|---|---|
| decode, single stream | 50.8 | **98.1 tok/s** |
| decode @ 64 concurrent | 472.0 | **712.8 tok/s** |
| decode @ 100k context | 38.8 | **90.0 tok/s** |
| prefill (25k–77k context) | ~5,300 | ~5,200 tok/s |
| time to first token @ 100k | 14.6 s | 14.6 s |

**Use pipeline parallel, not tensor parallel.** On this hardware PP beats TP by **6.6× on
prefill** — TP measures flat at ~800 tok/s from 1.5k to 77k tokens, because it performs 86
all-reduces per forward pass on a PCIe Gen2 x4 link with no P2P. PP moves the same data 3
times. Full reasoning in [SETTINGS.md](SETTINGS.md#--pipeline-parallel-size-4--not-tensor-parallel).

Full tables, correctness testing and limits: **[RESULTS.md](RESULTS.md)**.

---

## Troubleshooting: cards running ~4× slow (`PWRBRK#` / edge pin B30)

**Check this first if your throughput is nowhere near the numbers here.** It is not a
property of the CMP 170HX — it is a motherboard behaviour, and most boards do not do it.

`PWRBRK#` is an optional PCIe sideband signal on **edge pin B30** that lets a platform force
GPUs into an emergency low-power state. Some workstation boards assert it. On an
**ASUS Pro WS WRX80E-SAGE** (the board these results were produced on) it is asserted on the
x16 slots, which pins the card in a permanent hardware power brake:

| | braked | healthy |
|---|---|---|
| power draw | ~88 W of a 250 W budget | 105–180 W |
| clocks | 1140 MHz | ~1400 MHz |
| fp16 | 39.3 TFLOPS | **155.7 TFLOPS** |
| memory bandwidth | ~608 GB/s | ~1355 GB/s |

### Diagnosing it

```bash
nvidia-smi -q | grep -A1 "HW Power Brake Slowdown"
```

`Active` on a card that is not thermally or power limited means the platform is asserting
`PWRBRK#`. To count healthy cards across a 4-GPU box:

```bash
nvidia-smi -q | grep -c "HW Power Brake Slowdown  *: Not Active"   # want 4
```

This is worth ruling out early because it looks exactly like "these mining cards are just
slow" — the cards are fine, and it cost a great deal of time here before being identified.
A PCIe riser was what proved it: the same card in the same slot ran at full speed through a
riser that does not carry B30.

### Fixing it

- **Kapton tape over pin B30** on the card's edge connector (B30 is on the B-side, counted
  from the notch end). This is the usual fix.
- **A riser that does not route B30**, which also works and requires no modification.
- **A BIOS/BMC option**, if your board exposes one — many do not.

If your board does not assert `PWRBRK#`, do nothing. Taping a pin that was never being
driven gains you nothing and risks damaging the contact.

## Known limits

- ~~**Context ceiling ~150k, and it scales INVERSELY with `--max-model-len`.**~~ **FIXED —
  see [patch 0006](patches/README.md).** The ceiling was the sparse indexer's `[M, N]` float32
  logits transient; row-chunking it (`DSV4_LOGITS_ROW_CHUNK`) lifts the verified context from
  ~150k to **1,047,736 tokens**, the model's full 1M. The inverse-scaling advice that used to
  be here was a symptom of that bug and **no longer applies** — set `--max-model-len` to what
  you need. Costs nothing measurable; see [RESULTS](RESULTS.md#-context-ceiling--solved).
- ★ **An accumulating conversation has a LOWER ceiling than a one-shot prefill, and
  `DSV4_LOGITS_ROW_CHUNK` moves it.** At `128` a one-shot prefill reaches 1,047,736 but a real
  multi-turn chat dies at **~718–733k** (CUDA illegal memory access in
  `_top_k_per_row_prefill_torch`, always on the same PP rank, reproduced twice). At `64` a
  405-turn chat reached **1,002,852** clean. **Use 64 for conversational workloads.** A one-shot
  prefill at the same depth is unaffected — this is specific to the prefix-cached path, which
  needle tests never exercise. See
  [accumulated vs one-shot](RESULTS.md#accumulated-conversation--one-shot-prefill).
- ★ **Retrieval accuracy degrades with depth, long before any crash.** Facts planted at known
  positions and re-queried: **100% at 150k, 87% at 300k, 60% at 450k, 50% at 750k, 30% at 900k.**
  This is *not* a bug in these patches — chunk size does not change it (identical at matched
  depth) and no prompt fixes it. `index_topk` is a fixed 512 while the candidate pool grows
  linearly with context, so the fraction of context reachable falls ~6× from 150k to 900k.
  **Treat 1M as a large working set, not a reliable database.** Keep retrieval-critical context
  under ~150k, or re-inject facts that must survive.
- **Long context is slow, not free.** TTFT at 1M is **9.2 minutes** and decode drops to
  **35.6 tok/s**. Prefill is the expensive half — at the top of its range this is a
  batch/document tool, not an interactive one. Both curves are in
  [RESULTS](RESULTS.md#speed-vs-context).
- **DSpark output is not reproducible** at temperature 0, including run-to-run on the same
  server. Verified to be a property of DSpark itself, not of these patches (the stock
  upstream tensor-parallel path behaves the same way). Quality is unaffected in every test
  here. Run without `--speculative-config` if you need determinism.
- **Never use `--enforce-eager`** — 8–10 tok/s, worse than no speculation at all.

## Repo layout

```
patches/     8 patches (0002-0008 on c3046d1; 0001 only on the legacy f8ea5bb base) — see patches/README.md
docker/      Dockerfile.fullbuild (source build, c3046d1), Dockerfile.devel (precompiled, f8ea5bb-era),
             vastai-build-push.sh (remote build + registry push)
launch/      run-pp-dspark.sh (best config) and run-a100.sh (tensor-parallel variant)
bench/       the 8 harnesses every number in RESULTS.md came from
SETTINGS.md  every flag and why it has that value
RESULTS.md   measured results, correctness testing, limits, measurement pitfalls
```

## Contributing back

Patches 2–5 are small and general. Pipeline parallel was flagged as *the* uncovered
configuration in [vllm#50576](https://github.com/vllm-project/vllm/issues/50576), and patch 3
is essentially vLLM PR #46994, which upstream still needs. If you have hardware to test on —
especially a non-170HX sm_80 box such as an A100 or A800 — results either way are useful.

## License

Apache-2.0, matching vLLM. The patches are derivative of vLLM and of
haosdent/vllm@dsv4-flash-a100.
