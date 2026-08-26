#!/bin/bash
# Two Qwen3.8-27B instances on the two spare CMP 170HX cards (GPU 4 + 5),
# alongside the DeepSeek PP4 instance on GPUs 0-3 (launch/run-pp-dspark.sh).
# Official upstream vLLM image (the DeepSeek fork is not needed for Qwen).
#
# Model: Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8 -- W8A8 (weights AND
# activations INT8) keeps the native int8 tensor-core GEMM path on Ampere
# (weight-only W8A16 falls back to Marlin bf16 dequant, ~2x slower prefill);
# grafted BF16 MTP head gives working speculative decoding on sm_80 (the
# DFlash2 draft crashes with a device-side assert on GA100 -- do not use it).
# Flag set adapted from a known-good recipe (Discord), minus --max-num-seqs 1.
#
# Usage: run-qwen.sh [--plain]   (--plain = drop the MTP speculative config)
#
# qwen-a: GPU 4 -> host port 8080
# qwen-b: GPU 5 -> host port 8081
# Pinned nightly: the 0.27.1 release (:latest) lacks DFlash2DraftModel; it
# landed in main after the release. This nightly is verified to contain it.
IMG="${QWEN_IMAGE:-vllm/vllm-openai:nightly-a9a17e7095a66ef6c6685a1c7ddd657781a78d3c}"
MODEL="${QWEN_MODEL:-$HOME/models/Qwen3.8-27B-SmoothQuant-W8A8-INT8}"
TEMPLATE="${QWEN_TEMPLATE:-$HOME/models/qwen3.8-chat-template.jinja}"
MAXLEN="${QWEN_MAXLEN:-262144}"   # model's max_position_embeddings

SPEC='--speculative-config {"method":"mtp","num_speculative_tokens":7,"draft_sample_method":"greedy"}'
if [ "$1" = "--plain" ]; then SPEC=""; fi

MTP_OK=""
for m in "$MODEL/model-mtp.safetensors" "$MODEL/model_mtp.safetensors"; do
  [ -f "$m" ] && MTP_OK=1
done
for f in "$MODEL/config.json" "$TEMPLATE"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: $f not found. Download the model/template first."
    exit 1
  fi
done
if [ -z "$MTP_OK" ] && [ -n "$SPEC" ]; then
  echo "WARNING: no MTP head file in $MODEL -- falling back to --plain"
  SPEC=""
fi

start_one() {
  local name="$1" gpu="$2" port="$3"
  docker stop -t 60 "$name" >/dev/null 2>&1
  docker rm "$name" >/dev/null 2>&1
  # shellcheck disable=SC2086
  docker run -d --name "$name" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$gpu" \
    --restart unless-stopped \
    -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -v "$MODEL":/model -v "$TEMPLATE":/config/chat_template.jinja:ro \
    --shm-size=8g -p "$port:$port" \
    "$IMG" /model --served-model-name qwen3.8 \
      --port "$port" \
      --max-model-len "$MAXLEN" \
      --dtype bfloat16 \
      --language-model-only \
      --enable-auto-tool-choice --tool-call-parser qwen3_xml \
      --reasoning-parser qwen3 \
      --enable-prefix-caching --enable-chunked-prefill \
      --kv-cache-dtype fp8_e4m3 \
      --mamba-cache-mode align --mamba-ssm-cache-dtype auto \
      --prefix-match-unit 16 --prefix-cache-retention-interval 1648 \
      --prefix-caching-hash-algo sha256 \
      --max-num-batched-tokens 8192 --max-num-scheduled-tokens 8192 \
      --performance-mode interactivity \
      --chat-template /config/chat_template.jinja \
      --gpu-memory-utilization 0.9 \
      $SPEC >/dev/null
  echo "launched $name on :$port (gpu $gpu, spec: ${SPEC:-none})"
}

start_one qwen-a 4 8080
start_one qwen-b 5 8081
