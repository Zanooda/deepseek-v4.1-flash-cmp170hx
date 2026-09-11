#!/bin/bash
# Start the serving stack on this box: DeepSeek-V4.1-Flash on all eight cards.
# Everything (partition, memory, Engram in host RAM, device order, parsers) is
# configured in launch/run-v41-pp8.sh; this only supplies the two things that
# are specific to this host:
#   DSV41_VLLM_SRC   the vllm/ dir of the checkout with all patches/v41 applied.
#                    The image holds patches 0001-0008 only; 0009-0013 are
#                    bind-mounted from here and are required (see README).
#   NCCL_P2P_LEVEL   SYS: the cards run the cmpunlocker BAR1 P2P, so NCCL may
#                    route the pipeline hops GPU-to-GPU (no measured speed
#                    change, but it is the path that stays valid across the
#                    four root ports).
# The previous 4-card DeepSeek-V4 layout is launch/run-pp-dspark.sh.
set -e
cd "$(dirname "$0")"
DSV41_VLLM_SRC="${DSV41_VLLM_SRC:-$HOME/vllm-v41/vllm}" \
NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}" \
bash launch/run-v41-pp8.sh "$@"
