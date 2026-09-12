#!/usr/bin/env bash
# Does the speed estimate hold? Force a model to stream its experts, then compare.
#
# The estimate divides bytes fetched per token by the measured read speed. The only way to check it
# is to make a model actually fetch: cap the memory a run may use, so the page cache cannot hold the
# experts, and see what llama.cpp does.
#
# Three things are needed for the measurement to mean anything:
#   1. a private copy of the model, so the page cache holding the original does not serve the reads;
#   2. that copy evicted from the cache (posix_fadvise DONTNEED - no root needed);
#   3. the run inside a cgroup with MemoryMax, so what it caches is bounded and charged to it.
#
#   bash moe-fit-validate.sh <model.gguf> <memory-cap-GB> [tokens]
set -euo pipefail
MODEL=${1:?usage: moe-fit-validate.sh <model.gguf> <memory-cap-GB> [tokens]}
CAP_GB=${2:?}
TOKENS=${3:-24}
LLAMA=$HOME/llamacpp/llama-b10905/llama-cli
WORK=$HOME/.cache/moe-fit-validate
COPY=$WORK/model.gguf

mkdir -p "$WORK"
if [ ! -f "$COPY" ] || [ "$(stat -c%s "$COPY")" != "$(stat -c%s "$MODEL")" ]; then
  echo "== copying the model so the original's cached pages cannot serve it"
  cp "$MODEL" "$COPY"
fi

echo "== evicting it from the page cache"
python3 - "$COPY" <<'PY'
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)
print("   evicted")
PY

echo "== what moefit predicts for a machine with ${CAP_GB} GB and no GPU"
cd /tmp/moefit-dev
PYTHONPATH=src python3 -m moefit.cli plan "$COPY" --disk "$WORK" --vram-gb 0 --ram-gb "$CAP_GB" \
  --context 4096 | sed 's/^/   /'
FLAGS=$(PYTHONPATH=src python3 -m moefit.cli flags "$COPY" --disk "$WORK" --no-measure \
  --vram-gb 0 --ram-gb "$CAP_GB" --context 4096)

echo "== running llama.cpp under a ${CAP_GB} GB cap"
# --no-mmap is deliberately NOT used: mmap is what lets the page cache hold the hot experts and the
# rest stay on disk, which is the arrangement being tested.
systemd-run --user --scope -q -p MemoryMax="${CAP_GB}G" -p MemorySwapMax=0 \
  "$LLAMA" $FLAGS -ngl 0 -n "$TOKENS" -st --no-warmup \
  -p "Write one sentence about mixture-of-experts models." 2>&1 |
  grep -E "eval time|tokens per second|load time|error|failed" | sed 's/^/   /'
