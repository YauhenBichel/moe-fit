# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""Where each tensor should live, and how fast that will be.

A mixture-of-experts model is mostly experts: in DeepSeek V3.1 they are 97% of the weights, and a
token touches 8 of 256 of them. Everything else - attention, embeddings, the shared expert - runs
for every token and is small. That asymmetry is what makes a 405 GB model runnable on a 128 GB
machine, and it is the whole basis of the plan below:

  1. everything that is not a routed expert goes on the GPU, because every token needs all of it;
  2. the KV cache goes on the GPU next, because every token writes to it;
  3. whatever GPU memory is left holds as many whole layers of experts as fit;
  4. system memory holds more of them, as page cache over the model file;
  5. the rest is read from storage as it is needed.

The speed follows from step 5: bytes a token must fetch, divided by the storage read speed. That is
a floor on the time per token, not a promise - compute and the runtime add to it - so the estimate
is reported as a range and `moefit verify` measures the truth.
"""
from __future__ import annotations

from dataclasses import dataclass

from .gguf import Model
from .machine import Machine

# Held back from the GPU for the runtime's own allocations, and from system memory for the OS.
GPU_HEADROOM = 2_000_000_000
OS_HEADROOM = 8_000_000_000
# Experts are stored one tensor per projection per layer, so a layer is the smallest unit that can
# be placed on one side or the other.
@dataclass
class Placement:
    model_gb: float
    resident_gb: float          # attention, embeddings, shared experts: needed by every token
    expert_gb: float
    kv_cache_gb: float
    expert_layers_on_gpu: int
    expert_layers_on_cpu: int
    cached_expert_gb: float     # experts that fit in GPU memory plus page cache
    streamed_per_token_gb: float
    read_per_token_gb: float
    context: int
    fits: bool
    reason: str
    seconds_per_token_floor: float | None
    tokens_per_second_estimate: tuple[float, float] | None

    def summary(self) -> str:
        if not self.fits:
            return f"will not run: {self.reason}"
        if self.tokens_per_second_estimate is None:
            return "runs; speed unknown (storage was not measured)"
        low, high = self.tokens_per_second_estimate
        return f"runs at roughly {low:.2f}-{high:.2f} tokens per second"


def kv_cache_bytes(model: Model, context: int, bits_per_element: int = 8) -> int:
    """Rough KV cache size for `context` tokens.

    DeepSeek-style models use a compressed latent cache (MLA), which is far smaller than the usual
    two-tensors-per-head arithmetic, so it is computed from the model's own latent dimensions when
    they are present and from heads and layers otherwise.
    """
    arch = model.architecture
    layers = model.n_layers or 1
    meta = model.metadata
    kv_lora = meta.get(f"{arch}.attention.kv_lora_rank")
    rope_dim = meta.get(f"{arch}.rope.dimension_count") or meta.get(f"{arch}.attention.qk_rope_head_dim")
    if isinstance(kv_lora, int) and isinstance(rope_dim, int):
        per_token = (kv_lora + rope_dim) * layers                  # one latent vector per layer
    else:
        heads_kv = meta.get(f"{arch}.attention.head_count_kv")
        embedding = meta.get(f"{arch}.embedding_length")
        heads = meta.get(f"{arch}.attention.head_count")
        if isinstance(heads_kv, int) and isinstance(embedding, int) and isinstance(heads, int) and heads:
            per_token = 2 * heads_kv * (embedding // heads) * layers
        else:
            per_token = 2 * 4096 * layers                          # a last resort
    return int(per_token * context * bits_per_element / 8)


def make(model: Model, machine: Machine, context: int | None = None,
         kv_bits: int = 8, skew: float = 1.6) -> Placement:
    """Decide the placement and estimate the speed.

    `skew` says how much better the cache does than its share of the weights would suggest, because
    expert routing is not uniform - some experts are picked far more often, and those are the ones
    that stay cached. 1.0 assumes no such luck; the default is deliberately modest. The estimate is
    reported as a range from the no-luck case to this one.
    """
    context = context or model.context_length or 8192
    total = model.total_bytes
    resident = model.resident_bytes
    experts = model.expert_bytes
    kv = kv_cache_bytes(model, context, kv_bits)

    gpu = max(0, machine.vram_bytes - GPU_HEADROOM)
    ram = max(0, machine.ram_bytes - OS_HEADROOM)

    if resident + kv > gpu + ram:
        return Placement(
            model_gb=total / 1e9, resident_gb=resident / 1e9, expert_gb=experts / 1e9,
            kv_cache_gb=kv / 1e9, expert_layers_on_gpu=0, expert_layers_on_cpu=model.n_layers,
            cached_expert_gb=0.0, streamed_per_token_gb=0.0,
            read_per_token_gb=model.bytes_read_per_token() / 1e9, context=context, fits=False,
            reason=(f"what every token needs ({(resident + kv)/1e9:.1f} GB of weights and KV cache) "
                    f"is more than this machine's {(gpu + ram)/1e9:.1f} GB of memory"),
            seconds_per_token_floor=None, tokens_per_second_estimate=None)

    if total > machine.free_disk_bytes:
        return Placement(
            model_gb=total / 1e9, resident_gb=resident / 1e9, expert_gb=experts / 1e9,
            kv_cache_gb=kv / 1e9, expert_layers_on_gpu=0, expert_layers_on_cpu=model.n_layers,
            cached_expert_gb=0.0, streamed_per_token_gb=0.0,
            read_per_token_gb=model.bytes_read_per_token() / 1e9, context=context, fits=False,
            reason=(f"the model is {total/1e9:.0f} GB and there is "
                    f"{machine.free_disk_bytes/1e9:.0f} GB of free disk"),
            seconds_per_token_floor=None, tokens_per_second_estimate=None)

    # Whole layers of experts fill what the GPU has left.
    per_layer = model.expert_bytes_per_layer()
    moe_layers = sorted(per_layer)
    free_gpu = gpu - resident - kv
    on_gpu, used = 0, 0
    for layer in moe_layers:
        if used + per_layer[layer] <= free_gpu:
            used += per_layer[layer]
            on_gpu += 1
        else:
            break
    on_cpu = len(moe_layers) - on_gpu

    cached = used + min(ram, experts - used)        # page cache holds what system memory allows
    cached_share = min(1.0, cached / experts) if experts else 1.0
    per_token = model.bytes_read_per_token()

    # Best case: popular experts stay cached, so the miss rate is lower than the uncached share.
    miss_plain = max(0.0, 1.0 - cached_share)
    miss_skewed = max(0.0, 1.0 - min(1.0, cached_share * skew))
    speed = machine.read_bytes_per_second

    estimate: tuple[float, float] | None = None
    floor: float | None = None
    if speed:
        slow = per_token * miss_plain / speed
        fast = per_token * miss_skewed / speed
        floor = slow
        estimate = (1 / slow if slow > 0 else float("inf"),
                    1 / fast if fast > 0 else float("inf"))

    return Placement(
        model_gb=total / 1e9, resident_gb=resident / 1e9, expert_gb=experts / 1e9,
        kv_cache_gb=kv / 1e9, expert_layers_on_gpu=on_gpu, expert_layers_on_cpu=on_cpu,
        cached_expert_gb=cached / 1e9, streamed_per_token_gb=per_token * miss_plain / 1e9,
        read_per_token_gb=per_token / 1e9, context=context, fits=True, reason="",
        seconds_per_token_floor=floor, tokens_per_second_estimate=estimate)


def llama_flags(model: Model, placement: Placement, model_path: str, context: int | None = None,
                kv_bits: int = 8) -> list[str]:
    """The llama.cpp arguments this placement corresponds to.

    `--n-cpu-moe N` keeps the experts of the first N layers off the GPU, which is exactly step 3
    above. mmap is left on deliberately: it is what lets the page cache hold the hot experts and the
    rest stay on disk, and `--mlock` would try to pin 400 GB of it and fail.
    """
    flags = ["--model", model_path,
             "--ctx-size", str(context or placement.context),
             "--n-gpu-layers", "999"]
    if placement.expert_layers_on_cpu > 0:
        flags += ["--n-cpu-moe", str(placement.expert_layers_on_cpu)]
    if kv_bits == 8:
        flags += ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
    flags += ["--flash-attn", "on"]
    return flags
