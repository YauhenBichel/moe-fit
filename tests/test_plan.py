# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""The placement: what goes where, whether it runs at all, and what that costs per token."""
from __future__ import annotations

from pathlib import Path

import pytest

from moefit import plan
from moefit.machine import Machine
from test_gguf import moe_model

GB = 1_000_000_000


def _vram_for(wanted: int) -> int:
    """GPU memory a machine needs so that `wanted` bytes remain usable after the headroom."""
    return int(wanted / (1 - plan.GPU_HEADROOM_SHARE)) + 1


def a_machine(vram_gb: float = 64, ram_gb: float = 64, disk_gb: float = 1000,
              read_gb_s: float | None = 4.5) -> Machine:
    return Machine(vram_bytes=int(vram_gb * GB), ram_bytes=int(ram_gb * GB),
                   free_disk_bytes=int(disk_gb * GB), disk_path="/tmp",
                   read_bytes_per_second=read_gb_s * GB if read_gb_s else None)


def test_a_model_that_fits_entirely_needs_no_streaming(tmp_path: Path) -> None:
    model = moe_model(tmp_path, layers=4, experts=8, used=2)
    placement = plan.make(model, a_machine())
    assert placement.fits
    assert placement.expert_layers_on_cpu == 0, "there is room for every expert on the GPU"
    assert placement.streamed_per_token_gb == 0
    assert placement.tokens_per_second_estimate[0] == float("inf")
    assert "storage is not the limit" in placement.summary(), "no infinities in front of a reader"


def test_a_model_whose_resident_part_is_too_big_will_not_run(tmp_path: Path) -> None:
    model = moe_model(tmp_path, layers=4)
    tiny = a_machine(vram_gb=0.001, ram_gb=0.001)
    placement = plan.make(model, tiny)
    assert not placement.fits
    assert "more than this machine" in placement.reason


def test_a_model_larger_than_the_disk_will_not_run(tmp_path: Path) -> None:
    model = moe_model(tmp_path, layers=4)
    placement = plan.make(model, a_machine(disk_gb=0.000001))
    assert not placement.fits
    assert "free disk" in placement.reason


def test_experts_fill_the_gpu_a_whole_layer_at_a_time(tmp_path: Path) -> None:
    model = moe_model(tmp_path, layers=4, experts=8, used=2)
    per_layer = next(iter(model.expert_bytes_per_layer().values()))
    # Room for the resident weights and two layers of experts, and not a byte more.
    vram = _vram_for(model.resident_bytes + int(2.5 * per_layer))
    placement = plan.make(model, a_machine(vram_gb=vram / GB, ram_gb=0.001))
    assert placement.expert_layers_on_gpu == 2
    assert placement.expert_layers_on_cpu == 2


def test_the_estimate_falls_as_less_is_cached(tmp_path: Path) -> None:
    model = moe_model(tmp_path, layers=8, experts=32, used=4)
    # Both machines can run it; the mean one just has nowhere to keep the experts.
    # Room for what every token needs and one spare gigabyte, so a little caching happens.
    just_enough = _vram_for(model.resident_bytes + GB) / GB
    generous = plan.make(model, a_machine(vram_gb=40, ram_gb=40))
    mean = plan.make(model, a_machine(vram_gb=just_enough, ram_gb=1.0))
    assert generous.fits and mean.fits
    assert generous.cached_expert_gb > mean.cached_expert_gb
    assert generous.tokens_per_second_estimate[0] > mean.tokens_per_second_estimate[0]


def test_without_a_measurement_there_is_no_speed_claim(tmp_path: Path) -> None:
    model = moe_model(tmp_path, layers=8, experts=32, used=4)
    just_enough = _vram_for(model.resident_bytes + GB) / GB
    placement = plan.make(model, a_machine(vram_gb=just_enough, ram_gb=1.0,
                                           read_gb_s=None))
    assert placement.fits
    assert placement.tokens_per_second_estimate is None
    assert "speed unknown" in placement.summary()


def test_a_compressed_kv_cache_is_recognised(tmp_path: Path) -> None:
    """DeepSeek keeps one small latent per layer rather than a key and a value per head.

    Measured against grouped-query attention with 8 key/value heads of 112 dimensions, the latent
    cache is about three times smaller (576 against 1,792 elements per layer per token). Against
    full multi-head attention the gap is far larger; three is the conservative comparison.
    """
    model = moe_model(tmp_path, layers=61)
    latent = plan.kv_cache_bytes(model, context=16384, bits_per_element=8)
    del model.metadata["deepseek2.attention.kv_lora_rank"]
    model.metadata["deepseek2.attention.head_count_kv"] = 8
    model.metadata["deepseek2.attention.head_count"] = 64
    model.metadata["deepseek2.embedding_length"] = 7168
    ordinary = plan.kv_cache_bytes(model, context=16384, bits_per_element=8)
    assert latent < ordinary / 3


def test_the_flags_say_how_many_layers_of_experts_to_keep_off_the_gpu(tmp_path: Path) -> None:
    model = moe_model(tmp_path, layers=4, experts=8, used=2)
    per_layer = next(iter(model.expert_bytes_per_layer().values()))
    vram = _vram_for(model.resident_bytes + int(1.2 * per_layer))
    placement = plan.make(model, a_machine(vram_gb=vram / GB, ram_gb=0.001))
    flags = plan.llama_flags(model, placement, "/models/m.gguf", context=4096)
    assert "--n-cpu-moe" in flags
    assert flags[flags.index("--n-cpu-moe") + 1] == str(placement.expert_layers_on_cpu)
    assert "--mlock" not in flags, "400 GB cannot be pinned; mmap is the point"
    assert flags[flags.index("--ctx-size") + 1] == "4096"


def test_a_fully_resident_model_gets_no_offload_flag(tmp_path: Path) -> None:
    model = moe_model(tmp_path, layers=4)
    placement = plan.make(model, a_machine())
    assert "--n-cpu-moe" not in plan.llama_flags(model, placement, "/models/m.gguf")


def test_a_partly_cached_model_reports_a_floor_not_an_infinity(tmp_path: Path) -> None:
    """When the cache-skew guess wipes out the misses, the upper bound is unbounded. Printing
    "inf tokens per second" would be nonsense; the floor with an explanation is the honest form."""
    model = moe_model(tmp_path, layers=8, experts=32, used=4)
    per_layer = next(iter(model.expert_bytes_per_layer().values()))
    vram = _vram_for(model.resident_bytes + int(5.5 * per_layer))
    placement = plan.make(model, a_machine(vram_gb=vram / GB, ram_gb=1.0),
                          skew=4.0)
    low, high = placement.tokens_per_second_estimate
    assert low < float("inf") and high == float("inf")
    assert "or better" in placement.summary()
    assert "inf" not in placement.summary().replace("infinit", "")


def test_the_headroom_never_swallows_a_small_machine(tmp_path: Path) -> None:
    """Subtracting a flat 8 GB left a 6 GB machine with nothing, so the planner said a model it can
    hold "will not run". Found by pointing tools/validate.sh at a 6 GB cap."""
    assert plan.usable_ram(6 * GB) > 4 * GB
    assert plan.usable_ram(64 * GB) == 64 * GB - plan.OS_HEADROOM      # the ceiling still applies
    assert plan.usable_gpu(0) == 0
    model = moe_model(tmp_path, layers=4, experts=8, used=2)
    assert plan.make(model, a_machine(vram_gb=0, ram_gb=6)).fits
