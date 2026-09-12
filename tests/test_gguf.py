# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""Reading the index: a GGUF file is built here, so the tests need no network and no large files."""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

from moefit import gguf


def _string(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, type_id: int, packed: bytes) -> bytes:
    return _string(key) + struct.pack("<I", type_id) + packed


def build_gguf(path: Path, tensors: list[tuple[str, tuple[int, ...], int]],
               metadata: dict[str, object] | None = None, version: int = 3) -> Path:
    """A GGUF file with a real index and no weights behind it: enough for every index-only read."""
    metadata = metadata or {}
    body = b""
    for key, value in metadata.items():
        if isinstance(value, bool):
            body += _kv(key, 7, struct.pack("<?", value))
        elif isinstance(value, int):
            body += _kv(key, 4, struct.pack("<I", value))
        elif isinstance(value, str):
            body += _kv(key, 8, _string(value))
        else:
            raise AssertionError(f"the builder has no case for {type(value)}")

    offset = 0
    index = b""
    for name, shape, type_id in tensors:
        index += _string(name) + struct.pack("<I", len(shape))
        for dim in shape:
            index += struct.pack("<Q", dim)
        index += struct.pack("<I", type_id) + struct.pack("<Q", offset)
        size = gguf._tensor_bytes(shape, type_id) or 0
        offset += size

    header = (gguf.MAGIC + struct.pack("<I", version)
              + struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(metadata)))
    path.write_bytes(header + body + index)
    return path


F32, Q4_K = 0, 12


def moe_model(tmp_path: Path, layers: int = 4, experts: int = 8, used: int = 2) -> gguf.Model:
    tensors: list[tuple[str, tuple[int, ...], int]] = [("token_embd.weight", (4096, 2048), Q4_K)]
    for layer in range(layers):
        tensors += [
            (f"blk.{layer}.attn_q.weight", (4096, 4096), Q4_K),
            (f"blk.{layer}.attn_norm.weight", (4096,), F32),
            (f"blk.{layer}.ffn_gate_shexp.weight", (4096, 2048), Q4_K),
            (f"blk.{layer}.ffn_gate_exps.weight", (experts, 4096, 2048), Q4_K),
            (f"blk.{layer}.ffn_up_exps.weight", (experts, 4096, 2048), Q4_K),
            (f"blk.{layer}.ffn_down_exps.weight", (experts, 2048, 4096), Q4_K),
        ]
    build_gguf(tmp_path / "m.gguf", tensors, {
        "general.architecture": "deepseek2", "general.name": "Test MoE",
        "deepseek2.block_count": layers, "deepseek2.expert_count": experts,
        "deepseek2.expert_used_count": used, "deepseek2.context_length": 8192,
        "deepseek2.attention.kv_lora_rank": 512, "deepseek2.rope.dimension_count": 64})
    return gguf.read(tmp_path / "m.gguf")


def test_a_file_that_is_not_gguf_is_refused(tmp_path: Path) -> None:
    bad = tmp_path / "not.gguf"
    bad.write_bytes(b"this is not a model")
    with pytest.raises(gguf.GGUFError, match="magic"):
        gguf.read(bad)


def test_an_unsupported_version_is_refused(tmp_path: Path) -> None:
    build_gguf(tmp_path / "old.gguf", [("a", (2,), F32)], {"general.name": "old"}, version=1)
    with pytest.raises(gguf.GGUFError, match="version 1"):
        gguf.read(tmp_path / "old.gguf")


def test_metadata_and_shapes_are_read(tmp_path: Path) -> None:
    model = moe_model(tmp_path)
    assert model.name == "Test MoE"
    assert model.architecture == "deepseek2"
    assert model.n_layers == 4
    assert model.n_experts == 8
    assert model.n_experts_used == 2
    assert model.context_length == 8192


def test_tensor_sizes_follow_the_quantisation(tmp_path: Path) -> None:
    build_gguf(tmp_path / "q.gguf", [("a", (256, 4), Q4_K), ("b", (16,), F32)], {"general.name": "q"})
    model = gguf.read(tmp_path / "q.gguf")
    sizes = {t.name: t.n_bytes for t in model.tensors}
    assert sizes["a"] == 256 * 4 // 256 * 144      # Q4_K: 256 elements per 144-byte block
    assert sizes["b"] == 16 * 4                    # F32


def test_experts_are_told_apart_from_everything_else(tmp_path: Path) -> None:
    model = moe_model(tmp_path)
    assert model.is_moe
    experts = [t for t in model.tensors if t.is_expert]
    shared = [t for t in model.tensors if t.is_shared_expert]
    assert len(experts) == 12                       # three projections on each of four layers
    assert len(shared) == 4
    # The shared expert runs for every token, so it must not be counted as a routed expert.
    assert all(not t.is_expert for t in shared)
    assert model.resident_bytes == model.total_bytes - model.expert_bytes
    assert model.resident_bytes > 0


def test_a_dense_model_is_not_a_mixture_of_experts(tmp_path: Path) -> None:
    build_gguf(tmp_path / "d.gguf", [("blk.0.ffn_up.weight", (4096, 4096), Q4_K)],
               {"general.architecture": "llama", "general.name": "dense", "llama.block_count": 1})
    model = gguf.read(tmp_path / "d.gguf")
    assert not model.is_moe
    assert model.expert_bytes == 0
    assert model.bytes_read_per_token() == 0


def test_bytes_per_token_is_the_used_share_of_the_experts(tmp_path: Path) -> None:
    model = moe_model(tmp_path, experts=8, used=2)
    assert model.bytes_read_per_token() == pytest.approx(model.expert_bytes / 4, rel=1e-6)


def test_expert_bytes_are_reported_per_layer(tmp_path: Path) -> None:
    model = moe_model(tmp_path, layers=3)
    per_layer = model.expert_bytes_per_layer()
    assert sorted(per_layer) == [0, 1, 2]
    assert sum(per_layer.values()) == model.expert_bytes


def test_layers_come_from_the_tensor_names_when_metadata_is_silent(tmp_path: Path) -> None:
    build_gguf(tmp_path / "n.gguf", [("blk.0.attn_q.weight", (8, 8), F32),
                                     ("blk.1.attn_q.weight", (8, 8), F32)],
               {"general.architecture": "llama", "general.name": "n"})
    assert gguf.read(tmp_path / "n.gguf").n_layers == 2


def test_a_long_array_is_summarised_rather_than_kept(tmp_path: Path) -> None:
    """Vocabularies run to hundreds of thousands of strings and change no decision."""
    path = tmp_path / "vocab.gguf"
    body = _string("tokenizer.ggml.tokens") + struct.pack("<I", 9)
    body += struct.pack("<I", 8) + struct.pack("<Q", 2000)
    body += b"".join(_string(f"t{i}") for i in range(2000))
    header = gguf.MAGIC + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1)
    path.write_bytes(header + body)
    model = gguf.read(path)
    assert model.metadata["tokenizer.ggml.tokens"] == "<array of 2000>"
