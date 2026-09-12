# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""Read a GGUF file's index: what tensors it holds, how big each one is, and which are experts.

The index sits at the front of the file, so this reads a few hundred kilobytes whether the model is
on disk or on a web server. That is the point: a 404 GB model can be measured before a byte of its
weights is downloaded.

GGUF layout (v2 and v3): the magic "GGUF", a version, the tensor count and the metadata count, then
the metadata key/value pairs, then one record per tensor (name, shape, type, offset), then padding
to `general.alignment`, then the weights themselves.
"""
from __future__ import annotations

import re
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

MAGIC = b"GGUF"

# ggml type -> (elements per block, bytes per block). A quantised tensor stores whole blocks, so its
# size is elements / block * bytes. Unknown types fall back to the gap to the next tensor's offset.
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17),
}

# Names llama.cpp gives the routed expert weights. These are the tensors that make a mixture-of-
# experts model enormous and that only a few of are read per token, which is what makes offloading
# them worthwhile. The shared expert (`_shexp`) runs for every token and is deliberately excluded.
EXPERT_PATTERN = re.compile(r"\.ffn_(gate|up|down)_exps\.weight$")
SHARED_EXPERT_PATTERN = re.compile(r"\.ffn_(gate|up|down)_shexp\.weight$")
LAYER_PATTERN = re.compile(r"^blk\.(\d+)\.")


class GGUFError(Exception):
    """The file is not GGUF, or its index cannot be read."""


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    type_id: int
    offset: int
    n_bytes: int

    @property
    def type_name(self) -> str:
        return GGML_TYPES.get(self.type_id, (f"type{self.type_id}", 0, 0))[0]

    @property
    def is_expert(self) -> bool:
        return bool(EXPERT_PATTERN.search(self.name))

    @property
    def is_shared_expert(self) -> bool:
        return bool(SHARED_EXPERT_PATTERN.search(self.name))

    @property
    def layer(self) -> int | None:
        match = LAYER_PATTERN.match(self.name)
        return int(match.group(1)) if match else None


@dataclass
class Model:
    """What the index says about a model, with no weights read."""

    path: str
    metadata: dict[str, object]
    tensors: list[Tensor]
    file_size: int | None = None

    # --- the numbers a placement decision needs ---------------------------------------------

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", "unknown"))

    @property
    def name(self) -> str:
        return str(self.metadata.get("general.name", Path(self.path).name))

    def _arch_key(self, suffix: str) -> object | None:
        return self.metadata.get(f"{self.architecture}.{suffix}")

    @property
    def n_layers(self) -> int:
        value = self._arch_key("block_count")
        if isinstance(value, int):
            return value
        layers = {t.layer for t in self.tensors if t.layer is not None}
        return len(layers)

    @property
    def n_experts(self) -> int:
        value = self._arch_key("expert_count")
        return value if isinstance(value, int) else 0

    @property
    def n_experts_used(self) -> int:
        value = self._arch_key("expert_used_count")
        return value if isinstance(value, int) else 0

    @property
    def context_length(self) -> int:
        value = self._arch_key("context_length")
        return value if isinstance(value, int) else 0

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 1 and any(t.is_expert for t in self.tensors)

    @property
    def total_bytes(self) -> int:
        return sum(t.n_bytes for t in self.tensors)

    @property
    def expert_bytes(self) -> int:
        return sum(t.n_bytes for t in self.tensors if t.is_expert)

    @property
    def resident_bytes(self) -> int:
        """Everything that is not a routed expert: attention, embeddings, the shared expert."""
        return self.total_bytes - self.expert_bytes

    def expert_bytes_per_layer(self) -> dict[int, int]:
        per_layer: dict[int, int] = {}
        for tensor in self.tensors:
            if tensor.is_expert and tensor.layer is not None:
                per_layer[tensor.layer] = per_layer.get(tensor.layer, 0) + tensor.n_bytes
        return per_layer

    def bytes_read_per_token(self) -> int:
        """Expert weights one token touches: used experts out of all of them, layer by layer.

        The routed experts of a layer are stored as one tensor per projection with the expert as the
        first dimension, so a token reads `expert_used_count / expert_count` of it.
        """
        if not self.is_moe or not self.n_experts:
            return 0
        share = self.n_experts_used / self.n_experts
        return int(self.expert_bytes * share)


# --- reading ----------------------------------------------------------------------------------


class _Reader:
    """Sequential reads over a file or an HTTP range request, whichever the path calls for."""

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream

    def read(self, n: int) -> bytes:
        data = self.stream.read(n)
        if len(data) != n:
            raise GGUFError(f"file ended early: wanted {n} bytes, got {len(data)}")
        return data

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        return self.read(self.u64()).decode("utf-8", errors="replace")

    def value(self, type_id: int) -> object:
        simple = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
                  6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
        if type_id in simple:
            fmt, size = simple[type_id]
            return struct.unpack(fmt, self.read(size))[0]
        if type_id == 8:
            return self.string()
        if type_id == 9:
            element_type, count = self.u32(), self.u64()
            # Token vocabularies run to hundreds of thousands of strings and none of them affect
            # placement, so long arrays are counted and skipped rather than kept.
            if count > 1024:
                for _ in range(count):
                    self.value(element_type)
                return f"<array of {count}>"
            return [self.value(element_type) for _ in range(count)]
        raise GGUFError(f"unknown metadata type {type_id}")


def _tensor_bytes(shape: tuple[int, ...], type_id: int) -> int | None:
    elements = 1
    for dim in shape:
        elements *= dim
    spec = GGML_TYPES.get(type_id)
    if spec is None:
        return None
    _, block_elements, block_bytes = spec
    if elements % block_elements:
        return None
    return elements // block_elements * block_bytes


def _parse(stream: BinaryIO, path: str, file_size: int | None) -> Model:
    reader = _Reader(stream)
    if reader.read(4) != MAGIC:
        raise GGUFError(f"{path} does not start with the GGUF magic")
    version = reader.u32()
    if version not in (2, 3):
        raise GGUFError(f"GGUF version {version} is not supported (this reads v2 and v3)")
    tensor_count, metadata_count = reader.u64(), reader.u64()

    metadata: dict[str, object] = {}
    for _ in range(metadata_count):
        key = reader.string()
        metadata[key] = reader.value(reader.u32())

    raw: list[tuple[str, tuple[int, ...], int, int]] = []
    for _ in range(tensor_count):
        name = reader.string()
        shape = tuple(reader.u64() for _ in range(reader.u32()))
        raw.append((name, shape, reader.u32(), reader.u64()))

    tensors: list[Tensor] = []
    for index, (name, shape, type_id, offset) in enumerate(raw):
        n_bytes = _tensor_bytes(shape, type_id)
        if n_bytes is None:                       # an unknown quantisation: measure the gap instead
            following = raw[index + 1][3] if index + 1 < len(raw) else None
            n_bytes = (following - offset) if following is not None else 0
        tensors.append(Tensor(name, shape, type_id, offset, n_bytes))
    return Model(path=path, metadata=metadata, tensors=tensors, file_size=file_size)


def read_local(path: str | Path) -> Model:
    path = Path(path)
    with path.open("rb") as handle:
        return _parse(handle, str(path), path.stat().st_size)


def read_remote(url: str, first_chunk: int = 8 << 20) -> Model:
    """Read the index over HTTP, without downloading the weights.

    Needs a server that honours Range requests; Hugging Face does. The index of a large model can
    exceed the first chunk, so the chunk is doubled until the whole index has been read (in
    practice one or two requests).
    """
    size: int | None = None
    while first_chunk <= (256 << 20):
        request = urllib.request.Request(url, headers={"Range": f"bytes=0-{first_chunk - 1}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
            content_range = response.headers.get("Content-Range", "")
            if "/" in content_range:
                tail = content_range.rsplit("/", 1)[1]
                size = int(tail) if tail.isdigit() else None
        try:
            import io

            return _parse(io.BytesIO(data), url, size)
        except GGUFError as exc:
            if "ended early" not in str(exc):
                raise
            first_chunk *= 2
    raise GGUFError("the index did not fit in 256 MB; is this really a GGUF file?")


def read_one(source: str | Path) -> Model:
    text = str(source)
    return read_remote(text) if text.startswith(("http://", "https://")) else read_local(text)


SHARD = re.compile(r"^(?P<stem>.*)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$")


def read(source: str | Path, shards: bool = True) -> Model:
    """Read a model's index, following the other shards when it is split into several files.

    A split model keeps its metadata in the first shard and the rest of its tensors in the others,
    so placement needs all of them. Each shard costs one small ranged read, not its weights.
    """
    first = read_one(source)
    count = first.metadata.get("split.count")
    if not shards or not isinstance(count, int) or count <= 1:
        return first

    match = SHARD.match(str(source))
    if not match:                       # split metadata but an unexpected name: use what we have
        return first

    stem, total = match.group("stem"), int(match.group("count"))
    seen = {t.name for t in first.tensors}
    size = first.file_size or 0
    for number in range(1, total + 1):
        if number == int(match.group("no")):
            continue
        other = read_one(f"{stem}-{number:05d}-of-{total:05d}.gguf")
        size += other.file_size or 0
        for tensor in other.tensors:
            if tensor.name not in seen:
                seen.add(tensor.name)
                first.tensors.append(tensor)
    first.file_size = size
    return first


def layers(model: Model) -> Iterator[int]:
    seen = sorted({t.layer for t in model.tensors if t.layer is not None})
    yield from seen
