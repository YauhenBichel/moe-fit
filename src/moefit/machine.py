# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""What this machine can hold and how fast it can read, measured rather than assumed.

A plan for a mixture-of-experts model is only as good as three numbers: how much the GPU can hold,
how much the system can hold, and how fast experts arrive from storage when they are in neither.
The last one decides the speed and is the one nobody knows offhand, so it is measured here with
O_DIRECT reads at the size an expert is actually read in - bypassing the page cache, which would
otherwise report the speed of RAM.
"""
from __future__ import annotations

import ctypes
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

GIB = 1 << 30


@dataclass
class Machine:
    vram_bytes: int
    ram_bytes: int
    free_disk_bytes: int
    disk_path: str
    read_bytes_per_second: float | None = None
    read_samples: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {"vram_gb": round(self.vram_bytes / 1e9, 1),
                "ram_gb": round(self.ram_bytes / 1e9, 1),
                "free_disk_gb": round(self.free_disk_bytes / 1e9, 1),
                "disk_path": self.disk_path,
                "read_gb_per_second": round(self.read_bytes_per_second / 1e9, 2)
                if self.read_bytes_per_second else None,
                "notes": self.notes}


def vram_bytes() -> tuple[int, list[str]]:
    """GPU memory, from the kernel. Works for an AMD integrated GPU without any vendor tool."""
    notes: list[str] = []
    total = 0
    for path in sorted(Path("/sys/class/drm").glob("card*/device/mem_info_vram_total")):
        try:
            total = max(total, int(path.read_text().strip()))
        except (OSError, ValueError):
            continue
    if total:
        return total, notes
    notes.append("no GPU memory reported by the kernel; assuming CPU only")
    return 0, notes


def ram_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return 0


def measure_read(path: str | Path, block_bytes: int = 8 << 20, samples: int = 24,
                 file_bytes: int = 2 << 30) -> tuple[float | None, list[float], list[str]]:
    """Random reads of `block_bytes`, with the page cache bypassed, in bytes per second.

    Experts are read in chunks of a few megabytes scattered across the file, which is nothing like
    a sequential copy, so the benchmark matches that shape. O_DIRECT needs the buffer and the
    offsets aligned to the block device's logical size; 4096 satisfies every current drive.
    """
    notes: list[str] = []
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    scratch = directory / ".moefit-read-test"

    align = 4096
    block_bytes = max(align, (block_bytes // align) * align)
    file_bytes = max(block_bytes * 8, (file_bytes // align) * align)

    try:
        if not scratch.exists() or scratch.stat().st_size < file_bytes:
            # Written once, then reused. Random bytes so no filesystem compresses it away.
            with scratch.open("wb") as handle:
                chunk = os.urandom(1 << 20)
                for _ in range(file_bytes // len(chunk)):
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
    except OSError as exc:
        return None, [], [f"could not write the test file to {scratch}: {exc}"]

    try:
        fd = os.open(str(scratch), os.O_RDONLY | getattr(os, "O_DIRECT", 0))
    except OSError as exc:
        notes.append(f"O_DIRECT refused ({exc}); falling back to a cached read, which overstates speed")
        try:
            fd = os.open(str(scratch), os.O_RDONLY)
        except OSError as exc2:
            return None, [], [f"cannot open the test file: {exc2}"]

    # An aligned buffer, which O_DIRECT requires.
    raw = ctypes.create_string_buffer(block_bytes + align)
    offset_into_buffer = (align - (ctypes.addressof(raw) % align)) % align
    buffer = (ctypes.c_char * block_bytes).from_buffer(raw, offset_into_buffer)

    size = os.fstat(fd).st_size
    highest = ((size - block_bytes) // align) * align
    speeds: list[float] = []
    try:
        readinto = os.preadv
        for _ in range(samples):
            offset = random.randrange(0, max(align, highest), align)
            started = time.perf_counter()
            got = readinto(fd, [buffer], offset)
            elapsed = time.perf_counter() - started
            if got > 0 and elapsed > 0:
                speeds.append(got / elapsed)
    except OSError as exc:
        notes.append(f"read failed: {exc}")
    finally:
        del buffer
        os.close(fd)

    if not speeds:
        return None, [], notes or ["no successful reads"]
    return statistics.median(speeds), speeds, notes


def describe(disk_path: str | Path | None = None, measure: bool = True,
             block_bytes: int = 8 << 20) -> Machine:
    disk = Path(disk_path) if disk_path else Path.home() / ".cache" / "moe-fit"
    vram, notes = vram_bytes()
    usage = os.statvfs(disk if disk.exists() else disk.parent if disk.parent.exists() else Path.home())
    machine = Machine(vram_bytes=vram, ram_bytes=ram_bytes(),
                      free_disk_bytes=usage.f_bavail * usage.f_frsize, disk_path=str(disk),
                      notes=notes)
    if measure:
        speed, samples, more = measure_read(disk, block_bytes=block_bytes)
        machine.read_bytes_per_second, machine.read_samples = speed, samples
        machine.notes.extend(more)
    return machine
