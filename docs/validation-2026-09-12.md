# Validating the speed estimate, and the twentyfold error it found

12 September 2026. `tools/validate.sh`, on a Ryzen AI MAX+ 395 (Strix Halo) with 62 GiB of system
memory and a Crucial 2 TB NVMe drive.

## What was tested

The estimate divides bytes fetched per token by measured read speed. The only honest check is to
make a model actually fetch, so:

1. a private copy of the model, so the page cache holding the original could not serve the reads;
2. that copy evicted with `posix_fadvise(POSIX_FADV_DONTNEED)`, which needs no root;
3. the run inside a cgroup with `MemoryMax`, so what it could cache was bounded.

Model: `Qwen3 Coder 30B A3B Instruct`, Q4_K_M, 18.6 GB — 17.6 GB of it routed experts, 128 experts
with 8 used per token. CPU only (`-ngl 0`), 4,096 context, 24 tokens generated.

## What happened

| memory cap | result |
|---|---|
| 6 GB | killed while loading (exit 137) |
| 10 GB | killed while loading |
| 14 GB | ran: **0.3 tokens per second** generation, 2.1 prompt |

The estimate for that machine was about **7 tokens per second**. The measurement was **0.3**.
Twenty times slower. An estimate meant to be a floor was a ceiling.

## Why

The benchmark read 8 MiB blocks. llama.cpp does not read 8 MiB blocks: it maps the file and lets
the kernel fault pages in, one page plus readahead at a time. The same drive, same file, same
random offsets, `O_DIRECT` throughout:

| fetch size | speed |
|---|---|
| 4 KiB | 0.028 GB/s |
| 128 KiB | 0.625 GB/s |
| 1 MiB | 2.462 GB/s |
| 8 MiB | 4.130 GB/s |

**A 147-fold spread on one drive.** Quoting a drive's speed without its fetch size is close to
meaningless, and the tool had been quoting the best case for a workload that gets the worst.

## What changed

- The default fetch size is now 128 KiB, Linux's readahead for a mapped file, and `--fetch-kib`
  exposes it. `moefit bench --profile` prints the whole curve, because the single number misleads
  without it.
- `moefit plan` now says how long a 200-word answer would take, which is what the rate means.

DeepSeek V3.1 on this machine went from a claimed **0.46-0.59 tokens per second** to **0.07-0.09**:
about 66 minutes for a 200-word answer. It still runs. It is not something you sit and wait for.

## What is still not validated

The measured 0.3 tokens per second is not all storage: CPU compute for a 3B-active model is in
there too, so the corrected estimate remains a floor, and the true figure for a GPU-resident
placement will differ. One model, one machine, one runtime. The next useful measurements are on
different drives (a SATA SSD, a RAID of NVMes) and with experts on the GPU rather than the CPU.

## Reproducing

```bash
bash tools/validate.sh /path/to/model.gguf 14 24
```

It copies the model, evicts it from the cache, prints the prediction, and runs `llama.cpp` under a
matching cap. A cap below about 14 GB kills the loader on this model rather than making it stream,
which is itself worth knowing.
