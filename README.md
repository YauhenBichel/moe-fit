# moe-fit

**Will this mixture-of-experts model run on my machine, and how fast?**
Answered in seconds, from the model's index, before downloading a single gigabyte of weights.

[![tests](https://github.com/YauhenBichel/moe-fit/actions/workflows/tests.yml/badge.svg)](https://github.com/YauhenBichel/moe-fit/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

The usual way to find out whether DeepSeek V3.1 runs on your box is to download 405 GB and see what
happens. This reads the 3 MB index instead.

```console
$ moefit plan https://huggingface.co/unsloth/DeepSeek-V3.1-GGUF/resolve/main/Q4_K_M/DeepSeek-V3.1-Q4_K_M-00001-of-00009.gguf
Deepseek-V3.1 on this machine

  model              405.4 GB
  needed every token 11.4 GB of weights + 576 MB of KV cache at 16,384 context
  experts            394.0 GB, 7 layers on the GPU, 51 layers off it
  kept in memory     108.3 GB of experts
  fetched per token  8.9 GB from storage (of 12.3 GB read)

  runs at roughly 0.07-0.09 tokens per second
  a 200-word answer would take about 66 minutes: this runs, but it is not something you sit and wait for
```

That machine has 64 GiB of GPU memory and 62 GiB of system memory — **a sixth of the model** — and
the answer is still "yes, it runs", with an honest number attached.

## Why a 405 GB model fits in 128 GB

Because almost none of it is needed at once. In DeepSeek V3.1:

| | size | needed per token |
|---|---|---|
| routed experts | 394.0 GB (97%) | 8 of 256, so 12.3 GB |
| attention, embeddings, shared expert | 11.4 GB (3%) | all of it |

So the plan writes itself: put the 11.4 GB that every token needs on the GPU, fill the rest of the
GPU with whole layers of experts, let system memory cache more of them, and read the remainder from
storage as the router asks for it. Speed then comes down to one number — how fast your disk serves
the small random reads a mapped file faults in — which `moe-fit` measures rather than assumes.

This is not a new inference engine. It is the placement arithmetic that experienced people do by
hand and that everyone else discovers after a very long download, plus the exact `llama.cpp`
arguments that implement it.

## Install

```bash
pip install moe-fit        # or: uv tool install moe-fit
```

No dependencies — the standard library does all of it. Python 3.10+.

## Use

```bash
moefit inspect MODEL     # what the model is made of
moefit bench             # what this machine can hold and how fast it reads
moefit plan MODEL        # the placement and the speed estimate
moefit flags MODEL       # the llama.cpp arguments for that placement
moefit verify MODEL      # run llama.cpp and compare the real speed with the estimate
```

`MODEL` is a local `.gguf` file or an `https://` URL to one. A split model is followed across its
shards automatically; only indexes are read, one small ranged request each.

To ask about a machine you do not have — before buying memory, say — describe it instead:

```bash
moefit plan MODEL --vram-gb 24 --ram-gb 64 --read-gb-s 7
```

```console
$ moefit flags DeepSeek-V3.1-Q4_K_M-00001-of-00009.gguf
--model DeepSeek-V3.1-Q4_K_M-00001-of-00009.gguf --ctx-size 16384 --n-gpu-layers 999 \
  --n-cpu-moe 51 --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on
```

Feed those straight to `llama-server` or `llama-cli`.

## What the numbers mean, and how much to trust them

**Structure is fact.** Sizes, expert share, what must be resident, bytes read per token and the
resulting placement are read out of the index and are exact.

**Speed is an estimate.** It is a floor derived from bytes fetched per token divided by measured
read speed, so it ignores compute and runtime overhead, which can only make things slower. It is
reported as a range: the low end assumes cached experts are no more likely to be reused than any
other, the high end assumes the well-documented skew in expert routing helps by `--skew` (1.6 by
default). **`moefit verify` is what turns the estimate into a measurement** — run it before
believing a number.

The storage benchmark uses `O_DIRECT` random reads at the size the runtime actually fetches —
128 KiB by default, which is what the kernel faults in for a mapped file. **That size matters more
than the drive:** this NVMe does 4.13 GB/s in 8 MiB chunks and 0.028 GB/s in 4 KiB ones, a 147-fold
spread, so a speed quoted without its fetch size means nothing. `moefit bench --profile` prints the
curve. Where the kernel refuses `O_DIRECT` the tool says so, because a cached read would report the
speed of RAM and flatter the result.

## Honest limitations

- **The speed model has been validated once, and it was wrong the first time.** Benchmarking at
  8 MiB while llama.cpp faults 128 KiB pages overstated a real run by twentyfold; the whole
  experiment, including the numbers that caught it, is in
  [docs/validation-2026-09-12.md](docs/validation-2026-09-12.md). It is right on one model, one
  machine, one runtime. Reports from other hardware are the contribution I most want.
- Only `llama.cpp`-style GGUF is understood — not vLLM, not SGLang, not ktransformers, all of which
  place tensors differently and would deserve their own back end.
- The KV cache estimate covers compressed-latent attention (DeepSeek's MLA) and ordinary
  grouped-query attention. An architecture that does neither gets a crude fallback.
- Prefill is not modelled. A long prompt reads far more than one token's worth of experts.
- It does not make a slow model fast. If the answer is 0.5 tokens per second, that is the machine
  telling you something true.

## Contributing

Issues and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). The tests build GGUF
files byte by byte, so the whole suite runs in under a second with no network and no model files.

```bash
uv run pytest -q
```

## Contributors

Thank you to everyone who has helped.

<!-- readme: contributors,bots/- -start -->
<p align="center">
  <a href="https://github.com/YauhenBichel" title="Yauhen Bichel" aria-label="Yauhen Bichel"><img src=".github/faces/YauhenBichel.svg" width="87" height="99" alt="Yauhen Bichel" /></a>
</p>
<!-- readme: contributors,bots/- -end -->

## Licence

Apache-2.0. See [LICENSE](LICENSE).
