# Hugging Face post

Written for huggingface.co/posts. Kept here so the claims in it stay next to the code that makes
them, and so the numbers can be checked when the tool changes.

Every figure below was measured on 2026-09-12 on one machine: an AMD Ryzen AI MAX+ 395 (Strix Halo)
with 128 GB of unified memory — 64 GiB of it given to the GPU in the BIOS, leaving 62 GiB to the
system — and a Crucial 2 TB NVMe drive.

---

**Your 128 GB machine can run DeepSeek V3.1. It just won't be fast — and you can know that in 3
seconds instead of after a 405 GB download.**

I kept asking a question with no quick answer: will this MoE model run on my box? The honest way to
find out was to download it and see. So I wrote a tool that reads the GGUF **index** instead — a
few hundred kilobytes, over a ranged HTTP request — and does the arithmetic.

For DeepSeek V3.1 Q4_K_M, the index says something I found genuinely surprising:

| | size | needed per token |
|---|---|---|
| routed experts | 394.0 GB (97%) | 8 of 256 → 12.3 GB |
| attention + embeddings + shared expert | 11.4 GB (3%) | all of it |

**Only 11.4 GB of a 405 GB model is needed by every token.** That is what makes it runnable on
hardware a sixth of its size: put those 11.4 GB on the GPU, fill the rest of the GPU with whole
layers of experts, let the page cache hold more, and stream the remainder from NVMe as the router
asks for it.

On my box that gives:

```
  model              405.4 GB
  needed every token 11.4 GB of weights + 576 MB of KV cache at 16,384 context
  experts            394.0 GB, 7 layers on the GPU, 51 layers off it
  kept in memory     108.3 GB of experts
  fetched per token  8.9 GB from storage (of 12.3 GB read)

  runs at roughly 0.46-0.59 tokens per second
```

Half a token per second is about 20 words a minute. That is a real answer, not a useful chat — and
knowing it in 3 seconds is worth a lot more than knowing it after a 405 GB download.

The speed follows from one number nobody knows offhand: how fast your disk serves random
multi-megabyte reads. The tool measures it with `O_DIRECT` at the size experts are actually fetched
in, because a cached read would report the speed of your RAM and flatter the result. Mine does
4.5 GB/s; 8.9 GB fetched per token is where the ~2 s/token comes from.

It also prints the `llama.cpp` arguments for the placement it worked out:

```
--model DeepSeek-V3.1-Q4_K_M-00001-of-00009.gguf --ctx-size 16384 \
  --n-gpu-layers 999 --n-cpu-moe 51 --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on
```

```bash
pip install moe-fit
moefit plan https://huggingface.co/unsloth/DeepSeek-V3.1-GGUF/resolve/main/Q4_K_M/DeepSeek-V3.1-Q4_K_M-00001-of-00009.gguf
```

**What to trust.** The structure is exact — sizes, expert share, what must be resident, bytes per
token — it is read straight out of the index. The speed is an *estimate*: a floor from bytes fetched
divided by measured read rate, ignoring compute, and it rests on an assumption about how often a
cached expert gets reused. It is printed as a range and labelled as one, and `moefit verify` runs
llama.cpp and compares. **I have validated the arithmetic, not yet the speed estimate across a range
of hardware** — if you run both commands, please open an issue with the two numbers. That is the
contribution I would most like.

No dependencies, Apache-2.0, and the tests build GGUF files byte by byte so the suite runs in 0.05 s
with no network and no model files.

https://github.com/YauhenBichel/moe-fit
