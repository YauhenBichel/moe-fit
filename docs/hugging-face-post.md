# Hugging Face post

Written for huggingface.co/posts. Kept here so the claims in it stay next to the code that makes
them, and so the numbers can be checked when the tool changes.

Every figure below was measured on 2026-09-12 on one machine: an AMD Ryzen AI MAX+ 395 (Strix Halo)
with 128 GB of unified memory — 64 GiB of it given to the GPU in the BIOS, leaving 62 GiB to the
system — and a Crucial 2 TB NVMe drive.

---

Your 128 GB machine can run DeepSeek V3.1. It just won't be fast — and you can know that in 3 seconds instead of after a 405 GB download.

I got tired of having no quick answer to "will this MoE run on my box?", so I wrote a tool that reads the GGUF **index** — a few hundred KB over a ranged HTTP request — and does the arithmetic.

For DeepSeek V3.1 Q4_K_M the index says something I found genuinely surprising:

- routed experts: **394 GB (97%)**, of which a token touches 8 of 256 → 12.3 GB
- attention + embeddings + shared expert: **11.4 GB**, needed by every token

Only 11.4 GB of a 405 GB model is needed by *every* token. That is what makes it runnable on hardware a sixth of its size: put those on the GPU, fill the rest of the GPU with whole layers of experts, let the page cache hold more, stream the remainder from NVMe.

On my machine (Ryzen AI MAX+ 395, 64 GiB VRAM + 62 GiB RAM, NVMe at 4.5 GB/s measured with O_DIRECT):

```
experts   394.0 GB, 7 layers on the GPU, 51 layers off it
fetched   8.9 GB per token from storage
runs at roughly 0.46-0.59 tokens per second
```

About 20 words a minute. A considered answer, not a conversation — but knowing that in 3 seconds beats finding out after the download. It also prints the llama.cpp flags for the placement it worked out.

```
pip install moe-fit
moefit plan <url to a .gguf>
```

What to trust: the structure is exact, read from the index. The speed is an estimate — a floor from bytes-per-token over measured read rate — printed as a range and labelled as one. `moefit verify` runs llama.cpp and compares. **I have validated the arithmetic, not yet the speed model across hardware.** If you run both, please open an issue with the two numbers; that is the contribution I'd most like.

No dependencies, Apache-2.0.

https://github.com/YauhenBichel/moe-fit

---

Hugging Face posts are capped at about 2,000 characters (the ten newest posts on the site top
out at 1,999), and there is no API for creating one — every plausible route answers 404, so a
post is made in the web interface. The long form above the cut is kept for the repository.
