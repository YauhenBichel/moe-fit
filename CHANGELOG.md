# Changelog

## 0.1.0 (2026-09-12)

- `moefit plan`, `inspect`, `bench`, `flags` and `verify`.
- Reads a GGUF index from a local file or over HTTP, following the shards of a split model, so a
  405 GB model can be measured before any of it is downloaded.
- Places tensors by the one asymmetry that matters in a mixture of experts: everything that every
  token needs goes on the GPU, and the experts are cached and streamed.
- Measures storage with `O_DIRECT` random reads at the size experts are fetched in, and says so
  when the kernel refuses `O_DIRECT`.
- Understands compressed-latent (MLA) and grouped-query KV caches.
