# Contributing

Thanks for looking. Issues and pull requests are both welcome.

## The most useful contribution

**Measurements.** The structural arithmetic is exact, but the speed estimate rests on an assumption
about how much the expert cache helps. If you run `moefit plan` and then `moefit verify` on real
hardware, please open an issue with both numbers, your model, and your storage. That is what turns
the estimate into something trustworthy — especially on hardware unlike mine (one AMD Strix Halo
box with an NVMe drive).

## Working on the code

```bash
uv run pytest -q        # under a second; no network, no model files
```

The tests build GGUF files byte by byte in `tests/test_gguf.py`, so a new field or quantisation can
be covered without downloading anything.

Please keep to the shape of what is there:

- **No dependencies.** The standard library has been enough so far.
- **A test for the behaviour you change**, and a comment saying *why* where the reason is not
  obvious from the code.
- **Claims carry their evidence.** If you state a number in a comment, a docstring or the README,
  say where it was measured. Several comments in this codebase exist because an earlier assumption
  turned out to be wrong.

## Adding a back end

`plan.llama_flags` is deliberately the only place that knows about `llama.cpp`. A back end for
vLLM, SGLang or ktransformers belongs beside it, with its own placement rules — those runtimes
split tensors differently and the arithmetic here should not be assumed to carry over.
