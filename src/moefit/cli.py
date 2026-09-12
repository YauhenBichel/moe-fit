# Copyright 2026 Yauhen Bichel
# SPDX-License-Identifier: Apache-2.0
"""moefit: will this mixture-of-experts model run on this machine, and how fast?

  moefit plan <gguf or url>     read the index, measure the machine, print the placement and speed
  moefit inspect <gguf or url>  what the model is made of, without judging the machine
  moefit bench                  measure this machine only
  moefit flags <gguf or url>    print the llama.cpp arguments for the plan
  moefit verify <gguf> [...]    run llama.cpp and compare the real speed with the estimate
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time

from . import gguf, machine, plan


def _human(gb: float) -> str:
    return f"{gb:,.1f} GB" if gb >= 1 else f"{gb * 1000:,.0f} MB"


def cmd_inspect(args: argparse.Namespace) -> int:
    model = gguf.read(args.model, shards=not args.no_shards)
    if args.json:
        print(json.dumps({"name": model.name, "architecture": model.architecture,
                          "layers": model.n_layers, "experts": model.n_experts,
                          "experts_used": model.n_experts_used, "is_moe": model.is_moe,
                          "total_gb": model.total_bytes / 1e9,
                          "expert_gb": model.expert_bytes / 1e9,
                          "resident_gb": model.resident_bytes / 1e9,
                          "read_per_token_gb": model.bytes_read_per_token() / 1e9,
                          "context_length": model.context_length}, indent=1))
        return 0
    print(f"{model.name}  ({model.architecture})")
    print(f"  {len(model.tensors):,} tensors, {model.n_layers} layers, "
          f"trained context {model.context_length:,}")
    print(f"  weights            {_human(model.total_bytes / 1e9)}")
    if model.is_moe:
        share = 100 * model.expert_bytes / model.total_bytes
        print(f"  routed experts     {_human(model.expert_bytes / 1e9)}  ({share:.0f}% of the model)")
        print(f"  everything else    {_human(model.resident_bytes / 1e9)}  "
              f"<- every token needs all of this")
        print(f"  experts            {model.n_experts}, of which {model.n_experts_used} run per token")
        print(f"  read per token     {_human(model.bytes_read_per_token() / 1e9)}")
    else:
        print("  not a mixture of experts: every token reads all of it")
    return 0


def _machine_for(args: argparse.Namespace) -> machine.Machine:
    return machine.describe(args.disk, measure=not args.no_measure,
                            block_bytes=args.block_size << 20)


def cmd_bench(args: argparse.Namespace) -> int:
    found = _machine_for(args)
    if args.json:
        print(json.dumps(found.as_dict(), indent=1))
        return 0
    print(f"  GPU memory         {_human(found.vram_bytes / 1e9)}")
    print(f"  system memory      {_human(found.ram_bytes / 1e9)}")
    print(f"  free disk          {_human(found.free_disk_bytes / 1e9)}  ({found.disk_path})")
    if found.read_bytes_per_second:
        print(f"  storage read       {found.read_bytes_per_second / 1e9:.2f} GB/s  "
              f"(random {args.block_size} MiB reads, page cache bypassed)")
    for note in found.notes:
        print(f"  note: {note}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    model = gguf.read(args.model, shards=not args.no_shards)
    found = _machine_for(args)
    placement = plan.make(model, found, context=args.context, kv_bits=args.kv_bits, skew=args.skew)

    if args.json:
        print(json.dumps({"model": model.name, "machine": found.as_dict(),
                          "placement": placement.__dict__}, indent=1, default=str))
        return 0 if placement.fits else 2

    print(f"{model.name} on this machine\n")
    print(f"  model              {_human(placement.model_gb)}")
    print(f"  needed every token {_human(placement.resident_gb)} of weights "
          f"+ {_human(placement.kv_cache_gb)} of KV cache at {placement.context:,} context")
    if not placement.fits:
        print(f"\n  WILL NOT RUN: {placement.reason}")
        return 2
    print(f"  experts            {_human(placement.expert_gb)}, "
          f"{placement.expert_layers_on_gpu} layers on the GPU, "
          f"{placement.expert_layers_on_cpu} layers off it")
    print(f"  kept in memory     {_human(placement.cached_expert_gb)} of experts")
    print(f"  fetched per token  {_human(placement.streamed_per_token_gb)} from storage "
          f"(of {_human(placement.read_per_token_gb)} read)")
    print(f"\n  {placement.summary()}")
    if placement.tokens_per_second_estimate:
        low, _ = placement.tokens_per_second_estimate
        if low < 5:
            words = int(low * 60 * 0.75)
            print(f"  that is about {words} words a minute: usable for a considered answer, "
                  f"not for a conversation")
        print("\n  This speed is an estimate from the measured read rate. "
              "Check it with: moefit verify")
    return 0


def cmd_flags(args: argparse.Namespace) -> int:
    model = gguf.read(args.model, shards=not args.no_shards)
    found = _machine_for(args)
    placement = plan.make(model, found, context=args.context, kv_bits=args.kv_bits, skew=args.skew)
    if not placement.fits:
        print(f"will not run: {placement.reason}", file=sys.stderr)
        return 2
    print(" ".join(plan.llama_flags(model, placement, args.model, args.context, args.kv_bits)))
    return 0


TOKENS_PER_SECOND = re.compile(r"(\d+\.\d+)\s*tokens per second|eval time.*?(\d+\.\d+)\s*tokens per second")


def cmd_verify(args: argparse.Namespace) -> int:
    """Run llama.cpp once and compare the measured speed with the estimate."""
    binary = args.llama or shutil.which("llama-cli") or shutil.which("llama-server")
    if not binary:
        print("llama-cli was not found; pass --llama /path/to/llama-cli", file=sys.stderr)
        return 2
    model = gguf.read(args.model, shards=not args.no_shards)
    found = _machine_for(args)
    placement = plan.make(model, found, context=args.context, kv_bits=args.kv_bits, skew=args.skew)
    if not placement.fits:
        print(f"will not run: {placement.reason}", file=sys.stderr)
        return 2

    flags = plan.llama_flags(model, placement, args.model, args.context, args.kv_bits)
    command = [binary, *flags, "-n", str(args.tokens), "-p", args.prompt, "--no-warmup"]
    print("running:", " ".join(command[:9]), "...\n")
    started = time.time()
    result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
    elapsed = time.time() - started

    measured = None
    for line in (result.stderr + result.stdout).splitlines():
        if "tokens per second" in line or "tokens/s" in line:
            numbers = re.findall(r"(\d+\.\d+)", line)
            if numbers and ("eval" in line.lower() or measured is None):
                measured = float(numbers[-1])
    if measured is None and elapsed > 0:
        measured = args.tokens / elapsed

    estimate = placement.tokens_per_second_estimate
    print(f"  estimated   {estimate[0]:.2f}-{estimate[1]:.2f} tokens per second" if estimate
          else "  estimated   unknown")
    print(f"  measured    {measured:.2f} tokens per second" if measured else "  measured    unknown")
    if estimate and measured:
        low, high = estimate
        verdict = "inside" if low <= measured <= high else ("faster than" if measured > high else "slower than")
        print(f"\n  the measurement is {verdict} the estimate")
    if result.returncode != 0:
        print(f"\n  llama.cpp exited {result.returncode}:\n{result.stderr[-600:]}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moefit", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_model(p: argparse.ArgumentParser) -> None:
        p.add_argument("model", help="a .gguf file, or an https URL to one (only the index is read)")
        p.add_argument("--no-shards", action="store_true",
                       help="do not follow the other shards of a split model")

    def add_machine(p: argparse.ArgumentParser) -> None:
        p.add_argument("--disk", default=None,
                       help="where the model will live; its read speed is what gets measured")
        p.add_argument("--no-measure", action="store_true", help="skip the storage measurement")
        p.add_argument("--block-size", type=int, default=8, metavar="MIB",
                       help="read size for the storage measurement (default 8)")

    def add_plan(p: argparse.ArgumentParser) -> None:
        p.add_argument("--context", type=int, default=None, help="context length to plan for")
        p.add_argument("--kv-bits", type=int, default=8, choices=(8, 16),
                       help="KV cache precision (default 8)")
        p.add_argument("--skew", type=float, default=1.6,
                       help="how much better than chance the expert cache does (default 1.6)")

    p = sub.add_parser("inspect", help="what the model is made of")
    add_model(p); p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("bench", help="measure this machine")
    add_machine(p); p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_bench)

    p = sub.add_parser("plan", help="placement and speed for this model on this machine")
    add_model(p); add_machine(p); add_plan(p)
    p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_plan)

    p = sub.add_parser("flags", help="the llama.cpp arguments for the plan")
    add_model(p); add_machine(p); add_plan(p); p.set_defaults(func=cmd_flags)

    p = sub.add_parser("verify", help="run llama.cpp and compare the real speed with the estimate")
    add_model(p); add_machine(p); add_plan(p)
    p.add_argument("--llama", default=None, help="path to llama-cli")
    p.add_argument("--tokens", type=int, default=32, help="tokens to generate (default 32)")
    p.add_argument("--prompt", default="Explain what a mixture of experts is, in two sentences.")
    p.add_argument("--timeout", type=int, default=3600)
    p.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except gguf.GGUFError as exc:
        print(f"moefit: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
