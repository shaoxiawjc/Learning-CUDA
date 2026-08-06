#!/usr/bin/env python3

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmark import summarize


# (case, batch, target length, source length, query heads, KV heads,
#  head dimension, causal). Cases 1-3 use head dimensions 1, 2, and 4,
# which are not supported by the official FlashAttention CUDA kernels.
CASES = (
    (4, 2, 16, 16, 16, 8, 8, True),
    (5, 1, 32, 32, 32, 16, 16, False),
    (6, 4, 64, 64, 64, 32, 32, True),
    (7, 1, 8, 8, 8, 2, 8, False),
    (8, 1, 8, 8, 8, 2, 8, True),
    (9, 2, 16, 16, 12, 3, 8, False),
    (10, 1, 64, 64, 16, 4, 8, True),
    (11, 1, 16, 32, 8, 4, 16, True),
    (12, 2, 32, 16, 16, 4, 16, False),
    (13, 2, 256, 256, 32, 32, 64, False),
    (14, 4, 512, 2048, 64, 64, 32, True),
)


def repository_revision(repository):
    if not (repository / ".git").is_dir():
        raise RuntimeError(
            f"official repository not found at {repository}; clone "
            "git@github.com:Dao-AILab/flash-attention.git there first"
        )
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the official Dao-AILab FlashAttention fp16 forward kernel"
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--repository", type=Path, default=Path("third_party/flash-attention")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/attention_flash_attn_baseline.json")
    )
    args = parser.parse_args()
    if min(args.runs, args.warmup, args.iterations) < 1:
        parser.error("--runs, --warmup, and --iterations must be at least 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    revision = repository_revision(args.repository)
    try:
        import flash_attn
        from flash_attn import flash_attn_func
    except ImportError as error:
        raise RuntimeError(
            "flash-attn is not installed; build the cloned repository with "
            "MAX_JOBS=4 pip install --no-build-isolation ./third_party/flash-attention"
        ) from error

    results = {}
    device = torch.device("cuda")
    for case, batch, target, source, q_heads, kv_heads, dim, causal in CASES:
        q = torch.randn(
            batch, target, q_heads, dim, device=device, dtype=torch.float16
        )
        k = torch.randn(
            batch, source, kv_heads, dim, device=device, dtype=torch.float16
        )
        v = torch.randn_like(k)

        for _ in range(args.warmup):
            flash_attn_func(q, k, v, dropout_p=0.0, causal=causal)
        torch.cuda.synchronize()

        samples = []
        for _ in range(args.runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                flash_attn_func(q, k, v, dropout_p=0.0, causal=causal)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / args.iterations)

        key = f"attention_case{case}_half"
        results[key] = summarize(samples)
        print(f"{key:<28} {results[key]['median_ms']:>12.6f} ms")

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": "official flash_attn.flash_attn_func forward, device tensors",
        "operator": "attention",
        "dtype": "half",
        "runs": args.runs,
        "iterations_per_run": args.iterations,
        "warmup_iterations": args.warmup,
        "repository": str(args.repository),
        "repository_revision": revision,
        "flash_attn_version": flash_attn.__version__,
        "flash_attn_module": str(Path(flash_attn.__file__).resolve()),
        "device": torch.cuda.get_device_name(device),
        "skipped_cases": {
            "attention_case1_half": "official kernel does not support head_dim=1",
            "attention_case2_half": "official kernel does not support head_dim=2",
            "attention_case3_half": "official kernel does not support head_dim=4",
        },
        "outlier_rule": "Tukey fences: Q1 - 1.5*IQR, Q3 + 1.5*IQR",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nBaseline saved to {args.output}")


if __name__ == "__main__":
    main()
