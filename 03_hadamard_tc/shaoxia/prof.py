"""Launch a single hadamard (dtype, rows, cols) for ncu profiling.

Reuses the same load_inline extension as bench.py (already cached), so it does
not recompile. Keep the process minimal: one shape, a handful of launches.

Usage (run directly):
    .venv/bin/python prof.py --dtype fp16 --rows 1024 --cols 8192

Usage (under ncu, profiling only our kernel):
    ncu --kernel-name regex:hadamard_kernel --launch-count 1 --set full \
        .venv/bin/python prof.py --dtype fp16 --rows 1024 --cols 8192
"""
import os
import argparse

# Make .venv/bin tools (ninja/nvcc) visible to load_inline even under `sudo`,
# which resets PATH. Must run before `import bench` (which triggers load_inline).
_HERE = os.path.dirname(os.path.abspath(__file__))
_VENV_BIN = os.path.join(_HERE, ".venv", "bin")
if _VENV_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _VENV_BIN + os.pathsep + os.environ.get("PATH", "")

import torch

import bench  # noqa: F401  (loads the cached extension; defines bench.OURS)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # --impl is read by bench.py at *import* time (it compiles the extension with
    # -DHADAMARD_IMPL=<impl>), so it must be present in sys.argv before `import bench`
    # runs at the top of this module. Declaring it here is enough: the same sys.argv
    # is still intact when `import bench` executes, so bench picks it up automatically.
    p.add_argument("--impl", default=bench.DEFAULT_IMPL, choices=list(bench.IMPLS),
                   help="kernel implementation to compile: " +
                        "; ".join(f"{k}={v}" for k, v in bench.IMPLS.items()))
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--cols", type=int, required=True)
    p.add_argument("--iters", type=int, default=3,
                   help="kernel launches; raise it and use ncu --launch-skip to profile a warm launch")
    args = p.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    x = torch.randn(args.rows, args.cols, device="cuda", dtype=dtype)
    out = torch.empty_like(x)

    for _ in range(args.iters):
        bench.OURS[dtype](x, out)
    torch.cuda.synchronize()
    print(f"launched {args.iters}x hadamard {args.dtype} {args.rows}x{args.cols}")


if __name__ == "__main__":
    main()
