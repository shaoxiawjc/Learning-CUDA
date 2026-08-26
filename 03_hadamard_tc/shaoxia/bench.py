"""Minimal comparison: hadamard.cu (ours) vs fast-hadamard-transform.

Usage: .venv/bin/python bench.py
"""
import torch
from torch.utils.cpp_extension import load_inline
import fast_hadamard_transform as fht
import fast_hadamard_transform_cuda as fht_cuda
import argparse
import json
import math
import os
import time
from datetime import datetime

DEVICE = "cuda"

# Available kernel implementations. The key is what `--impl` accepts and is passed
# to nvcc as -DHADAMARD_IMPL=<key>; each key must have a matching `hadamard_v<key>`
# launcher and an `#elif HADAMARD_IMPL == <key>` branch in hadamard.cu.
# To add an implementation: 1) add an entry here, 2) add the launcher + branch in hadamard.cu.
IMPLS = {
    "1": "scalar warp-shuffle (hadamard_v1)",
    "2": "vectorized 8-wide (hadamard_v2)",
    "3": "multi-warp-per-row (hadamard_v3)",
    "4": "auto-select by shape (hadamard_v4)",
}
DEFAULT_IMPL = "2"

# ---- build our kernel extension (load_inline caches in ~/.cache/torch_extensions)
CPP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T> void hadamard(const T*, T*, int, int, cudaStream_t);

void hadamard_fp16(torch::Tensor x, torch::Tensor out) {
    hadamard<__half>(
        reinterpret_cast<const __half*>(x.data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()),
        static_cast<int>(x.size(0)), static_cast<int>(x.size(1)),
        at::cuda::getCurrentCUDAStream().stream());
}
void hadamard_bf16(torch::Tensor x, torch::Tensor out) {
    hadamard<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        static_cast<int>(x.size(0)), static_cast<int>(x.size(1)),
        at::cuda::getCurrentCUDAStream().stream());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hadamard_fp16", &hadamard_fp16);
    m.def("hadamard_bf16", &hadamard_bf16);
}
"""

# --impl must be read before load_inline (the extension compiles at import time).
_impl_parser = argparse.ArgumentParser(add_help=False)
_impl_parser.add_argument("--impl", default=DEFAULT_IMPL, choices=list(IMPLS))
_impl, _ = _impl_parser.parse_known_args()

ext = load_inline(
    name="ours_hadamard",
    cpp_sources=CPP_SRC,
    cuda_sources=open("hadamard.cu").read(),
    extra_cuda_cflags=[
        "-O3",
        f"-DHADAMARD_IMPL={_impl.impl}",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ],
    verbose=False,
)

OURS = {torch.float16: ext.hadamard_fp16, torch.bfloat16: ext.hadamard_bf16}


def make_sylvester(n):
    """H_n with H[m][k] = (-1)^popcount(m & k), i.e. y = x @ H."""
    k = torch.arange(n, dtype=torch.int64, device=DEVICE)
    a = k[:, None] & k[None, :]
    tab = torch.tensor([bin(i).count("1") & 1 for i in range(256)],
                       dtype=torch.int64, device=DEVICE)
    parity = (tab[a & 0xFF] ^ tab[(a >> 8) & 0xFF] ^ tab[(a >> 16) & 0xFF]).bool()
    H = torch.ones(n, n, device=DEVICE)
    H[parity] = -1.0
    return H


# requirement: max abs error of ours vs fht (fp16: 1e-2, bf16: 5e-2)
MAX_ERR_VS_FHT = {torch.float16: 1e-2, torch.bfloat16: 5e-2}


def check_correctness(dtype, rows, cols, H):
    x = torch.randn(rows, cols, device=DEVICE, dtype=dtype)
    ref = x.float() @ H

    ours = torch.empty_like(x)
    OURS[dtype](x, ours)

    theirs = fht.hadamard_transform(x)

    def err(y):
        return (y.float() - ref).abs().max().item()

    err_vs_fht = (ours.float() - theirs.float()).abs().max().item()
    tol = MAX_ERR_VS_FHT[dtype]
    ok = "PASS" if err_vs_fht <= tol else "FAIL"

    print(f"  dtype={str(dtype)[6:]}, rows={rows}, cols={cols}")
    print(f"    max|ours  - ref| = {err(ours):.4e}")
    print(f"    max|fht   - ref| = {err(theirs):.4e}")
    print(f"    max|ours  - fht| = {err_vs_fht:.4e}  (requirement <= {tol:.0e}) [{ok}]")
    assert err_vs_fht <= tol, (
        f"{dtype}: max|ours - fht| = {err_vs_fht:.4e} exceeds requirement {tol:.0e}"
    )


def time_fn(fn, iters=100, warmup_ms=200.0):
    """Time `fn` (ms/iter), warming up until at least warmup_ms of GPU time has run."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()
    while True:
        fn()
        end.record()
        torch.cuda.synchronize()
        if start.elapsed_time(end) >= warmup_ms:
            break

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def warmup_gpu(seconds=1.0):
    """Sustained dummy work to ramp the GPU clock out of its idle/low-power state."""
    a = torch.randn(2048, 2048, device=DEVICE)
    b = torch.randn(2048, 2048, device=DEVICE)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        a = a @ b
    torch.cuda.synchronize()
    del a, b
    torch.cuda.empty_cache()


def load_results(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_results(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def bench(dtype, rows, cols):
    x = torch.randn(rows, cols, device=DEVICE, dtype=dtype)
    out = torch.empty_like(x)

    ours = time_fn(lambda: OURS[dtype](x, out))
    # Raw CUDA entry: skip fht's autograd.Function + Python wrapper overhead so we
    # time the kernel, not the call glue. scale=1.0 matches the default.
    theirs = time_fn(lambda: fht_cuda.fast_hadamard_transform(x, 1.0))
    bytes_moved = 2 * rows * cols * x.element_size()  # read + write
    flops = rows * cols * int(math.log2(cols))        # FWHT: N log2(N) per row
    gbps = lambda ms: bytes_moved / (ms * 1e-3) / 1e9
    gflops = lambda ms: flops / (ms * 1e-3) / 1e9
    speedup = theirs / ours
    print(f"  dtype={str(dtype)[6:]}, rows={rows:>6}, cols={cols:>5} | "
          f"ours {ours:8.3f} ms ({gflops(ours):7.1f} GFLOP/s) | "
          f"fht  {theirs:8.3f} ms ({gflops(theirs):7.1f} GFLOP/s) | "
          f"speedup {speedup:6.2f}x")
    return {
        "dtype": str(dtype)[6:],
        "rows": rows,
        "cols": cols,
        "ours_ms": ours,
        "fht_ms": theirs,
        "ours_gbps": gbps(ours),
        "fht_gbps": gbps(theirs),
        "ours_gflops": gflops(ours),
        "fht_gflops": gflops(theirs),
        "speedup": speedup,
    }


if __name__ == "__main__":
    torch.manual_seed(0)

    # rows / dims are split into three magnitude levels so you can target a regime.
    ROWS = {
        "small":  [1, 4, 16, 32, 64, 128, 256, 512],
        "middle": [1024, 2048, 4096, 8192],
        "large":  [16384, 32768, 65536],
    }
    DIMS = {
        "small":  [2, 4, 8, 16, 32, 64, 128, 256],
        "middle": [512, 1024, 2048, 4096, 8192],
        "large":  [16384, 32768, 65536],
    }

    def resolve(spec, levels):
        """'small,middle' / 'all' / concrete ints (e.g. '1024,65536') -> list of ints.

        Tokens may be level names, 'all', or integers; duplicates are dropped.
        """
        out = []
        for t in spec.split(","):
            t = t.strip()
            if t == "all":
                for k in levels:
                    out.extend(levels[k])
            elif t in levels:
                out.extend(levels[t])
            elif t.isdigit():
                out.append(int(t))
            else:
                raise SystemExit(
                    f"unknown level/value '{t}'; use small/middle/large, 'all', or an integer")
        seen, result = set(), []
        for v in out:
            if v not in seen:
                seen.add(v)
                result.append(v)
        return result

    parser = argparse.ArgumentParser(
        description="Benchmark hadamard.cu (ours) vs fast-hadamard-transform.")
    parser.add_argument("--impl", default=DEFAULT_IMPL, choices=list(IMPLS),
                        help="kernel implementation to compile: " +
                             "; ".join(f"{k}={v}" for k, v in IMPLS.items()))
    parser.add_argument("--rows", default="small",
                        help="comma list of row levels (small/middle/large), 'all', "
                             "or concrete ints (e.g. '1024,65536')")
    parser.add_argument("--dims", default="small",
                        help="comma list of dim levels (small/middle/large), 'all', "
                             "or concrete ints (e.g. '128,512')")
    parser.add_argument("--dtype", default="all", choices=["fp16", "bf16", "all"],
                        help="which dtype(s) to run")
    parser.add_argument("--no-check", action="store_true", help="skip the correctness pass")
    parser.add_argument("--no-bench", action="store_true", help="skip the timing pass")
    parser.add_argument("--name", default=None,
                        help="save timing results under this name in --result-file")
    parser.add_argument("--result-file", default="bench_results.json",
                        help="JSON file to write results into")
    args = parser.parse_args()

    rows = resolve(args.rows, ROWS)
    dims = resolve(args.dims, DIMS)
    dtypes = {"fp16": (torch.float16,), "bf16": (torch.bfloat16,),
              "all": (torch.float16, torch.bfloat16)}[args.dtype]

    # The correctness reference builds an n x n Sylvester matrix on device, so we
    # only check dims that keep that matrix reasonable (same cap as before).
    MAX_CHECK_DIM = 8192
    check_dims = [d for d in dims if d <= MAX_CHECK_DIM]
    skipped = [d for d in dims if d > MAX_CHECK_DIM]

    if not args.no_check:
        print("== correctness (vs fp32 y = x @ H) ==")
        for cols in check_dims:
            H = make_sylvester(cols)
            for dtype in dtypes:
                for r in rows:
                    check_correctness(dtype, r, cols, H)
        if skipped:
            print(f"  (skipped dims {skipped}: reference needs an n x n Sylvester matrix)")

    if not args.no_bench:
        print("== timing ==")
        warmup_gpu()
        results = []
        for dtype in dtypes:
            for r in rows:
                for cols in dims:
                    results.append(bench(dtype, r, cols))

        if args.name:
            data = load_results(args.result_file)
            data[args.name] = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "gpu": torch.cuda.get_device_name(0),
                "results": results,
            }
            save_results(args.result_file, data)
            print(f"\nsaved {len(results)} result(s) to '{args.result_file}' "
                  f"under name '{args.name}'")
