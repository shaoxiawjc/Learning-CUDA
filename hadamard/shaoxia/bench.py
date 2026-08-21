"""Minimal comparison: hadamard.cu (ours) vs fast-hadamard-transform.

Usage: .venv/bin/python bench.py
"""
import torch
from torch.utils.cpp_extension import load_inline
import fast_hadamard_transform as fht

DEVICE = "cuda"

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

ext = load_inline(
    name="ours_hadamard",
    cpp_sources=CPP_SRC,
    cuda_sources=open("hadamard.cu").read(),
    extra_cuda_cflags=[
        "-O3",
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


def time_fn(fn, iters=100, warmup=10):
    for _ in range(warmup):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def bench(dtype, rows, cols):
    x = torch.randn(rows, cols, device=DEVICE, dtype=dtype)
    out = torch.empty_like(x)

    ours = time_fn(lambda: OURS[dtype](x, out))
    theirs = time_fn(lambda: fht.hadamard_transform(x))
    gbps = lambda ms: 2 * rows * cols * x.element_size() / (ms * 1e-3) / 1e9
    print(f"  dtype={str(dtype)[6:]}, rows={rows:>6}, cols={cols:>5} | "
          f"ours {ours:8.3f} ms ({gbps(ours):7.1f} GB/s) | "
          f"fht  {theirs:8.3f} ms ({gbps(theirs):7.1f} GB/s) | "
          f"ours speedup {theirs / ours:6.2f}x")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("== correctness (vs fp32 y = x @ H) ==")
    cases = [
        (dtype, rows, cols) for dtype in (torch.float16, torch.bfloat16) for rows in (1024, 4096, 65536) for cols in (32, 64, 128, 256, 512, 1024)
    ]
    for dtype, rows, cols in cases:
        check_correctness(dtype, rows, cols, make_sylvester(cols))

    print("== timing ==")
    for dtype, rows, cols in cases:
        bench(dtype, rows, cols)
