#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error__ = (call);                                      \
        if (error__ != cudaSuccess) {                                      \
            std::fprintf(                                                  \
                stderr,                                                    \
                "CUDA error at %s:%d\n"                                    \
                "  expression: %s\n"                                       \
                "  error code: %d\n"                                       \
                "  error name: %s\n"                                       \
                "  error message: %s\n",                                   \
                __FILE__,                                                  \
                __LINE__,                                                  \
                #call,                                                     \
                static_cast<int>(error__),                                 \
                cudaGetErrorName(error__),                                 \
                cudaGetErrorString(error__)                                \
            );                                                             \
            std::abort();                                                  \
        }                                                                  \
    } while (0)

__device__ __host__ inline
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

template <typename T, const int WarpSize>
__device__ inline T warp_reduce_max(T val) {
#pragma unroll
  for (int mask = WarpSize >> 1; mask >= 1; mask >>= 1) {
    val = max(val, __shfl_xor_sync(0xffffffff, val, mask, WarpSize));
  }
  return val;
}

template <typename T, const int WarpSize>
__device__ inline T warp_reduce_sum(T val) {
#pragma unroll
  for (int mask = WarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask, WarpSize);
  }
  return val;
}

#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n)                                                 \
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_CG(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))
#define LDMATRIX_X2(R0, R1, addr)                                              \
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"    \
               : "=r"(R0), "=r"(R1)                                            \
               : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr)                                            \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"       \
      : "=r"(R0), "=r"(R1)                                                     \
      : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                      \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"     \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
      : "r"(addr))
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)            \
  asm volatile(                                                                \
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, "  \
      "%4, %5}, {%6, %7}, {%8, %9};\n"                                         \
      : "=r"(RD0), "=r"(RD1)                                                   \
      : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0),  \
        "r"(RC1))
#define HMMA16832(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1,  \
                  RC2, RC3)                                                    \
  asm volatile(                                                                \
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, "   \
      "{%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"                    \
      : "=f"(RD0), "=f"(RD1), "=f"(RD2), "=f"(RD3)                             \
      : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "f"(RC0),  \
        "f"(RC1), "f"(RC2), "f"(RC3))