#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cassert>
#include <cstdint>
#include <type_traits>

#define WARP_SIZE 32

// ============================================================================
// Fused Hadamard transform + per-row quantization  (template)
// ============================================================================
//
//   input  : fp16 / bf16, shape [rows, cols]   (cols must be a power of 2)
//   output : int8,        shape [rows, cols]   (quantized result)
//   scale  : fp32,        shape [rows]         (per-row scale)
//
//   Per row r:
//       y        = FWHT(x[r])                  // natural-order Hadamard transform
//       scale[r] = f(y)                        // e.g. absmax(y) / 127
//       out[r][c] = round(y[c] / scale[r])     // clamped to int8
//
// The FWHT butterfly, the per-row scale reduction, and the exact quant scheme
// are left to you; the signatures, layout, load/store helpers, and dispatch
// glue below are ready to fill in. See the `TODO` blocks inside the kernel.
//
// Python side (torch.utils.cpp_extension.load_inline) needs a binding like:
//   void fuse_hadamard_per_row_quant_fp16(torch::Tensor x, torch::Tensor out, torch::Tensor scale) {
//       fused_hadamard_per_row_quant<__half>(
//           reinterpret_cast<const __half*>(x.data_ptr()),
//           reinterpret_cast<int8_t*>(out.data_ptr()),
//           scale.data_ptr<float>(),
//           static_cast<int>(x.size(0)), static_cast<int>(x.size(1)),
//           at::cuda::getCurrentCUDAStream().stream());
//   }
// (plus the __nv_bfloat16 twin), and m.def(...) for each.


// ---------- fp16 / bf16 -> float8 load helpers -------------------------------
struct alignas(16) Half8
{
    __half x[8];
};

struct alignas(16) BFloat16_8
{
    __nv_bfloat16 x[8];
};

static_assert(sizeof(Half8) == 16);
static_assert(sizeof(BFloat16_8) == 16);

__device__ __forceinline__ void load_8_to_float8(
    const __half *ptr,
    float (&v)[8])
{
    Half8 h = *reinterpret_cast<const Half8 *>(ptr);
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        v[i] = __half2float(h.x[i]);
    }
}

__device__ __forceinline__ void load_8_to_float8(
    const __nv_bfloat16 *ptr,
    float (&v)[8])
{
    BFloat16_8 h = *reinterpret_cast<const BFloat16_8 *>(ptr);
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        v[i] = __bfloat162float(h.x[i]);
    }
}



__device__ __forceinline__ void quantize_store_float8_as_int8(
    const float (&v)[8], float scale, int8_t *ptr)
{
    int8_t out[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        float q = rintf(v[i] / scale);
        q = fminf(127.0f, fmaxf(-127.0f, q)); // symmetric, no -128
        out[i] = (int8_t)q;
    }
    *reinterpret_cast<long long *>(ptr) = *reinterpret_cast<const long long *>(out);
}


__device__ __forceinline__ float warp_reduce_absmax(float v)
{
    v = fabsf(v);
#pragma unroll
    for (int j = WARP_SIZE / 2; j >= 1; j >>= 1)
    {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, j));
    }
    return v;
}


__device__ __forceinline__ int8_t quantize_float_to_int8(float v, float scale)
{
    float q = rintf(v / scale);
    q = fminf(127.0f, fmaxf(-127.0f, q));
    return (int8_t)q;
}

// Flattened index into the 1-D shared-memory exchange buffer. Mirrors the old
// 4-D layout smem[warp][vec][k][lane] (VEC_SIZE is fixed at 8). `vec_per_warp`
// is NUM_VEC (or CHUNKS_PER_ROUND when the exchange is batched in rounds). After
// the FWHT, the first WARPS_PER_ROW entries are reused for the per-warp absmax
// reduction, so one buffer serves both phases.
__device__ __forceinline__ int exchange_smem_idx(int warp_id, int vec, int k, int lane_id, int vec_per_warp)
{
    return (((warp_id * vec_per_warp + vec) * 8 + k) * WARP_SIZE + lane_id);
}


// One warp handles WARP_SIZE / COLS rows (COLS = 2/4/8/16); each row uses COLS
// lanes (one element per lane), so the FWHT butterfly is done with
// __shfl_xor_sync, followed by a per-row absmax scale and int8 quantize.
// Mirrors hadamard_kernel_small + per-row quant.
template <typename T, const int COLS>
__global__ void fuse_hadamard_per_row_quant_kernel_small(
    const T *__restrict__ input,
    int8_t *__restrict__ output,
    float *__restrict__ output_scale,
    const int rows)
{
    static_assert(COLS < WARP_SIZE && WARP_SIZE % COLS == 0,
                  "COLS must be a power of 2 dividing WARP_SIZE and < WARP_SIZE");
    constexpr int ROWS_PER_WARP = WARP_SIZE / COLS;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int lane_in_row = lane_id % COLS;
    const int row = (blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id) * ROWS_PER_WARP
                    + lane_id / COLS;

    const bool active = row < rows;
    float local = active ? float(input[row * COLS + lane_in_row]) : 0.0f;

    #pragma unroll
    for (int j = 1; j < COLS; j <<= 1) {
        float other = __shfl_xor_sync(0xffffffff, local, j);
        if (lane_in_row & j) {
            local = other - local;
        } else {
            local = local + other;
        }
    }

    // Per-row absmax: reduce over the COLS lanes of this row (XOR shuffle with
    // j < COLS stays within the row's lane group).
    float amax = fabsf(local);
    #pragma unroll
    for (int j = COLS >> 1; j >= 1; j >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, j));
    }
    float scale = amax / 127.0f;

    if (active) {
        output[row * COLS + lane_in_row] = quantize_float_to_int8(local, scale);
        if (lane_in_row == 0) {
            output_scale[row] = scale;
        }
    }
}


template <
    typename T,
    const int COLS,
    const int ROWS_PER_BLOCK
>
__global__ void fuse_hadamard_per_row_quant_kernel_1warp1row_scalar(
    const T *__restrict__ input,
    int8_t *__restrict__ output,
    float *__restrict__ output_scale,
    const int rows)
{
    static_assert(COLS >= WARP_SIZE, "COLS must be >= WARP_SIZE");
    constexpr int NUM_PER_THREAD = COLS / WARP_SIZE;
    float local[NUM_PER_THREAD];

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int row = blockIdx.x * ROWS_PER_BLOCK + warp_id;

    if (row >= rows)
    {
        return;
    }

    #pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; ++i) {
        local[i] = float(input[row * COLS + lane_id + i * WARP_SIZE]);
        #pragma unroll
        for (int j = 1 ; j < WARP_SIZE ; j <<= 1) {
            float other = __shfl_xor_sync(0xffffffff, local[i], j);
            if (lane_id & j) {
                local[i] = other - local[i];
            } else {
                local[i] = local[i] + other;
            }
        }
    }

    #pragma unroll
    for (int i = 1 ; i < NUM_PER_THREAD ; i <<= 1) {
        #pragma unroll
        for (int j = 0 ; j < NUM_PER_THREAD ; ++j) {
            if ((j & i) == 0) {
                float a = local[j];
                float b = local[j ^ i];
                local[j] = a + b;
                local[j ^ i] = a - b;
            }
        }
    }

    float amax = -1.0f;
    #pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; ++i){
        amax = fmaxf(fabsf(local[i]), amax);
    }
    amax = warp_reduce_absmax(amax);
    float scale = amax / 127;

    #pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; ++i)
    {
        output[row * COLS + lane_id + i * WARP_SIZE] = quantize_float_to_int8(local[i], scale);
    }

    if (lane_id == 0)
    {
        output_scale[row] = scale;
    }
}

template <
    typename T,
    const int COLS,
    const int ROWS_PER_BLOCK>
__global__ void fuse_hadamard_per_row_quant_kernel_1warp1row_vec(
    const T *__restrict__ input,
    int8_t *__restrict__ output,
    float *__restrict__ output_scale,
    const int rows)
{
    constexpr int VEC_SIZE = 8;
    constexpr int ELEMS_PER_WARP = VEC_SIZE * WARP_SIZE;
    constexpr int VEC_PER_THREAD = COLS / VEC_SIZE / WARP_SIZE;
    float local[VEC_PER_THREAD][8];
    static_assert(COLS >= ELEMS_PER_WARP,
              "COLS must be >= 256");
    static_assert(COLS % ELEMS_PER_WARP == 0,
                "COLS must be divisible by 256");
    static_assert((VEC_PER_THREAD & (VEC_PER_THREAD - 1)) == 0,
                "VEC_PER_THREAD must be power of two");

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int row = blockIdx.x * ROWS_PER_BLOCK + warp_id;

    if (row >= rows) {
        return;
    }

    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; ++i) {
        load_8_to_float8(&input[row * COLS + lane_id * VEC_SIZE + i * WARP_SIZE * VEC_SIZE], local[i]);
        #pragma unroll
        for (int j = 1 ; j < 8 ; j <<= 1) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                if (k & j){
                    float b = local[i][k];
                    float a = local[i][k ^ j];
                    local[i][k] = a - b;
                    local[i][k ^ j] = a + b;
                }   
            }
        }

        #pragma unroll
        for (int j = 1 ; j < WARP_SIZE ; j <<= 1) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float other = __shfl_xor_sync(0xffffffff, local[i][k], j);
                if (lane_id & j) {
                    local[i][k] = other - local[i][k];
                } else {
                    local[i][k] = local[i][k] + other;
                }
            }
        }
    }

    #pragma unroll
    for (int i = 1 ; i < VEC_PER_THREAD ; i <<= 1) {
        #pragma unroll
        for (int j = 0 ; j < VEC_PER_THREAD ; ++j) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                if ((j & i) == 0) {
                    float a = local[j][k];
                    float b = local[j ^ i][k];
                    local[j][k] = a + b;
                    local[j ^ i][k] = a - b;
                }
            }
            
        }
    }

    float amax = -1.0f;
    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; ++i)
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j)
            amax = fmaxf(fabsf(local[i][j]), amax);
    
    amax = warp_reduce_absmax(amax);
    float scale = amax / 127;

    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; ++i)
    {
        quantize_store_float8_as_int8(
            local[i], scale,
            &output[row * COLS + lane_id * VEC_SIZE + i * WARP_SIZE * VEC_SIZE]);
    }

    if (lane_id == 0)
    {
        output_scale[row] = scale;
    }

}

template <typename T, const int COLS, const int WARPS_PER_ROW>
__global__ void fuse_hadamard_per_row_quant_kernel_multi_warp_per_row(
    const T *__restrict__ input,
    int8_t *__restrict__ output,
    float *__restrict__ output_scale,
    const int rows)
{
    constexpr int VEC_SIZE = 8;
    constexpr int ELEMENT_NUM_PER_WARP = COLS / WARPS_PER_ROW;
    constexpr int NUM_VEC = ELEMENT_NUM_PER_WARP / WARP_SIZE / VEC_SIZE;
    // 1-D shared memory: cross-warp FWHT exchange first, then reused (its first
    // WARPS_PER_ROW entries) for the per-warp absmax reduction.
    constexpr int SMEM_FLOATS = WARPS_PER_ROW * NUM_VEC * VEC_SIZE * WARP_SIZE;
    __shared__ float smem[SMEM_FLOATS];

    const int row = blockIdx.x;
    if (row >= rows)
        return;
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int col_offset = warp_id * ELEMENT_NUM_PER_WARP;

    float local[NUM_VEC][VEC_SIZE];

    // FWHT: intra-vec -> intra-warp -> intra-thread -> cross-warp (smem).
    #pragma unroll
    for (int i = 0; i < NUM_VEC; ++i) {
        load_8_to_float8(&input[row * COLS + col_offset + i * WARP_SIZE * VEC_SIZE + lane_id * VEC_SIZE], local[i]);
        #pragma unroll
        for (int j = 1; j < VEC_SIZE; j <<= 1) {
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k) {
                if (k & j) {
                    float a = local[i][k ^ j];
                    float b = local[i][k];
                    local[i][k ^ j] = a + b;
                    local[i][k] = a - b;
                }
            }
        }
        #pragma unroll
        for (int j = 1; j < WARP_SIZE; j <<= 1) {
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k) {
                float other = __shfl_xor_sync(0xffffffff, local[i][k], j);
                if (lane_id & j) {
                    local[i][k] = other - local[i][k];
                } else {
                    local[i][k] = local[i][k] + other;
                }
            }
        }
    }

    #pragma unroll
    for (int i = 1; i < NUM_VEC; i <<= 1) {
        #pragma unroll
        for (int j = 0; j < NUM_VEC; ++j) {
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k) {
                if ((j & i) == 0) {
                    float a = local[j][k];
                    float b = local[j ^ i][k];
                    local[j][k] = a + b;
                    local[j ^ i][k] = a - b;
                }
            }
        }
    }

    for (int i = 0; i < NUM_VEC; ++i)
        for (int k = 0; k < VEC_SIZE; ++k)
            smem[exchange_smem_idx(warp_id, i, k, lane_id, NUM_VEC)] = local[i][k];
    __syncthreads();

    #pragma unroll
    for (int i = 1; i < WARPS_PER_ROW; i <<= 1) {
        #pragma unroll
        for (int j = 0; j < NUM_VEC; ++j) {
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k) {
                float other = smem[exchange_smem_idx(warp_id ^ i, j, k, lane_id, NUM_VEC)];
                if (warp_id & i) {
                    local[j][k] = other - local[j][k];
                } else {
                    local[j][k] = local[j][k] + other;
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < NUM_VEC; ++j) {
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k) {
                smem[exchange_smem_idx(warp_id, j, k, lane_id, NUM_VEC)] = local[j][k];
            }
        }
        __syncthreads();
    }

    // Per-row scale: warp absmax, then cross-warp max via the now-free smem prefix.
    float amax = -1.0f;
    #pragma unroll
    for (int i = 0; i < NUM_VEC; ++i)
        #pragma unroll
        for (int k = 0; k < VEC_SIZE; ++k)
            amax = fmaxf(fabsf(local[i][k]), amax);
    amax = warp_reduce_absmax(amax);

    if (lane_id == 0) {
        smem[warp_id] = amax;
    }
    __syncthreads();

    float scale;
    if (warp_id == 0) {
        // 0.0f sentinel: values are already non-negative absmax, so a plain max
        // (warp_reduce_absmax's fabsf is a no-op) keeps invalid lanes from
        // polluting the result.
        float w = (lane_id < WARPS_PER_ROW) ? smem[lane_id] : 0.0f;
        w = warp_reduce_absmax(w);
        if (lane_id == 0) {
            smem[0] = w / 127.0f;
        }
    }
    __syncthreads();
    scale = smem[0];

    #pragma unroll
    for (int i = 0; i < NUM_VEC; ++i) {
        quantize_store_float8_as_int8(
            local[i], scale,
            &output[row * COLS + col_offset + i * WARP_SIZE * VEC_SIZE + lane_id * VEC_SIZE]);
    }

    if (tid == 0) {
        output_scale[row] = scale;
    }
}


template <typename T, const int COLS, const int WARPS_PER_ROW>
__global__ void fuse_hadamard_per_row_quant_kernel_multi_warp_per_row_chunked(
    const T *__restrict__ input,
    int8_t *__restrict__ output,
    float *__restrict__ output_scale,
    const int rows)
{
    constexpr int VEC_SIZE = 8;
    constexpr int ELEMENT_NUM_PER_WARP = COLS / WARPS_PER_ROW;
    constexpr int NUM_VEC = ELEMENT_NUM_PER_WARP / WARP_SIZE / VEC_SIZE;
    // Fixed 32 KB of shared memory: the cross-warp exchange is batched in rounds
    // of CHUNKS_PER_ROUND chunks; after the FWHT the buffer prefix is reused for
    // the per-warp absmax reduction.
    constexpr int SMEM_BYTES = 32 * 1024;
    constexpr int CHUNKS_PER_ROUND = SMEM_BYTES / (WARPS_PER_ROW * VEC_SIZE * WARP_SIZE * (int)sizeof(float));
    constexpr int NUM_ROUNDS = NUM_VEC / CHUNKS_PER_ROUND;
    static_assert(CHUNKS_PER_ROUND >= 1, "CHUNKS_PER_ROUND must be >= 1");
    static_assert(NUM_VEC % CHUNKS_PER_ROUND == 0, "NUM_VEC must be divisible by CHUNKS_PER_ROUND");
    constexpr int SMEM_FLOATS = WARPS_PER_ROW * CHUNKS_PER_ROUND * VEC_SIZE * WARP_SIZE;
    __shared__ float smem[SMEM_FLOATS];

    const int row = blockIdx.x;
    if (row >= rows)
        return;
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int col_offset = warp_id * ELEMENT_NUM_PER_WARP;

    float local[NUM_VEC][VEC_SIZE];

    #pragma unroll
    for (int i = 0; i < NUM_VEC; ++i) {
        load_8_to_float8(&input[row * COLS + col_offset + i * WARP_SIZE * VEC_SIZE + lane_id * VEC_SIZE], local[i]);
        #pragma unroll
        for (int j = 1; j < VEC_SIZE; j <<= 1) {
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k) {
                if (k & j) {
                    float a = local[i][k ^ j];
                    float b = local[i][k];
                    local[i][k ^ j] = a + b;
                    local[i][k] = a - b;
                }
            }
        }
        #pragma unroll
        for (int j = 1; j < WARP_SIZE; j <<= 1) {
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k) {
                float other = __shfl_xor_sync(0xffffffff, local[i][k], j);
                if (lane_id & j) {
                    local[i][k] = other - local[i][k];
                } else {
                    local[i][k] = local[i][k] + other;
                }
            }
        }
    }

    #pragma unroll
    for (int i = 1; i < NUM_VEC; i <<= 1) {
        #pragma unroll
        for (int j = 0; j < NUM_VEC; ++j) {
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k) {
                if ((j & i) == 0) {
                    float a = local[j][k];
                    float b = local[j ^ i][k];
                    local[j][k] = a + b;
                    local[j ^ i][k] = a - b;
                }
            }
        }
    }

    // Cross-warp exchange, batched so smem stays fixed at 32 KB.
    #pragma unroll
    for (int r = 0; r < NUM_ROUNDS; ++r) {
        const int base = r * CHUNKS_PER_ROUND;
        #pragma unroll
        for (int c = 0; c < CHUNKS_PER_ROUND; ++c)
            #pragma unroll
            for (int k = 0; k < VEC_SIZE; ++k)
                smem[exchange_smem_idx(warp_id, c, k, lane_id, CHUNKS_PER_ROUND)] = local[base + c][k];
        __syncthreads();

        #pragma unroll
        for (int i = 1; i < WARPS_PER_ROW; i <<= 1) {
            #pragma unroll
            for (int c = 0; c < CHUNKS_PER_ROUND; ++c) {
                #pragma unroll
                for (int k = 0; k < VEC_SIZE; ++k) {
                    float other = smem[exchange_smem_idx(warp_id ^ i, c, k, lane_id, CHUNKS_PER_ROUND)];
                    if (warp_id & i) {
                        local[base + c][k] = other - local[base + c][k];
                    } else {
                        local[base + c][k] = local[base + c][k] + other;
                    }
                }
            }
            __syncthreads();
            #pragma unroll
            for (int c = 0; c < CHUNKS_PER_ROUND; ++c) {
                #pragma unroll
                for (int k = 0; k < VEC_SIZE; ++k) {
                    smem[exchange_smem_idx(warp_id, c, k, lane_id, CHUNKS_PER_ROUND)] = local[base + c][k];
                }
            }
            __syncthreads();
        }
    }

    // Per-row scale (same as the non-chunked multi-warp kernel).
    float amax = -1.0f;
    #pragma unroll
    for (int i = 0; i < NUM_VEC; ++i)
        #pragma unroll
        for (int k = 0; k < VEC_SIZE; ++k)
            amax = fmaxf(fabsf(local[i][k]), amax);
    amax = warp_reduce_absmax(amax);

    if (lane_id == 0) {
        smem[warp_id] = amax;
    }
    __syncthreads();

    float scale;
    if (warp_id == 0) {
        float w = (lane_id < WARPS_PER_ROW) ? smem[lane_id] : 0.0f;
        w = warp_reduce_absmax(w);
        if (lane_id == 0) {
            smem[0] = w / 127.0f;
        }
    }
    __syncthreads();
    scale = smem[0];

    #pragma unroll
    for (int i = 0; i < NUM_VEC; ++i) {
        quantize_store_float8_as_int8(
            local[i], scale,
            &output[row * COLS + col_offset + i * WARP_SIZE * VEC_SIZE + lane_id * VEC_SIZE]);
    }

    if (tid == 0) {
        output_scale[row] = scale;
    }
}


// ---------- host launcher ----------------------------------------------------
template <typename T>
void fused_hadamard_per_row_quant(
    const T *input, int8_t *output, float *output_scale,
    int rows, int cols, cudaStream_t stream = 0)
{
    static_assert(std::is_same<T, __half>::value ||
                      std::is_same<T, __nv_bfloat16>::value,
                  "T must be __half or __nv_bfloat16");
    assert(cols > 0 && (cols & (cols - 1)) == 0 && "cols must be a power of 2");

    // Dispatch on cols exactly like hadamard_v4, mapping each tier to its fused
    // hadamard + per-row-quant kernel.
    if (cols < WARP_SIZE) {
        constexpr int THREADS_PER_BLOCK = 256;
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid((rows + THREADS_PER_BLOCK / cols - 1) / (THREADS_PER_BLOCK / cols));
        switch (cols) {
        case 2:
            fuse_hadamard_per_row_quant_kernel_small<T, 2><<<grid, block, 0, stream>>>(input, output, output_scale, rows);
            break;
        case 4:
            fuse_hadamard_per_row_quant_kernel_small<T, 4><<<grid, block, 0, stream>>>(input, output, output_scale, rows);
            break;
        case 8:
            fuse_hadamard_per_row_quant_kernel_small<T, 8><<<grid, block, 0, stream>>>(input, output, output_scale, rows);
            break;
        case 16:
            fuse_hadamard_per_row_quant_kernel_small<T, 16><<<grid, block, 0, stream>>>(input, output, output_scale, rows);
            break;
        default:
            assert(false && "cols not yet handled; extend the switch");
        }
    } else if (cols < 256) {
        constexpr int ROWS_PER_BLOCK = 4;
        constexpr int THREADS_PER_BLOCK = ROWS_PER_BLOCK * WARP_SIZE;
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid((rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
        switch (cols) {
        case 32:
            fuse_hadamard_per_row_quant_kernel_1warp1row_scalar<T, 32, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, output_scale, rows);
            break;
        case 64:
            fuse_hadamard_per_row_quant_kernel_1warp1row_scalar<T, 64, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, output_scale, rows);
            break;
        case 128:
            fuse_hadamard_per_row_quant_kernel_1warp1row_scalar<T, 128, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, output_scale, rows);
            break;
        default:
            assert(false && "cols not yet handled; extend the switch");
        }
    } else if (cols <= 2048) {
        constexpr int ROWS_PER_BLOCK = 4;
        constexpr int THREADS_PER_BLOCK = ROWS_PER_BLOCK * WARP_SIZE;
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid((rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
        switch (cols) {
        case 256:
            fuse_hadamard_per_row_quant_kernel_1warp1row_vec<T, 256, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, output_scale, rows);
            break;
        case 512:
            fuse_hadamard_per_row_quant_kernel_1warp1row_vec<T, 512, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, output_scale, rows);
            break;
        case 1024:
            fuse_hadamard_per_row_quant_kernel_1warp1row_vec<T, 1024, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, output_scale, rows);
            break;
        case 2048:
            fuse_hadamard_per_row_quant_kernel_1warp1row_vec<T, 2048, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, output_scale, rows);
            break;
        default:
            assert(false && "cols not yet handled; extend the switch");
        }
    } else if (cols <= 8192) {
        constexpr int WARPS_PER_ROW = 4;
        constexpr int THREADS_PER_BLOCK = WARPS_PER_ROW * WARP_SIZE;
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(rows);
        switch (cols) {
        case 4096:
            fuse_hadamard_per_row_quant_kernel_multi_warp_per_row<T, 4096, WARPS_PER_ROW><<<grid, block, 0, stream>>>(input, output, output_scale, rows);
            break;
        case 8192:
            fuse_hadamard_per_row_quant_kernel_multi_warp_per_row<T, 8192, WARPS_PER_ROW><<<grid, block, 0, stream>>>(input, output, output_scale, rows);
            break;
        default:
            assert(false && "cols not yet handled; extend the switch");
        }
    } else {
        switch (cols) {
        case 16384: {
            constexpr int WARPS_PER_ROW = 8;
            constexpr int THREADS_PER_BLOCK = WARPS_PER_ROW * WARP_SIZE;
            dim3 block(THREADS_PER_BLOCK);
            dim3 grid(rows);
            fuse_hadamard_per_row_quant_kernel_multi_warp_per_row_chunked<T, 16384, WARPS_PER_ROW><<<grid, block, 0, stream>>>(input, output, output_scale, rows);
            break;
        }
        case 32768: {
            constexpr int WARPS_PER_ROW = 16;
            constexpr int THREADS_PER_BLOCK = WARPS_PER_ROW * WARP_SIZE;
            dim3 block(THREADS_PER_BLOCK);
            dim3 grid(rows);
            fuse_hadamard_per_row_quant_kernel_multi_warp_per_row_chunked<T, 32768, WARPS_PER_ROW><<<grid, block, 0, stream>>>(input, output, output_scale, rows);
            break;
        }
        default:
            assert(false && "cols not yet handled; extend the switch");
        }
    }
}

template void fused_hadamard_per_row_quant<__half>(const __half *, int8_t *, float *, int, int, cudaStream_t);
template void fused_hadamard_per_row_quant<__nv_bfloat16>(const __nv_bfloat16 *, int8_t *, float *, int, int, cudaStream_t);
