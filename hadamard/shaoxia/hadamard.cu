#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cassert>
#include <type_traits>

#define WARP_SIZE 32
#define LD128BIT_CONST(value) (*(reinterpret_cast<const float4*>(&value)))
#define LD128BIT(value) (*(reinterpret_cast<float4*>(&value)))


struct alignas(16) Half8 {
    __half x[8];
};

struct alignas(16) BFloat16_8 {
    __nv_bfloat16 x[8];
};

static_assert(sizeof(Half8) == 16);
static_assert(sizeof(BFloat16_8) == 16);

__device__ __forceinline__
void load_8_to_float8(
    const __half* ptr,
    float (&v)[8]
) {
    Half8 h = *reinterpret_cast<const Half8*>(ptr);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = __half2float(h.x[i]);
    }
}


__device__ __forceinline__
void load_8_to_float8(
    const __nv_bfloat16* ptr,
    float (&v)[8]
) {
    BFloat16_8 h =
        *reinterpret_cast<const BFloat16_8*>(ptr);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = __bfloat162float(h.x[i]);
    }
}



__device__ __forceinline__
void store_float8_as_8(
    const float (&v)[8],
    __half* ptr
) {
    __half2 h2[4];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        h2[i] = __floats2half2_rn(
            v[2 * i],
            v[2 * i + 1]
        );
    }

    uint4 packed =
        *reinterpret_cast<const uint4*>(h2);

    *reinterpret_cast<uint4*>(ptr) = packed;
}


__device__ __forceinline__
void store_float8_as_8(
    const float (&v)[8],
    __nv_bfloat16* ptr
) {
    __nv_bfloat162 b2[4];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        b2[i] = __floats2bfloat162_rn(
            v[2 * i],
            v[2 * i + 1]
        );
    }

    uint4 packed =
        *reinterpret_cast<const uint4*>(b2);

    *reinterpret_cast<uint4*>(ptr) = packed;
}

template <
    typename T,
    const int COLS,
    const int ROWS_PER_BLOCK
>
__global__ void hadamard_kernel_scalar(const T* input, T* output, const int rows) {
    static_assert(COLS >= WARP_SIZE, "COLS must be >= WARP_SIZE");
    constexpr int NUM_PER_THREAD = COLS / WARP_SIZE;
    float local[NUM_PER_THREAD];

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int row = blockIdx.x * ROWS_PER_BLOCK + warp_id;

    if (row >= rows) {
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
    #pragma unroll
    for (int i = 0 ; i < NUM_PER_THREAD; ++i) {
        output[row * COLS + lane_id + i * WARP_SIZE] = T(local[i]);
    }
}



template <
    typename T,
    const int COLS,
    const int ROWS_PER_BLOCK
>
__global__ void hadamard_kernel_vec(const T* input, T* output, const int rows) {
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
    
    #pragma unroll
    for (int i = 0 ; i < VEC_PER_THREAD; ++i) {
        store_float8_as_8(local[i], &output[row * COLS + lane_id * VEC_SIZE + i * WARP_SIZE * VEC_SIZE]);
    }
}


template <typename T>
void hadamard(const T* input, T* output, int rows, int cols,
              cudaStream_t stream = 0) {
    static_assert(std::is_same<T, __half>::value ||
                      std::is_same<T, __nv_bfloat16>::value,
                  "T must be __half or __nv_bfloat16");
    int log2_n = 0;
    while ((1 << log2_n) < cols) {
        ++log2_n;
    }
    assert((1 << log2_n) == cols && "cols must be a power of 2");

    constexpr int ROWS_PER_BLOCK = 4;
    constexpr int THREADS_PER_BLOCK = ROWS_PER_BLOCK * WARP_SIZE;
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);

    switch (cols) {
        case 32:
            hadamard_kernel_scalar<T, 32, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        case 64:
            hadamard_kernel_scalar<T, 64, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        case 128:
            hadamard_kernel_scalar<T, 128, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        case 256:
            hadamard_kernel_vec<T, 256, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        case 512:
            hadamard_kernel_vec<T, 512, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        case 1024:
            hadamard_kernel_vec<T, 1024, ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(input, output, rows);
            break;
        default:
            assert(false && "cols must be a power of 2 in [32, 1024]");
    }
}

template void hadamard<__half>(const __half*, __half*, int, int, cudaStream_t);
template void hadamard<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int, int, cudaStream_t);
