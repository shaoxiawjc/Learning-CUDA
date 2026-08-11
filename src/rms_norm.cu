#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

#include "../tester/utils.h"
#include "./utils.h"

union alignas(16) PackHalf8 {
    float4 packed;
    __half2 data[4];
};

static constexpr size_t WARP_SIZE = 32;

// One warp handles one short row. Keeping all lanes active makes the shuffle
// reduction valid even when the row contains fewer than 32 elements.
template<typename T, size_t hidden_size>
__global__ void rms_norm_small_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output,
    float eps
) {
    const size_t col = threadIdx.x;
    const size_t offset = blockIdx.x * hidden_size + col;
    float value = 0.0f;
    if (col < hidden_size) {
        if constexpr (std::is_same_v<T, float>) {
            value = input[offset];
        } else {
            value = __half2float(input[offset]);
        }
    }

    float sum = value * value;
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, delta);
    }
    const float inv_rms = __shfl_sync(
        0xffffffff, rsqrtf(sum / static_cast<float>(hidden_size) + eps), 0);

    if (col < hidden_size) {
        if constexpr (std::is_same_v<T, float>) {
            output[offset] = value * inv_rms * weight[col];
        } else {
            output[offset] = __float2half_rn(
                value * inv_rms * __half2float(weight[col]));
        }
    }
}

template<size_t threads_per_block>
__device__ __forceinline__ float rms_inv_reduce(
    float sum, size_t hidden_dim, float eps,
    float* sdata, float* shared_rms_inv
) {
    const size_t lane_id = threadIdx.x % WARP_SIZE;
    const size_t warp_id = threadIdx.x / WARP_SIZE;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if constexpr (threads_per_block == WARP_SIZE) {
        if (lane_id == 0) {
            sum = rsqrtf(sum / static_cast<float>(hidden_dim) + eps);
        }
        return __shfl_sync(0xffffffff, sum, 0);
    } else {
        if (lane_id == 0) {
            sdata[warp_id] = sum;
        }
        __syncthreads();
        if (warp_id == 0) {
            sum = (threadIdx.x < threads_per_block / WARP_SIZE)
                      ? sdata[lane_id]
                      : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            if (lane_id == 0) {
                *shared_rms_inv =
                    rsqrtf(sum / static_cast<float>(hidden_dim) + eps);
            }
        }
        __syncthreads();
        return *shared_rms_inv;
    }
}

template<size_t threads_per_block, size_t items_per_thread = 0>
__global__ void rms_norm_fp32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    size_t rows, size_t hidden_dim, float eps
){
    constexpr size_t vec_size = 4;

    size_t num_vecs = hidden_dim / vec_size;
    size_t tid = threadIdx.x;
    size_t row = blockIdx.x;

    const float* row_input = input + row * hidden_dim;
    const float4* row_input_vec = reinterpret_cast<const float4*>(row_input);
    float* row_output = output + row * hidden_dim;
    float4* row_output_vec = reinterpret_cast<float4*>(row_output);

    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    __shared__ float sdata[WARP_SIZE];
    __shared__ float rms_inv;

    constexpr size_t cached_items = items_per_thread > 0 ? items_per_thread : 1;
    float4 cached_input[cached_items];
    float sum = 0.0f;
    if constexpr (items_per_thread > 0) {
        #pragma unroll
        for (size_t item = 0; item < items_per_thread; ++item) {
            const size_t i = tid + item * threads_per_block;
            if (i < num_vecs) {
                const float4 val = row_input_vec[i];
                cached_input[item] = val;
                sum += val.x * val.x + val.y * val.y +
                       val.z * val.z + val.w * val.w;
            }
        }
    } else {
        for (size_t i = tid; i < num_vecs; i += threads_per_block) {
            const float4 val = row_input_vec[i];
            sum += val.x * val.x + val.y * val.y +
                   val.z * val.z + val.w * val.w;
        }
    }

    const float row_rms_inv = rms_inv_reduce<threads_per_block>(
        sum, hidden_dim, eps, sdata, &rms_inv);

    if constexpr (items_per_thread > 0) {
        #pragma unroll
        for (size_t item = 0; item < items_per_thread; ++item) {
            const size_t i = tid + item * threads_per_block;
            if (i < num_vecs) {
                const float4 val = cached_input[item];
                const float4 w = weight_vec[i];
                float4 out;
                out.x = val.x * row_rms_inv * w.x;
                out.y = val.y * row_rms_inv * w.y;
                out.z = val.z * row_rms_inv * w.z;
                out.w = val.w * row_rms_inv * w.w;
                row_output_vec[i] = out;
            }
        }
    } else {
        for (size_t i = tid; i < num_vecs; i += threads_per_block) {
            const float4 val = row_input_vec[i];
            const float4 w = weight_vec[i];
            float4 out;
            out.x = val.x * row_rms_inv * w.x;
            out.y = val.y * row_rms_inv * w.y;
            out.z = val.z * row_rms_inv * w.z;
            out.w = val.w * row_rms_inv * w.w;
            row_output_vec[i] = out;
        }
    }
}


template<size_t threads_per_block, size_t items_per_thread = 0>
__global__ void rms_norm_fp16_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    size_t rows, size_t hidden_dim, float eps
){
    constexpr size_t vec_size = 8;

    size_t num_vecs = hidden_dim / vec_size;
    size_t tid = threadIdx.x;
    size_t row = blockIdx.x;
    if (row >= rows) return;

    const half* row_input = input + row * hidden_dim;
    half* row_output = output + row * hidden_dim;
    // LDST128BITS exposes a non-const float4 reference. These aliases are used
    // for loads only; input and weight remain logically read-only.
    half* row_input_128 = const_cast<half*>(row_input);
    half* weight_128 = const_cast<half*>(weight);
    __shared__ float sdata[WARP_SIZE];
    __shared__ float rms_inv;

    constexpr size_t cached_items = items_per_thread > 0 ? items_per_thread : 1;
    float4 cached_input[cached_items];
    float sum = 0.0f;
    if constexpr (items_per_thread > 0) {
        #pragma unroll
        for (size_t item = 0; item < items_per_thread; ++item) {
            const size_t i = tid + item * threads_per_block;
            if (i < num_vecs) {
                PackHalf8 val;
                val.packed = LDST128BITS(row_input_128[i * vec_size]);
                cached_input[item] = val.packed;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 value_f = __half22float2(val.data[j]);
                    sum += value_f.x * value_f.x;
                    sum += value_f.y * value_f.y;
                }
            }
        }
    } else {
        for (size_t i = tid; i < num_vecs; i += threads_per_block) {
            PackHalf8 val;
            val.packed = LDST128BITS(row_input_128[i * vec_size]);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 value_f = __half22float2(val.data[j]);
                sum += value_f.x * value_f.x;
                sum += value_f.y * value_f.y;
            }
        }
    }

    const float row_rms_inv = rms_inv_reduce<threads_per_block>(
        sum, hidden_dim, eps, sdata, &rms_inv);

    if constexpr (items_per_thread > 0) {
        #pragma unroll
        for (size_t item = 0; item < items_per_thread; ++item) {
            const size_t i = tid + item * threads_per_block;
            if (i < num_vecs) {
                PackHalf8 val;
                PackHalf8 w;
                val.packed = cached_input[item];
                w.packed = LDST128BITS(weight_128[i * vec_size]);
                PackHalf8 out;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 value_f = __half22float2(val.data[j]);
                    const float2 scale_f = __half22float2(w.data[j]);
                    out.data[j] = __floats2half2_rn(
                        value_f.x * row_rms_inv * scale_f.x,
                        value_f.y * row_rms_inv * scale_f.y
                    );
                }
                LDST128BITS(row_output[i * vec_size]) = out.packed;
            }
        }
    } else {
        for (size_t i = tid; i < num_vecs; i += threads_per_block) {
            PackHalf8 val;
            PackHalf8 w;
            val.packed = LDST128BITS(row_input_128[i * vec_size]);
            w.packed = LDST128BITS(weight_128[i * vec_size]);
            PackHalf8 out;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 value_f = __half22float2(val.data[j]);
                const float2 scale_f = __half22float2(w.data[j]);
                out.data[j] = __floats2half2_rn(
                    value_f.x * row_rms_inv * scale_f.x,
                    value_f.y * row_rms_inv * scale_f.y
                );
            }
            LDST128BITS(row_output[i * vec_size]) = out.packed;
        }
    }
}

// Scalar fallback kernels — used when hidden_dim is not divisible by vec_size
template<size_t threads_per_block, size_t items_per_thread = 0>
__global__ void rms_norm_fp32_scalar_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    size_t rows, size_t hidden_dim, float eps
){
    size_t tid = threadIdx.x;
    size_t lane_id = threadIdx.x % WARP_SIZE;
    size_t warp_id = threadIdx.x / WARP_SIZE;
    size_t row = blockIdx.x;
    if (row >= rows) return;

    const float* row_input = input + row * hidden_dim;
    float* row_output = output + row * hidden_dim;

    __shared__ float sdata[WARP_SIZE];
    __shared__ float rms_inv[1];

    constexpr size_t cached_items = items_per_thread > 0 ? items_per_thread : 1;
    float cached_input[cached_items];
    float sum = 0.0f;
    if constexpr (items_per_thread > 0) {
        #pragma unroll
        for (size_t item = 0; item < items_per_thread; ++item) {
            const size_t i = tid + item * threads_per_block;
            if (i < hidden_dim) {
                const float val = row_input[i];
                cached_input[item] = val;
                sum += val * val;
            }
        }
    } else {
        for (size_t i = tid; i < hidden_dim; i += threads_per_block) {
            const float val = row_input[i];
            sum += val * val;
        }
    }

    #pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, i);
    }

    if (lane_id == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0) {
        sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? sdata[lane_id] : 0.0f;
        #pragma unroll
        for (int i = 16; i > 0; i /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, i);
        }
        if (lane_id == 0) {
            rms_inv[0] = rsqrtf(sum / static_cast<float>(hidden_dim) + eps);
        }
    }
    __syncthreads();

    if constexpr (items_per_thread > 0) {
        #pragma unroll
        for (size_t item = 0; item < items_per_thread; ++item) {
            const size_t i = tid + item * threads_per_block;
            if (i < hidden_dim) {
                row_output[i] = cached_input[item] * rms_inv[0] * weight[i];
            }
        }
    } else {
        for (size_t i = tid; i < hidden_dim; i += threads_per_block) {
            row_output[i] = row_input[i] * rms_inv[0] * weight[i];
        }
    }
}

template<size_t threads_per_block, size_t items_per_thread = 0>
__global__ void rms_norm_fp16_scalar_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    size_t rows, size_t hidden_dim, float eps
){
    size_t tid = threadIdx.x;
    size_t lane_id = threadIdx.x % WARP_SIZE;
    size_t warp_id = threadIdx.x / WARP_SIZE;
    size_t row = blockIdx.x;
    if (row >= rows) return;

    const half* row_input = input + row * hidden_dim;
    half* row_output = output + row * hidden_dim;

    __shared__ float sdata[WARP_SIZE];
    __shared__ float rms_inv[1];

    constexpr size_t cached_items = items_per_thread > 0 ? items_per_thread : 1;
    half cached_input[cached_items];
    float sum = 0.0f;
    if constexpr (items_per_thread > 0) {
        #pragma unroll
        for (size_t item = 0; item < items_per_thread; ++item) {
            const size_t i = tid + item * threads_per_block;
            if (i < hidden_dim) {
                const half raw = row_input[i];
                const float val = __half2float(raw);
                cached_input[item] = raw;
                sum += val * val;
            }
        }
    } else {
        for (size_t i = tid; i < hidden_dim; i += threads_per_block) {
            const float val = __half2float(row_input[i]);
            sum += val * val;
        }
    }

    #pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, i);
    }

    if (lane_id == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0) {
        sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? sdata[lane_id] : 0.0f;
        #pragma unroll
        for (int i = 16; i > 0; i /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, i);
        }
        if (lane_id == 0) {
            rms_inv[0] = rsqrtf(sum / static_cast<float>(hidden_dim) + eps);
        }
    }
    __syncthreads();

    if constexpr (items_per_thread > 0) {
        #pragma unroll
        for (size_t item = 0; item < items_per_thread; ++item) {
            const size_t i = tid + item * threads_per_block;
            if (i < hidden_dim) {
                const float val = __half2float(cached_input[item]);
                const float w = __half2float(weight[i]);
                row_output[i] = __float2half(val * rms_inv[0] * w);
            }
        }
    } else {
        for (size_t i = tid; i < hidden_dim; i += threads_per_block) {
            const float val = __half2float(row_input[i]);
            const float w = __half2float(weight[i]);
            row_output[i] = __float2half(val * rms_inv[0] * w);
        }
    }
}
