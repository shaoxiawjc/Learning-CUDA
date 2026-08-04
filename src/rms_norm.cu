#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../tester/utils.h"
#include "./utils.h"

struct PackHalf8 {
    __half2 data[4];
};

static constexpr size_t WARP_SIZE = 32;

template<size_t threads_per_block>
__global__ void rms_norm_fp32_kernel(
    const float* input, const float* weight,
    float* output, size_t rows, size_t hidden_dim, float eps
){
    constexpr size_t vec_size = 4;

    size_t num_vecs = hidden_dim / vec_size;
    size_t tid = threadIdx.x;
    size_t lane_id = threadIdx.x % WARP_SIZE;
    size_t warp_id = threadIdx.x / WARP_SIZE;
    size_t row = blockIdx.x;
    if (row >= rows) return;

    const float* row_input = input + row * hidden_dim;
    const float4* row_input_vec = reinterpret_cast<const float4*>(row_input);
    float* row_output = output + row * hidden_dim;
    float4* row_output_vec = reinterpret_cast<float4*>(row_output);

    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    __shared__ float sdata[WARP_SIZE];
    __shared__ float rms_inv[1];

    float sum = 0.0f;
    for(size_t i = tid; i < num_vecs; i += threads_per_block){
        float4 val = row_input_vec[i];
        sum += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }
    size_t tail = num_vecs * vec_size + tid;
    if (tail < hidden_dim) {
        float value = row_input[tail];
        sum += value * value;
    }

    // warp reduce
#pragma unroll
    for (int i = 16 ; i > 0; i /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, i);
    }

    if (lane_id == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0) {
        sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? sdata[lane_id] : 0.0f;
#pragma unroll
        for (int i = 16 ; i > 0; i /= 2){
            sum += __shfl_down_sync(0xffffffff, sum, i);
        }
        if (lane_id == 0) {
            rms_inv[0] = rsqrtf(sum / static_cast<float>(hidden_dim) + eps);
        }
    }
    __syncthreads();

    for (size_t i = tid; i < num_vecs; i += threads_per_block) {
        float4 val = row_input_vec[i];
        float4 w = weight_vec[i];
        float4 out;
        out.x = val.x * rms_inv[0] * w.x;
        out.y = val.y * rms_inv[0] * w.y;
        out.z = val.z * rms_inv[0] * w.z;
        out.w = val.w * rms_inv[0] * w.w;
        row_output_vec[i] = out;
    }
    if (tail < hidden_dim) {
        row_output[tail] = row_input[tail] * rms_inv[0] * weight[tail];
    }
}


template<size_t threads_per_block>
__global__ void rms_norm_fp16_kernel(
    const half* input, const half* weight,
    half* output, size_t rows, size_t hidden_dim, float eps
){
    constexpr size_t vec_size = 8;

    size_t num_vecs = hidden_dim / vec_size;
    size_t tid = threadIdx.x;
    size_t lane_id = threadIdx.x % WARP_SIZE;
    size_t warp_id = threadIdx.x / WARP_SIZE;
    size_t row = blockIdx.x;
    if (row >= rows) return;

    const half* row_input = input + row * hidden_dim;
    const PackHalf8* row_input_vec = reinterpret_cast<const PackHalf8*>(row_input);
    half* row_output = output + row * hidden_dim;
    PackHalf8* row_output_vec = reinterpret_cast<PackHalf8*>(row_output);

    const PackHalf8* weight_vec = reinterpret_cast<const PackHalf8*>(weight);
    __shared__ float sdata[WARP_SIZE];
    __shared__ float rms_inv[1];

    float sum = 0.0f;
    for(size_t i = tid; i < num_vecs; i += threads_per_block){
        PackHalf8 val = row_input_vec[i];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 value_f = __half22float2(val.data[j]);
            sum += value_f.x * value_f.x;
            sum += value_f.y * value_f.y;
        }
    }
    size_t tail = num_vecs * vec_size + tid;
    if (tail < hidden_dim) {
        float value = __half2float(row_input[tail]);
        sum += value * value;
    }

    // warp reduce
#pragma unroll
    for (int i = 16 ; i > 0; i /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, i);
    }

    if (lane_id == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0) {
        sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? sdata[lane_id] : 0.0f;
#pragma unroll
        for (int i = 16 ; i > 0; i /= 2){
            sum += __shfl_down_sync(0xffffffff, sum, i);
        }
        if (lane_id == 0) {
            rms_inv[0] = rsqrtf(sum / static_cast<float>(hidden_dim) + eps);
        }
    }
    __syncthreads();

    for (size_t i = tid; i < num_vecs; i += threads_per_block) {
        PackHalf8 val = row_input_vec[i];
        PackHalf8 w = weight_vec[i];
        PackHalf8 out;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 value_f = __half22float2(val.data[j]);
            float2 scale_f = __half22float2(w.data[j]);

            out.data[j] =
                __floats2half2_rn(
                    value_f.x * rms_inv[0] * scale_f.x,
                    value_f.y * rms_inv[0] * scale_f.y
                );
        }
        row_output_vec[i] = out;
    }
    if (tail < hidden_dim) {
        float val = __half2float(row_input[tail]);
        float w = __half2float(weight[tail]);
        row_output[tail] = __float2half(val * rms_inv[0] * w);
    }
}

// Scalar fallback kernels — used when hidden_dim is not divisible by vec_size

template<size_t threads_per_block>
__global__ void rms_norm_fp32_scalar_kernel(
    const float* input, const float* weight,
    float* output, size_t rows, size_t hidden_dim, float eps
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

    float sum = 0.0f;
    for (size_t i = tid; i < hidden_dim; i += threads_per_block) {
        float val = row_input[i];
        sum += val * val;
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

    for (size_t i = tid; i < hidden_dim; i += threads_per_block) {
        row_output[i] = row_input[i] * rms_inv[0] * weight[i];
    }
}

template<size_t threads_per_block>
__global__ void rms_norm_fp16_scalar_kernel(
    const half* input, const half* weight,
    half* output, size_t rows, size_t hidden_dim, float eps
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

    float sum = 0.0f;
    for (size_t i = tid; i < hidden_dim; i += threads_per_block) {
        float val = __half2float(row_input[i]);
        sum += val * val;
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

    for (size_t i = tid; i < hidden_dim; i += threads_per_block) {
        float val = __half2float(row_input[i]);
        float w = __half2float(weight[i]);
        row_output[i] = __float2half(val * rms_inv[0] * w);
    }
}
