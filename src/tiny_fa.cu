#include <cuda_fp16.h>
#include <cfloat>
#include <cmath>

struct alignas(16) Half8 {
    __half2 data[4];
};

static_assert(sizeof(Half8) == 16);

__device__ __forceinline__
float warp_reduce_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__
float warp_reduce_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(
            value,
            __shfl_down_sync(0xffffffff, value, offset)
        );
    }
    return value;
}

__global__ __launch_bounds__(128)
void flash_attention_fp16_head_dim_1_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    float scale,
    int src_seq_len)
{
    constexpr int NUM_THREADS = 128;
    constexpr int NUM_WARPS = NUM_THREADS / 32;
    constexpr int HALF2_PER_PACK = 4;



    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;

    const float q = __half2float(Q[0]);

    // 每个线程读取连续 8 个 K 和 8 个 V。
    const Half8 k8 =
        reinterpret_cast<const Half8*>(K)[tid];
    const Half8 v8 =
        reinterpret_cast<const Half8*>(V)[tid];

    float local_max = -FLT_MAX;
    #pragma unroll
    for (int i = 0 ; i < HALF2_PER_PACK ; ++i) {
        const float2 k = __half22float2(k8.data[i]);
        const float score0 = q * k.x * scale;
        const float score1 = q * k.y * scale;
        local_max = fmaxf(local_max, score0);
        local_max = fmaxf(local_max, score1);
    }
    local_max = warp_reduce_max(local_max);
    __shared__ float warp_sdata[NUM_WARPS];
    __shared__ float block_max;
    if (lane_id == 0) {
        warp_sdata[warp_id] = local_max;
    }
    __syncthreads();
    if (warp_id == 0) {
        float x = lane_id < NUM_WARPS?warp_sdata[lane_id]:-FLT_MAX;
        x = warp_reduce_max(x);
        if (lane_id == 0) block_max = x;
    }
    __syncthreads();

    float up = 0.0f;
    float down = 0.0f;
    float bm = block_max;
    #pragma unroll
    for (int i = 0 ; i < HALF2_PER_PACK ; ++i) {
        const float2 k = __half22float2(k8.data[i]);
        const float2 v = __half22float2(v8.data[i]);
        float s1 = expf(q * k.x * scale - bm);
        float s2 = expf(q * k.y * scale - bm);
        down += (s1 + s2);
        up = fmaf(s1, v.x, up);
        up = fmaf(s2, v.y, up);
    }
    up = warp_reduce_sum(up);
    down = warp_reduce_sum(down);
    __shared__ float warp_sdata_2[NUM_WARPS];
    if (lane_id == 0) {
        warp_sdata[warp_id] = up;
        warp_sdata_2[warp_id] = down;
    }
    __syncthreads();
    if (warp_id == 0) {
        float x = lane_id < NUM_WARPS?warp_sdata[lane_id]:0;
        x = warp_reduce_sum(x);
        float y = lane_id < NUM_WARPS?warp_sdata_2[lane_id]:0;
        y = warp_reduce_sum(y);
        if (lane_id == 0) {
            O[0] = __float2half_rn(x/y);
        }
    }
}


__global__ __launch_bounds__(256)
void flash_attention_fp32_head_dim_1_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float scale,
    int src_seq_len)
{
    constexpr int NUM_THREADS = 256;
    constexpr int NUM_WARPS = NUM_THREADS / 32;
    constexpr int VEC_SIZE = 4;


    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;

    const float q = Q[0];

    const float4 k4 = reinterpret_cast<const float4*>(K)[tid];
    const float4 v4 = reinterpret_cast<const float4*>(V)[tid];

    float local_max = -FLT_MAX;
    const float score0 = q * k4.x * scale;
    const float score1 = q * k4.y * scale;
    const float score2 = q * k4.z * scale;
    const float score3 = q * k4.w * scale;
    local_max = fmaxf(local_max, score0);
    local_max = fmaxf(local_max, score1);
    local_max = fmaxf(local_max, score2);
    local_max = fmaxf(local_max, score3);

    local_max = warp_reduce_max(local_max);
    __shared__ float warp_sdata[NUM_WARPS];
    __shared__ float block_max;
    if (lane_id == 0) {
        warp_sdata[warp_id] = local_max;
    }
    __syncthreads();
    if (warp_id == 0) {
        float x = lane_id < NUM_WARPS?warp_sdata[lane_id]:-FLT_MAX;
        x = warp_reduce_max(x);
        if (lane_id == 0) block_max = x;
    }
    __syncthreads();

    float up = 0.0f;
    float down = 0.0f;
    float bm = block_max;

    float s0 = expf(q * k4.x * scale - bm);
    float s1 = expf(q * k4.y * scale - bm);
    float s2 = expf(q * k4.z * scale - bm);
    float s3 = expf(q * k4.w * scale - bm);
    down += (s0 + s1 + s2 + s3);
    up = fmaf(s0, v4.x, up);
    up = fmaf(s1, v4.y, up);
    up = fmaf(s2, v4.z, up);
    up = fmaf(s3, v4.w, up);
    

    up = warp_reduce_sum(up);
    down = warp_reduce_sum(down);
    __shared__ float warp_sdata_2[NUM_WARPS];
    if (lane_id == 0) {
        warp_sdata[warp_id] = up;
        warp_sdata_2[warp_id] = down;
    }
    __syncthreads();
    if (warp_id == 0) {
        float x = lane_id < NUM_WARPS?warp_sdata[lane_id]:0;
        x = warp_reduce_sum(x);
        float y = lane_id < NUM_WARPS?warp_sdata_2[lane_id]:0;
        y = warp_reduce_sum(y);
        if (lane_id == 0) {
            O[0] = x/y;
        }
    }
}