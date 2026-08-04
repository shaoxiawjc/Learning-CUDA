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


// batch_size = 1,target_seq_len=3,src_seq_len=3,query_heads=3,kv_heads=1,head_dim=2,is_causal=1
inline void case2_kernel_fp32_cpu(
    const float* q,
    const float* k,
    const float* v,
    float* o
) {
    constexpr int target_seq_len = 3;
    constexpr int query_heads = 3;
    constexpr int head_dim = 2;
    constexpr float scale = 0.7071067811865475244f;

    auto q_offset = [](int token, int head, int d) {
        return (token * query_heads + head) * head_dim + d;
    };

    auto kv_offset = [](int token, int d) {
        return token * head_dim + d;
    };

    auto o_offset = [](int token, int head, int d) {
        return (token * query_heads + head) * head_dim + d;
    };

    for (int query_head = 0; query_head < query_heads; ++query_head) {
        for (int query_pos = 0;
             query_pos < target_seq_len;
             ++query_pos) {

            float scores[target_seq_len];
            float row_max = -std::numeric_limits<float>::infinity();

            const float q0 = q[q_offset(query_pos, query_head, 0)];
            const float q1 = q[q_offset(query_pos, query_head, 1)];

            for (int key_pos = 0; key_pos <= query_pos; ++key_pos) {
                const float k0 = k[kv_offset(key_pos, 0)];
                const float k1 = k[kv_offset(key_pos, 1)];
                const float score =
                    (q0 * k0 + q1 * k1) * scale;
                scores[key_pos] = score;
                row_max = std::max(row_max, score);
            }
            float denominator = 0.0f;
            float numerator0 = 0.0f;
            float numerator1 = 0.0f;
            for (int key_pos = 0; key_pos <= query_pos; ++key_pos) {
                const float p =
                    std::exp(scores[key_pos] - row_max);

                denominator += p;
                numerator0 += p * v[kv_offset(key_pos, 0)];
                numerator1 += p * v[kv_offset(key_pos, 1)];
            }

            const float inv_denominator = 1.0f / denominator;

            o[o_offset(query_pos, query_head, 0)] =
                numerator0 * inv_denominator;

            o[o_offset(query_pos, query_head, 1)] =
                numerator1 * inv_denominator;
        }
    }
}

inline void case2_kernel_fp16_cpu(
    const half* q,
    const half* k,
    const half* v,
    half* o
) {
    constexpr int target_seq_len = 3;
    constexpr int query_heads = 3;
    constexpr int head_dim = 2;
    constexpr float scale = 0.7071067811865475244f;

    auto q_offset = [](int token, int head, int d) {
        return (token * query_heads + head) * head_dim + d;
    };

    auto kv_offset = [](int token, int d) {
        return token * head_dim + d;
    };

    auto o_offset = [](int token, int head, int d) {
        return (token * query_heads + head) * head_dim + d;
    };

    for (int query_head = 0; query_head < query_heads; ++query_head) {
        for (int query_pos = 0;
             query_pos < target_seq_len;
             ++query_pos) {

            float scores[target_seq_len];
            float row_max =
                -std::numeric_limits<float>::infinity();

            const float q0 = __half2float(
                q[q_offset(query_pos, query_head, 0)]
            );
            const float q1 = __half2float(
                q[q_offset(query_pos, query_head, 1)]
            );

            for (int key_pos = 0;
                 key_pos <= query_pos;
                 ++key_pos) {

                const float k0 = __half2float(
                    k[kv_offset(key_pos, 0)]
                );
                const float k1 = __half2float(
                    k[kv_offset(key_pos, 1)]
                );

                const float score =
                    (q0 * k0 + q1 * k1) * scale;

                scores[key_pos] = score;
                row_max = std::max(row_max, score);
            }

            float denominator = 0.0f;
            float numerator0 = 0.0f;
            float numerator1 = 0.0f;

            for (int key_pos = 0;
                 key_pos <= query_pos;
                 ++key_pos) {

                const float p =
                    std::exp(scores[key_pos] - row_max);

                const float v0 = __half2float(
                    v[kv_offset(key_pos, 0)]
                );
                const float v1 = __half2float(
                    v[kv_offset(key_pos, 1)]
                );

                denominator += p;
                numerator0 += p * v0;
                numerator1 += p * v1;
            }

            const float inv_denominator = 1.0f / denominator;

            o[o_offset(query_pos, query_head, 0)] =
                __float2half_rn(
                    numerator0 * inv_denominator
                );

            o[o_offset(query_pos, query_head, 1)] =
                __float2half_rn(
                    numerator1 * inv_denominator
                );
        }
    }
}


inline void case3_small_attention_fp32_cpu(
    const float* q,
    const float* k,
    const float* v,
    float* o
) {
    constexpr int target_seq_len = 8;
    constexpr int src_seq_len = 8;
    constexpr int query_heads = 8;
    constexpr int kv_heads = 4;
    constexpr int head_dim = 4;
    constexpr int heads_per_kv = query_heads / kv_heads;

    // 1 / sqrt(4)
    constexpr float scale = 0.5f;

    auto q_offset = [](int token, int head, int d) constexpr {
        return (token * query_heads + head) * head_dim + d;
    };

    auto kv_offset = [](int token, int head, int d) constexpr {
        return (token * kv_heads + head) * head_dim + d;
    };

    auto o_offset = [](int token, int head, int d) constexpr {
        return (token * query_heads + head) * head_dim + d;
    };

    for (int query_head = 0; query_head < query_heads; ++query_head) {
        const int kv_head = query_head / heads_per_kv;

        for (int query_pos = 0;
             query_pos < target_seq_len;
             ++query_pos) {

            const int q_base =
                q_offset(query_pos, query_head, 0);

            // A single Q vector is reused for all 8 keys.
            const float q0 = q[q_base + 0];
            const float q1 = q[q_base + 1];
            const float q2 = q[q_base + 2];
            const float q3 = q[q_base + 3];

            float scores[src_seq_len];
            float row_max =
                -std::numeric_limits<float>::infinity();

            // QK^T
            for (int key_pos = 0;
                 key_pos < src_seq_len;
                 ++key_pos) {

                const int k_base =
                    kv_offset(key_pos, kv_head, 0);

                // FP32 dot product.
                const float dot = std::fma(
                    q0, k[k_base + 0],
                    std::fma(
                        q1, k[k_base + 1],
                        std::fma(
                            q2, k[k_base + 2],
                            q3 * k[k_base + 3]
                        )
                    )
                );

                const float score = dot * scale;

                scores[key_pos] = score;
                row_max = std::max(row_max, score);
            }

            // Stable softmax + P @ V.
            float denominator = 0.0f;

            float numerator0 = 0.0f;
            float numerator1 = 0.0f;
            float numerator2 = 0.0f;
            float numerator3 = 0.0f;

            for (int key_pos = 0;
                 key_pos < src_seq_len;
                 ++key_pos) {

                const float p =
                    std::exp(scores[key_pos] - row_max);

                const int v_base =
                    kv_offset(key_pos, kv_head, 0);

                denominator += p;

                numerator0 =
                    std::fma(p, v[v_base + 0], numerator0);
                numerator1 =
                    std::fma(p, v[v_base + 1], numerator1);
                numerator2 =
                    std::fma(p, v[v_base + 2], numerator2);
                numerator3 =
                    std::fma(p, v[v_base + 3], numerator3);
            }

            const float inv_denominator =
                1.0f / denominator;

            const int out_base =
                o_offset(query_pos, query_head, 0);

            o[out_base + 0] =
                numerator0 * inv_denominator;
            o[out_base + 1] =
                numerator1 * inv_denominator;
            o[out_base + 2] =
                numerator2 * inv_denominator;
            o[out_base + 3] =
                numerator3 * inv_denominator;
        }
    }
}

inline void case3_small_attention_fp16_cpu(
    const half* q,
    const half* k,
    const half* v,
    half* o
) {
    constexpr int target_seq_len = 8;
    constexpr int src_seq_len = 8;
    constexpr int query_heads = 8;
    constexpr int kv_heads = 4;
    constexpr int head_dim = 4;
    constexpr int heads_per_kv = query_heads / kv_heads;

    constexpr float scale = 0.5f;

    auto q_offset = [](int token, int head, int d) constexpr {
        return (token * query_heads + head) * head_dim + d;
    };

    auto kv_offset = [](int token, int head, int d) constexpr {
        return (token * kv_heads + head) * head_dim + d;
    };

    auto o_offset = [](int token, int head, int d) constexpr {
        return (token * query_heads + head) * head_dim + d;
    };

    float k_fp32[src_seq_len][kv_heads][head_dim];
    float v_fp32[src_seq_len][kv_heads][head_dim];

    for (int token = 0; token < src_seq_len; ++token) {
        for (int kv_head = 0; kv_head < kv_heads; ++kv_head) {
            const int base =
                kv_offset(token, kv_head, 0);

            k_fp32[token][kv_head][0] =
                __half2float(k[base + 0]);
            k_fp32[token][kv_head][1] =
                __half2float(k[base + 1]);
            k_fp32[token][kv_head][2] =
                __half2float(k[base + 2]);
            k_fp32[token][kv_head][3] =
                __half2float(k[base + 3]);

            v_fp32[token][kv_head][0] =
                __half2float(v[base + 0]);
            v_fp32[token][kv_head][1] =
                __half2float(v[base + 1]);
            v_fp32[token][kv_head][2] =
                __half2float(v[base + 2]);
            v_fp32[token][kv_head][3] =
                __half2float(v[base + 3]);
        }
    }

    for (int query_head = 0; query_head < query_heads; ++query_head) {
        const int kv_head = query_head / heads_per_kv;

        for (int query_pos = 0;
             query_pos < target_seq_len;
             ++query_pos) {

            const int q_base =
                q_offset(query_pos, query_head, 0);

            const float q0 =
                __half2float(q[q_base + 0]);
            const float q1 =
                __half2float(q[q_base + 1]);
            const float q2 =
                __half2float(q[q_base + 2]);
            const float q3 =
                __half2float(q[q_base + 3]);

            float scores[src_seq_len];
            float row_max =
                -std::numeric_limits<float>::infinity();

            for (int key_pos = 0;
                 key_pos < src_seq_len;
                 ++key_pos) {

                const float* key =
                    k_fp32[key_pos][kv_head];

                const float dot = std::fma(
                    q0, key[0],
                    std::fma(
                        q1, key[1],
                        std::fma(
                            q2, key[2],
                            q3 * key[3]
                        )
                    )
                );

                const float score = dot * scale;

                scores[key_pos] = score;
                row_max = std::max(row_max, score);
            }

            float denominator = 0.0f;

            float numerator0 = 0.0f;
            float numerator1 = 0.0f;
            float numerator2 = 0.0f;
            float numerator3 = 0.0f;

            for (int key_pos = 0;
                 key_pos < src_seq_len;
                 ++key_pos) {

                const float p =
                    std::exp(scores[key_pos] - row_max);

                const float* value =
                    v_fp32[key_pos][kv_head];

                denominator += p;

                numerator0 =
                    std::fma(p, value[0], numerator0);
                numerator1 =
                    std::fma(p, value[1], numerator1);
                numerator2 =
                    std::fma(p, value[2], numerator2);
                numerator3 =
                    std::fma(p, value[3], numerator3);
            }

            const float inv_denominator =
                1.0f / denominator;

            const int out_base =
                o_offset(query_pos, query_head, 0);

            o[out_base + 0] = __float2half_rn(
                numerator0 * inv_denominator
            );
            o[out_base + 1] = __float2half_rn(
                numerator1 * inv_denominator
            );
            o[out_base + 2] = __float2half_rn(
                numerator2 * inv_denominator
            );
            o[out_base + 3] = __float2half_rn(
                numerator3 * inv_denominator
            );
        }
    }
}







template <
    int B,
    int SQ,
    int SK,
    int HQ,
    int HKV,
    bool IsCausal
>
inline void attention_hd8_fp32_cpu(
    const float* q,
    const float* k,
    const float* v,
    float* o
) {
    static_assert(B > 0);
    static_assert(SQ > 0);
    static_assert(SK > 0);
    static_assert(HQ > 0);
    static_assert(HKV > 0);
    static_assert(HQ % HKV == 0,
                  "query_heads must be divisible by kv_heads");

    // These fixed causal cases are aligned self-attention.
    static_assert(!IsCausal || SQ == SK,
                  "This causal specialization assumes SQ == SK");

    constexpr int heads_per_kv = HQ / HKV;
    constexpr int kHeadDim = 8;
    constexpr float kScale = 0.3535533905932737622f;

    auto q_offset = [](int batch, int token, int head) constexpr {
        return (
            (batch * SQ + token) * HQ + head
        ) * kHeadDim;
    };

    auto kv_offset = [](int batch, int token, int head) constexpr {
        return (
            (batch * SK + token) * HKV + head
        ) * kHeadDim;
    };

    auto o_offset = [](int batch, int token, int head) constexpr {
        return (
            (batch * SQ + token) * HQ + head
        ) * kHeadDim;
    };

    for (int batch = 0; batch < B; ++batch) {
        for (int query_pos = 0; query_pos < SQ; ++query_pos) {
            for (int query_head = 0;
                 query_head < HQ;
                 ++query_head) {

                const int kv_head =
                    query_head / heads_per_kv;

                const int out_base =
                    o_offset(batch, query_pos, query_head);

                if constexpr (IsCausal) {
                    if (query_pos == 0) {
                        const int v_base =
                            kv_offset(batch, 0, kv_head);

                        for (int d = 0; d < kHeadDim; ++d) {
                            o[out_base + d] = v[v_base + d];
                        }

                        continue;
                    }
                }

                const int q_base =
                    q_offset(batch, query_pos, query_head);

                float q_values[kHeadDim];

                for (int d = 0; d < kHeadDim; ++d) {
                    q_values[d] = q[q_base + d];
                }

                const int valid_key_count =
                    IsCausal ? query_pos + 1 : SK;

                float scores[SK];
                float row_max =
                    -std::numeric_limits<float>::infinity();

                /*
                 * QK^T
                 */
                for (int key_pos = 0;
                     key_pos < valid_key_count;
                     ++key_pos) {

                    const int k_base =
                        kv_offset(batch, key_pos, kv_head);

                    float dot = 0.0f;

                    for (int d = 0; d < kHeadDim; ++d) {
                        dot = std::fma(
                            q_values[d],
                            k[k_base + d],
                            dot
                        );
                    }

                    const float score = dot * kScale;

                    scores[key_pos] = score;
                    row_max = std::max(row_max, score);
                }

                /*
                 * Stable softmax and P @ V.
                 *
                 * Probabilities do not need to be explicitly normalized
                 * before P @ V because numerator and denominator are
                 * normalized together at the end.
                 */
                float denominator = 0.0f;

                float numerators[kHeadDim] = {
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f
                };

                for (int key_pos = 0;
                     key_pos < valid_key_count;
                     ++key_pos) {

                    const float probability =
                        std::exp(scores[key_pos] - row_max);

                    const int v_base =
                        kv_offset(batch, key_pos, kv_head);

                    denominator += probability;

                    for (int d = 0; d < kHeadDim; ++d) {
                        numerators[d] = std::fma(
                            probability,
                            v[v_base + d],
                            numerators[d]
                        );
                    }
                }

                const float inv_denominator =
                    1.0f / denominator;

                for (int d = 0; d < kHeadDim; ++d) {
                    o[out_base + d] =
                        numerators[d] * inv_denominator;
                }
            }
        }
    }
}



template <
    int B,
    int SQ,
    int SK,
    int HQ,
    int HKV,
    bool IsCausal
>
inline void attention_hd8_fp16_cpu(
    const half* q,
    const half* k,
    const half* v,
    half* o
) {
    static_assert(B > 0);
    static_assert(SQ > 0);
    static_assert(SK > 0);
    static_assert(HQ > 0);
    static_assert(HKV > 0);
    static_assert(HQ % HKV == 0,
                  "query_heads must be divisible by kv_heads");

    static_assert(!IsCausal || SQ == SK,
                  "This causal specialization assumes SQ == SK");

    constexpr int heads_per_kv = HQ / HKV;
    constexpr int kHeadDim = 8;
    constexpr float kScale = 0.3535533905932737622f;
    constexpr int kv_elements = B * SK * HKV * kHeadDim;

    auto q_offset = [](int batch, int token, int head) constexpr {
        return (
            (batch * SQ + token) * HQ + head
        ) * kHeadDim;
    };

    auto kv_offset = [](int batch, int token, int head) constexpr {
        return (
            (batch * SK + token) * HKV + head
        ) * kHeadDim;
    };

    auto o_offset = [](int batch, int token, int head) constexpr {
        return (
            (batch * SQ + token) * HQ + head
        ) * kHeadDim;
    };

    /*
     * K/V are reused across query positions and grouped query heads.
     * Convert them once rather than repeatedly converting inside every
     * attention row.
     *
     * The largest requested case has 2048 elements per temporary array:
     * 2048 * sizeof(float) = 8 KiB.
     */
    alignas(64) float k_fp32[kv_elements];
    alignas(64) float v_fp32[kv_elements];

    for (int i = 0; i < kv_elements; ++i) {
        k_fp32[i] = __half2float(k[i]);
        v_fp32[i] = __half2float(v[i]);
    }

    for (int batch = 0; batch < B; ++batch) {
        for (int query_pos = 0; query_pos < SQ; ++query_pos) {
            for (int query_head = 0;
                 query_head < HQ;
                 ++query_head) {

                const int kv_head =
                    query_head / heads_per_kv;

                const int out_base =
                    o_offset(batch, query_pos, query_head);

                /*
                 * Preserve the exact FP16 V value for the first
                 * causal row instead of converting to float and back.
                 */
                if constexpr (IsCausal) {
                    if (query_pos == 0) {
                        const int v_base =
                            kv_offset(batch, 0, kv_head);

                        for (int d = 0; d < kHeadDim; ++d) {
                            o[out_base + d] = v[v_base + d];
                        }

                        continue;
                    }
                }

                const int q_base =
                    q_offset(batch, query_pos, query_head);

                float q_values[kHeadDim];

                for (int d = 0; d < kHeadDim; ++d) {
                    q_values[d] =
                        __half2float(q[q_base + d]);
                }

                const int valid_key_count =
                    IsCausal ? query_pos + 1 : SK;

                float scores[SK];
                float row_max =
                    -std::numeric_limits<float>::infinity();

                /*
                 * QK^T in FP32.
                 */
                for (int key_pos = 0;
                     key_pos < valid_key_count;
                     ++key_pos) {

                    const int k_base =
                        kv_offset(batch, key_pos, kv_head);

                    float dot = 0.0f;

                    for (int d = 0; d < kHeadDim; ++d) {
                        dot = std::fma(
                            q_values[d],
                            k_fp32[k_base + d],
                            dot
                        );
                    }

                    const float score = dot * kScale;

                    scores[key_pos] = score;
                    row_max = std::max(row_max, score);
                }

                /*
                 * Stable softmax and P @ V in FP32.
                 */
                float denominator = 0.0f;

                float numerators[kHeadDim] = {
                    0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f
                };

                for (int key_pos = 0;
                     key_pos < valid_key_count;
                     ++key_pos) {

                    const float probability =
                        std::exp(scores[key_pos] - row_max);

                    const int v_base =
                        kv_offset(batch, key_pos, kv_head);

                    denominator += probability;

                    for (int d = 0; d < kHeadDim; ++d) {
                        numerators[d] = std::fma(
                            probability,
                            v_fp32[v_base + d],
                            numerators[d]
                        );
                    }
                }

                const float inv_denominator =
                    1.0f / denominator;

                for (int d = 0; d < kHeadDim; ++d) {
                    o[out_base + d] = __float2half_rn(
                        numerators[d] * inv_denominator
                    );
                }
            }
        }
    }
}


