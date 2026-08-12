#pragma once

#include <musa_fp16.h>
#include <musa_runtime.h>
#include <type_traits>

static const int FLASH_WARP_SIZE = 32;
static const unsigned FLASH_FULL_MASK = 0xffffffffu;

template <typename T>
struct FlashConvert;

template <typename T>
struct FlashExp;

template <>
struct FlashExp<float> {
  __device__ static __forceinline__ float eval(float value) {
    return __expf(value);
  }
};

template <>
struct FlashExp<half> {
  __device__ static __forceinline__ float eval(float value) {
    return __expf(value);
  }
};

template <>
struct FlashConvert<float> {
  __device__ static __forceinline__ float load(const float* pointer) {
    return *pointer;
  }
  __device__ static __forceinline__ void store(float* pointer, float value) {
    *pointer = value;
  }
};

template <>
struct FlashConvert<half> {
  __device__ static __forceinline__ float load(const half* pointer) {
    return __half2float(*pointer);
  }
  __device__ static __forceinline__ void store(half* pointer, float value) {
    *pointer = __float2half_rn(value);
  }
};

__device__ __forceinline__ float flash_warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(FLASH_FULL_MASK, value, offset);
  return __shfl_sync(FLASH_FULL_MASK, value, 0);
}



// One warp computes one query row. Scores are consumed online, so the kernel
// never materializes the [target_seq_len, src_seq_len] attention matrix.
template <typename T, int HEAD_DIM, bool IS_CAUSAL, int WARPS_PER_BLOCK = 1>
__global__ void flash_attention_warp_kernel(
    const T* __restrict__ query, const T* __restrict__ key,
    const T* __restrict__ value, T* __restrict__ output,
    int target_seq_len, int src_seq_len, int query_heads, int kv_heads,
    float scale) {
  const int lane = threadIdx.x & (FLASH_WARP_SIZE - 1);
  const int warp = threadIdx.x / FLASH_WARP_SIZE;
  const int work = blockIdx.x * WARPS_PER_BLOCK + warp;
  const int query_head = work % query_heads;
  const int token_and_batch = work / query_heads;
  const int query_token = token_and_batch % target_seq_len;
  const int batch = token_and_batch / target_seq_len;
  const int queries_per_kv = query_heads / kv_heads;
  const int kv_head = query_head / queries_per_kv;

  const size_t q_offset =
      (static_cast<size_t>(batch) * target_seq_len + query_token) *
          query_heads * HEAD_DIM +
      query_head * HEAD_DIM;
  const size_t kv_batch_offset =
      static_cast<size_t>(batch) * src_seq_len * kv_heads * HEAD_DIM;
  const T* q = query + q_offset;
  T* o = output + q_offset;

  float out0 = 0.0f;
  float out1 = 0.0f;
  float maximum = -INFINITY;
  float denominator = 0.0f;
  const int key_limit =
      IS_CAUSAL ? min(src_seq_len, query_token + 1) : src_seq_len;

  for (int key_token = 0; key_token < key_limit; ++key_token) {
    const size_t kv_offset =
        kv_batch_offset +
        (static_cast<size_t>(key_token) * kv_heads + kv_head) * HEAD_DIM;
    const T* k = key + kv_offset;
    const T* v = value + kv_offset;

    float dot = 0.0f;
    if (lane < HEAD_DIM)
      dot += FlashConvert<T>::load(q + lane) *
             FlashConvert<T>::load(k + lane);
    if (lane + FLASH_WARP_SIZE < HEAD_DIM)
      dot += FlashConvert<T>::load(q + lane + FLASH_WARP_SIZE) *
             FlashConvert<T>::load(k + lane + FLASH_WARP_SIZE);
    dot = flash_warp_sum(dot) * scale;

    const float next_maximum = fmaxf(maximum, dot);
    const float old_scale = FlashExp<T>::eval(maximum - next_maximum);
    const float probability = FlashExp<T>::eval(dot - next_maximum);
    denominator = denominator * old_scale + probability;
    if (lane < HEAD_DIM)
      out0 = out0 * old_scale +
             probability * FlashConvert<T>::load(v + lane);
    if (lane + FLASH_WARP_SIZE < HEAD_DIM)
      out1 = out1 * old_scale +
             probability *
                 FlashConvert<T>::load(v + lane + FLASH_WARP_SIZE);
    maximum = next_maximum;
  }

  const float inverse_denominator = 1.0f / denominator;
  if (lane < HEAD_DIM)
    FlashConvert<T>::store(o + lane, out0 * inverse_denominator);
  if (lane + FLASH_WARP_SIZE < HEAD_DIM)
    FlashConvert<T>::store(
        o + lane + FLASH_WARP_SIZE, out1 * inverse_denominator);
}

template <typename T, bool IS_CAUSAL>
__global__ void flash_attention_generic_warp_kernel(
    const T* __restrict__ query, const T* __restrict__ key,
    const T* __restrict__ value, T* __restrict__ output,
    int target_seq_len, int src_seq_len, int query_heads, int kv_heads,
    int head_dim, float scale) {
  const int lane = threadIdx.x;
  const int work = blockIdx.x;
  const int query_head = work % query_heads;
  const int token_and_batch = work / query_heads;
  const int query_token = token_and_batch % target_seq_len;
  const int batch = token_and_batch / target_seq_len;
  const int kv_head = query_head / (query_heads / kv_heads);
  const size_t q_offset =
      (static_cast<size_t>(batch) * target_seq_len + query_token) *
          query_heads * head_dim +
      query_head * head_dim;
  const size_t kv_batch_offset =
      static_cast<size_t>(batch) * src_seq_len * kv_heads * head_dim;
  const T* q = query + q_offset;
  T* o = output + q_offset;
  float accumulators[2] = {0.0f, 0.0f};
  float maximum = -INFINITY;
  float denominator = 0.0f;
  const int key_limit =
      IS_CAUSAL ? min(src_seq_len, query_token + 1) : src_seq_len;
  for (int key_token = 0; key_token < key_limit; ++key_token) {
    const size_t kv_offset =
        kv_batch_offset +
        (static_cast<size_t>(key_token) * kv_heads + kv_head) * head_dim;
    float dot = 0.0f;
    for (int d = lane; d < head_dim; d += FLASH_WARP_SIZE)
      dot += FlashConvert<T>::load(q + d) *
             FlashConvert<T>::load(key + kv_offset + d);
    dot = flash_warp_sum(dot) * scale;
    const float next_maximum = fmaxf(maximum, dot);
    const float old_scale = FlashExp<T>::eval(maximum - next_maximum);
    const float probability = FlashExp<T>::eval(dot - next_maximum);
    denominator = denominator * old_scale + probability;
    for (int d = lane, item = 0; d < head_dim;
         d += FLASH_WARP_SIZE, ++item)
      accumulators[item] =
          accumulators[item] * old_scale +
          probability * FlashConvert<T>::load(value + kv_offset + d);
    maximum = next_maximum;
  }
  for (int d = lane, item = 0; d < head_dim;
       d += FLASH_WARP_SIZE, ++item)
    FlashConvert<T>::store(o + d, accumulators[item] / denominator);
}

template <typename T>
struct MusaFlashLauncher {
  static void launch(const T* q, const T* k, const T* v, T* o,
                     int batch_size, int target_seq_len, int src_seq_len,
                     int query_heads, int kv_heads, int head_dim,
                     bool is_causal) {
    const dim3 grid(batch_size * target_seq_len * query_heads);
    const float scale = rsqrtf(static_cast<float>(head_dim));
#define FLASH_LAUNCH(D, C)                                                     \
    flash_attention_warp_kernel<T, D, C><<<grid, 32>>>(                        \
        q, k, v, o, target_seq_len, src_seq_len, query_heads, kv_heads, scale)
#define FLASH_CASE(D)                                                          \
    case D:                                                                    \
      if (is_causal) FLASH_LAUNCH(D, true);                                    \
      else FLASH_LAUNCH(D, false);                                             \
      break
    switch (head_dim) {
      FLASH_CASE(1);
      FLASH_CASE(2);
      FLASH_CASE(4);
      FLASH_CASE(8);
      FLASH_CASE(16);
      FLASH_CASE(32);
      FLASH_CASE(64);
      default:
        if (is_causal)
          flash_attention_generic_warp_kernel<T, true><<<grid, 32>>>(
              q, k, v, o, target_seq_len, src_seq_len, query_heads,
              kv_heads, head_dim, scale);
        else
          flash_attention_generic_warp_kernel<T, false><<<grid, 32>>>(
              q, k, v, o, target_seq_len, src_seq_len, query_heads,
              kv_heads, head_dim, scale);
    }
#undef FLASH_CASE
#undef FLASH_LAUNCH
  }
};
template <int HEAD_DIM>
__device__ __forceinline__ int musa_swizzle_fp32_shared_d(
    int logical_d, int row)
{
    if constexpr (HEAD_DIM == 32 || HEAD_DIM == 64) {
        constexpr int NUM_CHUNKS = HEAD_DIM / 4;
        const int logical_chunk = logical_d / 4;
        const int physical_chunk =
            logical_chunk ^ (row & (NUM_CHUNKS - 1));
        return physical_chunk * 4 + logical_d % 4;
    } else {
        return logical_d;
    }
}

template<
    const int Br,
    const int Bc,
    const int Wr,
    const int Wc,
    const int Tr,
    const int Tc,
    const int HEAD_DIM,
    const int NUM_THREADS,
    bool IS_CAUSAL
>
__device__ __forceinline__ void musa_flash_attention_fp32_tiled_body(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const float scale,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, unsigned char* smem_storage)
{
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    constexpr int THREADS_PER_WARP_M = Wr / Tr;
    constexpr int THREADS_PER_WARP_N = Wc / Tc;
    constexpr int OUTPUT_TILES = (HEAD_DIM + Wc - 1) / Wc;
    static_assert(
        THREADS_PER_WARP_M * THREADS_PER_WARP_N == WARP_SIZE,
        "Wr/Wc/Tr/Tc must cover one MUSA warp");

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid & (WARP_SIZE - 1);
    const int lane_m_id = lane_id / THREADS_PER_WARP_N;
    const int lane_n_id = lane_id % THREADS_PER_WARP_N;
    const int batch_id = blockIdx.x / query_heads;
    const int head_q_id = blockIdx.x % query_heads;
    const int head_group_num = query_heads / kv_heads;
    const int head_kv_id = head_q_id / head_group_num;
    const int tile_Br_id = blockIdx.y;
    const int tile_Br_begin = tile_Br_id * Br;

    float (*smem_Q)[HEAD_DIM] =
        reinterpret_cast<float (*)[HEAD_DIM]>(smem_storage);
    float (*smem_K)[HEAD_DIM] =
        reinterpret_cast<float (*)[HEAD_DIM]>(
            smem_storage + Br * HEAD_DIM * sizeof(float));
    float (*smem_V)[HEAD_DIM] =
        reinterpret_cast<float (*)[HEAD_DIM]>(
            smem_storage + (Br + Bc) * HEAD_DIM * sizeof(float));

    if constexpr (HEAD_DIM >= 4) {
        constexpr int VECS_PER_ROW = HEAD_DIM / 4;
        for (int vec_id = tid; vec_id < Br * VECS_PER_ROW;
             vec_id += NUM_THREADS) {
            const int row = vec_id / VECS_PER_ROW;
            const int d = (vec_id - row * VECS_PER_ROW) * 4;
            const int query_id = tile_Br_begin + row;
            const int physical_d =
                musa_swizzle_fp32_shared_d<HEAD_DIM>(
                    d, row % THREADS_PER_WARP_M);
            float4* smem_ptr = reinterpret_cast<float4*>(
                &smem_Q[row][physical_d]);
            if (query_id < target_seq_len) {
                const size_t offset =
                    ((size_t)(batch_id * target_seq_len + query_id) *
                     query_heads + head_q_id) * HEAD_DIM + d;
                *smem_ptr = *reinterpret_cast<const float4*>(Q + offset);
            } else {
                *smem_ptr = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
    } else {
        for (int x = tid; x < Br * HEAD_DIM; x += NUM_THREADS) {
            const int row = x / HEAD_DIM;
            const int d = x - row * HEAD_DIM;
            const int query_id = tile_Br_begin + row;
            if (query_id < target_seq_len) {
                const size_t offset =
                    ((size_t)(batch_id * target_seq_len + query_id) *
                     query_heads + head_q_id) * HEAD_DIM + d;
                smem_Q[row][d] = Q[offset];
            } else {
                smem_Q[row][d] = 0.0f;
            }
        }
    }
    __syncthreads();

    float block_row_max_old[Tr];
    float block_row_sum_old[Tr];
    float reg_Z[Tr][OUTPUT_TILES][Tc];
    #pragma unroll
    for (int i = 0; i < Tr; ++i) {
        block_row_max_old[i] = -INFINITY;
        block_row_sum_old[i] = 0.0f;
        #pragma unroll
        for (int out_tile = 0; out_tile < OUTPUT_TILES; ++out_tile) {
            #pragma unroll
            for (int j = 0; j < Tc; ++j) {
                reg_Z[i][out_tile][j] = 0.0f;
            }
        }
    }

    const int block_query_end = tile_Br_begin + Br;
    const int effective_tk = IS_CAUSAL && block_query_end < src_seq_len
        ? block_query_end : src_seq_len;
    const int num_kv_tiles = (effective_tk + Bc - 1) / Bc;

    #pragma unroll 1
    for (int tile_N_id = 0; tile_N_id < num_kv_tiles; ++tile_N_id) {
        if constexpr (HEAD_DIM >= 4) {
            constexpr int VECS_PER_ROW = HEAD_DIM / 4;
            for (int vec_id = tid; vec_id < Bc * VECS_PER_ROW;
                 vec_id += NUM_THREADS) {
                const int key = vec_id / VECS_PER_ROW;
                const int d = (vec_id - key * VECS_PER_ROW) * 4;
                const int key_id = tile_N_id * Bc + key;
                const int physical_d =
                    musa_swizzle_fp32_shared_d<HEAD_DIM>(d, key / Tc);
                float4* smem_K_ptr = reinterpret_cast<float4*>(
                    &smem_K[key][physical_d]);
                float4* smem_V_ptr = reinterpret_cast<float4*>(
                    &smem_V[key][d]);
                if (key_id < effective_tk) {
                    const size_t offset =
                        ((size_t)(batch_id * src_seq_len + key_id) *
                         kv_heads + head_kv_id) * HEAD_DIM + d;
                    *smem_K_ptr =
                        *reinterpret_cast<const float4*>(K + offset);
                    *smem_V_ptr =
                        *reinterpret_cast<const float4*>(V + offset);
                } else {
                    *smem_K_ptr = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    *smem_V_ptr = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        } else {
            for (int x = tid; x < Bc * HEAD_DIM; x += NUM_THREADS) {
                const int key = x / HEAD_DIM;
                const int d = x - key * HEAD_DIM;
                const int key_id = tile_N_id * Bc + key;
                if (key_id < effective_tk) {
                    const size_t offset =
                        ((size_t)(batch_id * src_seq_len + key_id) *
                         kv_heads + head_kv_id) * HEAD_DIM + d;
                    smem_K[key][d] = K[offset];
                    smem_V[key][d] = V[offset];
                } else {
                    smem_K[key][d] = 0.0f;
                    smem_V[key][d] = 0.0f;
                }
            }
        }
        __syncthreads();

        float reg_S[Tr][Tc];
        #pragma unroll
        for (int i = 0; i < Tr; ++i) {
            #pragma unroll
            for (int j = 0; j < Tc; ++j) {
                reg_S[i][j] = 0.0f;
            }
        }

        if constexpr (HEAD_DIM == 32 || HEAD_DIM == 64) {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d += 4) {
                float4 reg_Q[Tr];
                float4 reg_K[Tc];
                #pragma unroll
                for (int i = 0; i < Tr; ++i) {
                    const int local_m =
                        warp_id * Wr + lane_m_id +
                        i * THREADS_PER_WARP_M;
                    const int physical_d =
                        musa_swizzle_fp32_shared_d<HEAD_DIM>(
                            d, local_m % THREADS_PER_WARP_M);
                    reg_Q[i] = *reinterpret_cast<const float4*>(
                        &smem_Q[local_m][physical_d]);
                }
                #pragma unroll
                for (int j = 0; j < Tc; ++j) {
                    const int local_n = lane_n_id * Tc + j;
                    const int physical_d =
                        musa_swizzle_fp32_shared_d<HEAD_DIM>(d, local_n / Tc);
                    reg_K[j] = *reinterpret_cast<const float4*>(
                        &smem_K[local_n][physical_d]);
                }
                #pragma unroll
                for (int i = 0; i < Tr; ++i) {
                    #pragma unroll
                    for (int j = 0; j < Tc; ++j) {
                        reg_S[i][j] = __fmaf_rn(
                            reg_Q[i].x, reg_K[j].x, reg_S[i][j]);
                        reg_S[i][j] = __fmaf_rn(
                            reg_Q[i].y, reg_K[j].y, reg_S[i][j]);
                        reg_S[i][j] = __fmaf_rn(
                            reg_Q[i].z, reg_K[j].z, reg_S[i][j]);
                        reg_S[i][j] = __fmaf_rn(
                            reg_Q[i].w, reg_K[j].w, reg_S[i][j]);
                    }
                }
            }
        } else {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                float reg_Q[Tr];
                float reg_K[Tc];
                #pragma unroll
                for (int i = 0; i < Tr; ++i) {
                    const int local_m =
                        warp_id * Wr + lane_m_id +
                        i * THREADS_PER_WARP_M;
                    const int physical_d =
                        musa_swizzle_fp32_shared_d<HEAD_DIM>(
                            d, local_m % THREADS_PER_WARP_M);
                    reg_Q[i] = smem_Q[local_m][physical_d];
                }
                #pragma unroll
                for (int j = 0; j < Tc; ++j) {
                    const int local_n = lane_n_id * Tc + j;
                    const int physical_d =
                        musa_swizzle_fp32_shared_d<HEAD_DIM>(d, local_n / Tc);
                    reg_K[j] = smem_K[local_n][physical_d];
                }
                #pragma unroll
                for (int i = 0; i < Tr; ++i) {
                    #pragma unroll
                    for (int j = 0; j < Tc; ++j) {
                        reg_S[i][j] = __fmaf_rn(
                            reg_Q[i], reg_K[j], reg_S[i][j]);
                    }
                }
            }
        }
        __syncthreads();

        float block_row_max_new[Tr];
        float block_row_sum_new[Tr];
        float reg_P[Tr][Tc];
        #pragma unroll
        for (int i = 0; i < Tr; ++i) {
            const int local_m =
                warp_id * Wr + lane_m_id +
                i * THREADS_PER_WARP_M;
            const int query_id = tile_Br_begin + local_m;
            float maximum = -INFINITY;
            #pragma unroll
            for (int j = 0; j < Tc; ++j) {
                const int key_id = tile_N_id * Bc + lane_n_id * Tc + j;
                if (query_id >= target_seq_len || key_id >= src_seq_len ||
                    (IS_CAUSAL && key_id > query_id)) {
                    reg_S[i][j] = -INFINITY;
                }
                maximum = fmaxf(maximum, reg_S[i][j] * scale);
            }
            #pragma unroll
            for (int offset = THREADS_PER_WARP_N / 2;
                 offset > 0; offset >>= 1) {
                maximum = fmaxf(maximum, __shfl_xor_sync(
                    FLASH_FULL_MASK, maximum, offset, THREADS_PER_WARP_N));
            }
            block_row_max_new[i] =
                fmaxf(block_row_max_old[i], maximum);

            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < Tc; ++j) {
                const int key_id = tile_N_id * Bc + lane_n_id * Tc + j;
                float probability = 0.0f;
                if (query_id < target_seq_len && key_id < src_seq_len &&
                    (!IS_CAUSAL || key_id <= query_id)) {
                    probability = __expf(__fmaf_rn(
                        reg_S[i][j], scale, -block_row_max_new[i]));
                }
                reg_P[i][j] = probability;
                sum += probability;
            }
            #pragma unroll
            for (int offset = THREADS_PER_WARP_N / 2;
                 offset > 0; offset >>= 1) {
                sum += __shfl_xor_sync(
                    FLASH_FULL_MASK, sum, offset, THREADS_PER_WARP_N);
            }
            block_row_sum_new[i] = sum;
        }

        float reg_O[Tr][OUTPUT_TILES][Tc];
        #pragma unroll
        for (int i = 0; i < Tr; ++i) {
            #pragma unroll
            for (int out_tile = 0; out_tile < OUTPUT_TILES; ++out_tile) {
                #pragma unroll
                for (int j = 0; j < Tc; ++j) {
                    reg_O[i][out_tile][j] = 0.0f;
                }
            }
        }

        #pragma unroll
        for (int owner_lane_n = 0;
             owner_lane_n < THREADS_PER_WARP_N; ++owner_lane_n) {
            const int owner_lane =
                lane_m_id * THREADS_PER_WARP_N + owner_lane_n;
            #pragma unroll
            for (int owner_reg = 0; owner_reg < Tc; ++owner_reg) {
                const int key = owner_lane_n * Tc + owner_reg;
                #pragma unroll
                for (int i = 0; i < Tr; ++i) {
                    const float probability = __shfl_sync(
                        FLASH_FULL_MASK, reg_P[i][owner_reg], owner_lane,
                        WARP_SIZE);
                    #pragma unroll
                    for (int out_tile = 0;
                         out_tile < OUTPUT_TILES; ++out_tile) {
                        const int output_d =
                            out_tile * Wc + lane_n_id * Tc;
                        #pragma unroll
                        for (int j = 0; j < Tc; ++j) {
                            const int d = output_d + j;
                            if (d < HEAD_DIM) {
                                reg_O[i][out_tile][j] = __fmaf_rn(
                                    probability, smem_V[key][d],
                                    reg_O[i][out_tile][j]);
                            }
                        }
                    }
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < Tr; ++i) {
            const float alpha = block_row_sum_old[i] == 0.0f
                ? 0.0f : __expf(
                    block_row_max_old[i] - block_row_max_new[i]);
            #pragma unroll
            for (int out_tile = 0;
                 out_tile < OUTPUT_TILES; ++out_tile) {
                #pragma unroll
                for (int j = 0; j < Tc; ++j) {
                    reg_Z[i][out_tile][j] = __fmaf_rn(
                        alpha, reg_Z[i][out_tile][j],
                        reg_O[i][out_tile][j]);
                }
            }
            block_row_sum_old[i] = __fmaf_rn(
                alpha, block_row_sum_old[i], block_row_sum_new[i]);
            block_row_max_old[i] = block_row_max_new[i];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < Tr; ++i) {
        const int local_m =
            warp_id * Wr + lane_m_id + i * THREADS_PER_WARP_M;
        const int query_id = tile_Br_begin + local_m;
        if (query_id < target_seq_len) {
            const float inv_row_sum = 1.0f / block_row_sum_old[i];
            const size_t output_offset =
                ((size_t)(batch_id * target_seq_len + query_id) *
                 query_heads + head_q_id) * HEAD_DIM;
            #pragma unroll
            for (int out_tile = 0;
                 out_tile < OUTPUT_TILES; ++out_tile) {
                const int output_d =
                    out_tile * Wc + lane_n_id * Tc;
                #pragma unroll
                for (int j = 0; j < Tc; ++j) {
                    const int d = output_d + j;
                    if (d < HEAD_DIM) {
                        O[output_offset + d] =
                            reg_Z[i][out_tile][j] * inv_row_sum;
                    }
                }
            }
        }
    }
}

template<
    const int Br, const int Bc, const int Wr, const int Wc,
    const int Tr, const int Tc, const int HEAD_DIM,
    const int NUM_THREADS, bool IS_CAUSAL>
__global__ void musa_flash_attention_fp32_tiled_kernel(
    const float* Q, const float* K, const float* V, float* O,
    const float scale, int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads) {
    extern __shared__ __align__(16) unsigned char smem_storage[];
    musa_flash_attention_fp32_tiled_body<
        Br, Bc, Wr, Wc, Tr, Tc, HEAD_DIM, NUM_THREADS, IS_CAUSAL>(
            Q, K, V, O, scale, batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads, smem_storage);
}

template<
    const int Br, const int Bc, const int Wr, const int Wc,
    const int Tr, const int Tc, const int HEAD_DIM,
    const int NUM_THREADS, bool IS_CAUSAL>
__global__ void musa_flash_attention_fp32_tiled_static_kernel(
    const float* Q, const float* K, const float* V, float* O,
    const float scale, int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads) {
    __shared__ __align__(16)
        unsigned char smem_storage[(Br + 2 * Bc) * HEAD_DIM * sizeof(float)];
    musa_flash_attention_fp32_tiled_body<
        Br, Bc, Wr, Wc, Tr, Tc, HEAD_DIM, NUM_THREADS, IS_CAUSAL>(
            Q, K, V, O, scale, batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads, smem_storage);
}


template <typename T>
__device__ __forceinline__ T musa_flash_from_float(float x) {
    if constexpr (std::is_same<T, float>::value) {
        return x;
    } else {
        return __float2half_rn(x);
    }
}

template <typename T>
__global__ void musa_tiny_copy_value_kernel(const T* __restrict__ v,
                                       T* __restrict__ o) {
    if (threadIdx.x == 0) {
        o[blockIdx.x] = v[blockIdx.x];
    }
}

// One warp owns one query row. Lanes are laid out as [query_head, dim], so
// case 3 uses all 32 lanes and Q/O are single, fully coalesced transactions.
template <typename T, int TARGET_LEN, int SRC_LEN, int QUERY_HEADS,
          int KV_HEADS, int HEAD_DIM, bool IS_CAUSAL>
__global__ __launch_bounds__(TARGET_LEN * FLASH_WARP_SIZE)
void musa_tiny_flash_attention_kernel(const T* __restrict__ q,
                                 const T* __restrict__ k,
                                 const T* __restrict__ v,
                                 T* __restrict__ o, float scale) {
    static_assert(TARGET_LEN <= 8, "tiny kernel uses one warp per query row");
    static_assert(QUERY_HEADS * HEAD_DIM <= FLASH_WARP_SIZE,
                  "one warp must cover all head dimensions");
    constexpr int KV_ELEMS = SRC_LEN * KV_HEADS * HEAD_DIM;
    __shared__ __align__(16) T shared_k[KV_ELEMS];
    __shared__ __align__(16) T shared_v[KV_ELEMS];

    const int tid = threadIdx.x;
    if constexpr (HEAD_DIM == 4 && std::is_same<T, float>::value) {
        constexpr int VECTORS = KV_ELEMS / 4;
        for (int i = tid; i < VECTORS; i += blockDim.x) {
            reinterpret_cast<float4*>(shared_k)[i] =
                reinterpret_cast<const float4*>(k + blockIdx.x * KV_ELEMS)[i];
            reinterpret_cast<float4*>(shared_v)[i] =
                reinterpret_cast<const float4*>(v + blockIdx.x * KV_ELEMS)[i];
        }
    } else if constexpr (HEAD_DIM == 4 && std::is_same<T, half>::value) {
        constexpr int VECTORS = KV_ELEMS / 8;
        for (int i = tid; i < VECTORS; i += blockDim.x) {
            reinterpret_cast<int4*>(shared_k)[i] =
                reinterpret_cast<const int4*>(k + blockIdx.x * KV_ELEMS)[i];
            reinterpret_cast<int4*>(shared_v)[i] =
                reinterpret_cast<const int4*>(v + blockIdx.x * KV_ELEMS)[i];
        }
    } else if constexpr (HEAD_DIM == 2 && std::is_same<T, float>::value) {
        constexpr int VECTORS = KV_ELEMS / 2;
        for (int i = tid; i < VECTORS; i += blockDim.x) {
            reinterpret_cast<float2*>(shared_k)[i] =
                reinterpret_cast<const float2*>(k + blockIdx.x * KV_ELEMS)[i];
            reinterpret_cast<float2*>(shared_v)[i] =
                reinterpret_cast<const float2*>(v + blockIdx.x * KV_ELEMS)[i];
        }
    } else {
        constexpr int VECTORS = KV_ELEMS / 2;
        for (int i = tid; i < VECTORS; i += blockDim.x) {
            reinterpret_cast<half2*>(shared_k)[i] =
                reinterpret_cast<const half2*>(k + blockIdx.x * KV_ELEMS)[i];
            reinterpret_cast<half2*>(shared_v)[i] =
                reinterpret_cast<const half2*>(v + blockIdx.x * KV_ELEMS)[i];
        }
    }
    __syncthreads();

    const int query_pos = tid >> 5;
    const int lane = tid & 31;
    const int query_head = lane / HEAD_DIM;
    const int dim = lane % HEAD_DIM;
    const bool active = query_head < QUERY_HEADS;
    const int q_base = ((blockIdx.x * TARGET_LEN + query_pos) * QUERY_HEADS +
                        query_head) * HEAD_DIM;
    const float q_value = active ? static_cast<float>(q[q_base + dim]) : 0.0f;
    constexpr int HEADS_PER_KV = QUERY_HEADS / KV_HEADS;
    const int kv_head = query_head / HEADS_PER_KV;
    const int group_lane = query_head * HEAD_DIM;
    const int valid_keys = IS_CAUSAL ? query_pos + 1 : SRC_LEN;

    float scores[SRC_LEN];
    float row_max = -INFINITY;
    #pragma unroll
    for (int key_pos = 0; key_pos < SRC_LEN; ++key_pos) {
        const int kv_base = (key_pos * KV_HEADS + kv_head) * HEAD_DIM;
        float dot = 0.0f;
        if constexpr (HEAD_DIM == 4 && std::is_same<T, float>::value) {
            if (active) {
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dot = fmaf(static_cast<float>(q[q_base + d]),
                               static_cast<float>(shared_k[kv_base + d]), dot);
                }
            }
        } else {
            dot = active
                ? q_value * static_cast<float>(shared_k[kv_base + dim]) : 0.0f;
            #pragma unroll
            for (int offset = HEAD_DIM / 2; offset > 0; offset >>= 1)
                dot += __shfl_down_sync(
                    0xffffffff, dot, offset, HEAD_DIM
                );
            dot = __shfl_sync(0xffffffff, dot, group_lane);
        }
        const float score = dot * scale;
        scores[key_pos] = score;
        if (active && key_pos < valid_keys) row_max = fmaxf(row_max, score);
    }

    float denominator = 0.0f;
    float output = 0.0f;
    #pragma unroll
    for (int key_pos = 0; key_pos < SRC_LEN; ++key_pos) {
        if (active && key_pos < valid_keys) {
            float probability = dim == 0
                ? __expf(scores[key_pos] - row_max) : 0.0f;
            probability = __shfl_sync(0xffffffff, probability, group_lane);
            if (dim == 0) {
                denominator += probability;
            }
            const int kv_base = (key_pos * KV_HEADS + kv_head) * HEAD_DIM;
            output = fmaf(probability,
                          static_cast<float>(shared_v[kv_base + dim]), output);
        }
    }
    const float inv_denominator = __shfl_sync(
        0xffffffff, dim == 0 ? __frcp_rn(denominator) : 0.0f, group_lane);
    if (active) {
        o[q_base + dim] = musa_flash_from_float<T>(output * inv_denominator);
    }
}
template <>
struct MusaFlashLauncher<float> {
  static void launch(const float* q, const float* k, const float* v, float* o,
                     int batch_size, int target_seq_len, int src_seq_len,
                     int query_heads, int kv_heads, int head_dim,
                     bool is_causal) {
    const float scale = rsqrtf(static_cast<float>(head_dim));

    if (head_dim == 1) {
      musa_tiny_copy_value_kernel<float><<<batch_size, 32>>>(v, o);
      return;
    }
    if (head_dim == 2) {
      musa_tiny_flash_attention_kernel<
          float, 3, 3, 3, 1, 2, true><<<batch_size, 96>>>(
          q, k, v, o, scale);
      return;
    }
    if (head_dim == 4) {
      musa_tiny_flash_attention_kernel<
          float, 8, 8, 8, 4, 4, false><<<batch_size, 256>>>(
          q, k, v, o, scale);
      return;
    }

    if (head_dim == 8 && target_seq_len <= 16 && src_seq_len <= 16) {
      const dim3 grid(batch_size * query_heads, 1);
      if (is_causal)
        musa_flash_attention_fp32_tiled_kernel<
            16, 16, 16, 16, 4, 2, 8, 32, true><<<grid, 32, 1536>>>(
            q, k, v, o, scale, batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads);
      else
        musa_flash_attention_fp32_tiled_kernel<
            16, 16, 16, 16, 4, 2, 8, 32, false><<<grid, 32, 1536>>>(
            q, k, v, o, scale, batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads);
      return;
    }

    if (head_dim == 8) {
      const dim3 grid(batch_size * query_heads,
                      (target_seq_len + 31) / 32);
      if (is_causal)
        musa_flash_attention_fp32_tiled_kernel<
            32, 32, 16, 32, 4, 4, 8, 64, true><<<grid, 64, 3072>>>(
            q, k, v, o, scale, batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads);
      else
        musa_flash_attention_fp32_tiled_kernel<
            32, 32, 16, 32, 4, 4, 8, 64, false><<<grid, 64, 3072>>>(
            q, k, v, o, scale, batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads);
      return;
    }

    if (head_dim == 16) {
      if (target_seq_len == 16 && is_causal) {
        const dim3 grid(batch_size * query_heads, 1);
        musa_flash_attention_fp32_tiled_kernel<
            16, 16, 16, 16, 4, 2, 16, 32, true><<<grid, 32, 3072>>>(
            q, k, v, o, scale, batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads);
      } else {
        const dim3 grid(batch_size * query_heads,
                        (target_seq_len + 31) / 32);
        musa_flash_attention_fp32_tiled_kernel<
            32, 16, 16, 16, 4, 2, 16, 64, false><<<grid, 64, 4096>>>(
            q, k, v, o, scale, batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads);
      }
      return;
    }

    if (head_dim == 32) {
      const dim3 grid(batch_size * query_heads,
                      (target_seq_len + 63) / 64);
      if (is_causal)
        musa_flash_attention_fp32_tiled_kernel<
            64, 32, 16, 32, 4, 4, 32, 128, true><<<grid, 128, 16384>>>(
            q, k, v, o, scale, batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads);
      else
        musa_flash_attention_fp32_tiled_kernel<
            64, 32, 16, 32, 4, 4, 32, 128, false><<<grid, 128, 16384>>>(
            q, k, v, o, scale, batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads);
      return;
    }

    const int rows = batch_size * target_seq_len * query_heads;
    if (head_dim == 64) {
      const dim3 grid(batch_size * query_heads,
                      (target_seq_len + 31) / 32);
      if (is_causal)
        musa_flash_attention_fp32_tiled_static_kernel<
            32, 32, 8, 32, 2, 4, 64, 128, true><<<grid, 128>>>(
            q, k, v, o, scale, batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads);
      else
        musa_flash_attention_fp32_tiled_static_kernel<
            32, 32, 8, 32, 2, 4, 64, 128, false><<<grid, 128>>>(
            q, k, v, o, scale, batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads);
      return;
    }

    if (is_causal)
      flash_attention_generic_warp_kernel<float, true><<<rows, 32>>>(
          q, k, v, o, target_seq_len, src_seq_len, query_heads, kv_heads,
          head_dim, scale);
    else
      flash_attention_generic_warp_kernel<float, false><<<rows, 32>>>(
          q, k, v, o, target_seq_len, src_seq_len, query_heads, kv_heads,
          head_dim, scale);
  }
};

template <>
struct MusaFlashLauncher<half> {
  static void launch(const half* q, const half* k, const half* v, half* o,
                     int batch_size, int target_seq_len, int src_seq_len,
                     int query_heads, int kv_heads, int head_dim,
                     bool is_causal) {
    const float scale = rsqrtf(static_cast<float>(head_dim));
    if (head_dim == 1) {
      musa_tiny_copy_value_kernel<half><<<batch_size, 32>>>(v, o);
      return;
    }
    if (head_dim == 2) {
      musa_tiny_flash_attention_kernel<
          half, 3, 3, 3, 1, 2, true><<<batch_size, 96>>>(
          q, k, v, o, scale);
      return;
    }
    if (head_dim == 4) {
      musa_tiny_flash_attention_kernel<
          half, 8, 8, 8, 4, 4, false><<<batch_size, 256>>>(
          q, k, v, o, scale);
      return;
    }

    const int rows = batch_size * target_seq_len * query_heads;
    int warps_per_block = 4;
    if (head_dim == 8) warps_per_block = src_seq_len <= 16 ? 1 : 2;
    else if (head_dim == 16) warps_per_block = src_seq_len <= 8 ? 1 : 2;
    if (rows % warps_per_block != 0) warps_per_block = 1;
#define FLASH_FP16_WARP(D, C)                                                  \
    do {                                                                        \
      if (warps_per_block == 4) {                                               \
        flash_attention_warp_kernel<half, D, C, 4>                             \
            <<<rows / 4, 128>>>(                                               \
                q, k, v, o, target_seq_len, src_seq_len, query_heads,          \
                kv_heads, scale);                                               \
      } else if (warps_per_block == 2) {                                        \
        flash_attention_warp_kernel<half, D, C, 2>                             \
            <<<rows / 2, 64>>>(                                                \
                q, k, v, o, target_seq_len, src_seq_len, query_heads,          \
                kv_heads, scale);                                               \
      } else {                                                                  \
        flash_attention_warp_kernel<half, D, C>                                \
            <<<rows, 32>>>(                                                     \
                q, k, v, o, target_seq_len, src_seq_len, query_heads,          \
                kv_heads, scale);                                               \
      }                                                                         \
    } while (0)
#define FLASH_FP16_CASE(D)                                                      \
    case D:                                                                     \
      if (is_causal) FLASH_FP16_WARP(D, true);                                 \
      else FLASH_FP16_WARP(D, false);                                           \
      break
    switch (head_dim) {
      FLASH_FP16_CASE(8);
      FLASH_FP16_CASE(16);
      FLASH_FP16_CASE(32);
      FLASH_FP16_CASE(64);
      default:
        if (is_causal)
          flash_attention_generic_warp_kernel<half, true><<<rows, 32>>>(
              q, k, v, o, target_seq_len, src_seq_len, query_heads,
              kv_heads, head_dim, scale);
        else
          flash_attention_generic_warp_kernel<half, false><<<rows, 32>>>(
              q, k, v, o, target_seq_len, src_seq_len, query_heads,
              kv_heads, head_dim, scale);
    }
#undef FLASH_FP16_CASE
#undef FLASH_FP16_WARP
  }
};
