#pragma once

#include <musa_fp16.h>
#include <musa_runtime.h>

static const int RMS_WARP_SIZE = 32;
static const unsigned RMS_FULL_MASK = 0xffffffffu;

union alignas(16) MusaHalf8 {
  float4 packed;
  half2 values[4];
};

template <int BLOCK_THREADS>
__device__ __forceinline__ float rms_block_reduce(
    float value, size_t hidden_dim, float eps, float* warp_sums,
    float* inverse_rms) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(RMS_FULL_MASK, value, offset);

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x / RMS_WARP_SIZE;
  if (lane == 0) warp_sums[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < BLOCK_THREADS / RMS_WARP_SIZE ? warp_sums[lane] : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      value += __shfl_down_sync(RMS_FULL_MASK, value, offset);
    if (lane == 0)
      *inverse_rms =
          rsqrtf(value / static_cast<float>(hidden_dim) + eps);
  }
  __syncthreads();
  return *inverse_rms;
}

template <int HIDDEN_DIM>
__global__ void rms_small_float(const float* __restrict__ input,
                                const float* __restrict__ weight,
                                float* __restrict__ output, float eps) {
  const int col = threadIdx.x;
  const size_t offset = static_cast<size_t>(blockIdx.x) * HIDDEN_DIM + col;
  const float x = col < HIDDEN_DIM ? input[offset] : 0.0f;
  float sum = x * x;
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1)
    sum += __shfl_down_sync(RMS_FULL_MASK, sum, delta);
  const float inv = __shfl_sync(
      RMS_FULL_MASK, rsqrtf(sum / static_cast<float>(HIDDEN_DIM) + eps), 0);
  if (col < HIDDEN_DIM) output[offset] = x * inv * weight[col];
}

template <int HIDDEN_DIM>
__global__ void rms_small_half(const half* __restrict__ input,
                               const half* __restrict__ weight,
                               half* __restrict__ output, float eps) {
  const int col = threadIdx.x;
  const size_t offset = static_cast<size_t>(blockIdx.x) * HIDDEN_DIM + col;
  const float x = col < HIDDEN_DIM ? __half2float(input[offset]) : 0.0f;
  float sum = x * x;
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1)
    sum += __shfl_down_sync(RMS_FULL_MASK, sum, delta);
  const float inv = __shfl_sync(
      RMS_FULL_MASK, rsqrtf(sum / static_cast<float>(HIDDEN_DIM) + eps), 0);
  if (col < HIDDEN_DIM)
    output[offset] =
        __float2half_rn(x * inv * __half2float(weight[col]));
}

// One block owns one row. 128-bit transactions reduce load/store instruction
// pressure and input values stay in registers across the reduction.
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void rms_vec_float(const float* __restrict__ input,
                              const float* __restrict__ weight,
                              float* __restrict__ output, size_t rows,
                              size_t hidden_dim, float eps) {
  const size_t row = blockIdx.x;
  if (row >= rows) return;
  const int tid = threadIdx.x;
  const size_t vectors = hidden_dim / 4;
  const float4* row_in =
      reinterpret_cast<const float4*>(input + row * hidden_dim);
  const float4* scales = reinterpret_cast<const float4*>(weight);
  float4* row_out = reinterpret_cast<float4*>(output + row * hidden_dim);
  float4 cache[ITEMS_PER_THREAD];
  float sum = 0.0f;
#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const size_t index =
        tid + static_cast<size_t>(item) * BLOCK_THREADS;
    if (index < vectors) {
      const float4 x = row_in[index];
      cache[item] = x;
      sum += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
    }
  }
  __shared__ float warp_sums[BLOCK_THREADS / RMS_WARP_SIZE];
  __shared__ float inverse_rms;
  const float inv = rms_block_reduce<BLOCK_THREADS>(
      sum, hidden_dim, eps, warp_sums, &inverse_rms);
#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const size_t index =
        tid + static_cast<size_t>(item) * BLOCK_THREADS;
    if (index < vectors) {
      const float4 x = cache[item];
      const float4 w = scales[index];
      float4 y;
      y.x = x.x * inv * w.x;
      y.y = x.y * inv * w.y;
      y.z = x.z * inv * w.z;
      y.w = x.w * inv * w.w;
      row_out[index] = y;
    }
  }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void rms_vec_half(const half* __restrict__ input,
                             const half* __restrict__ weight,
                             half* __restrict__ output, size_t rows,
                             size_t hidden_dim, float eps) {
  const size_t row = blockIdx.x;
  if (row >= rows) return;
  const int tid = threadIdx.x;
  const size_t vectors = hidden_dim / 8;
  const float4* row_in =
      reinterpret_cast<const float4*>(input + row * hidden_dim);
  const float4* scales = reinterpret_cast<const float4*>(weight);
  float4* row_out = reinterpret_cast<float4*>(output + row * hidden_dim);
  float4 cache[ITEMS_PER_THREAD];
  float sum = 0.0f;
#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const size_t index =
        tid + static_cast<size_t>(item) * BLOCK_THREADS;
    if (index < vectors) {
      MusaHalf8 x;
      x.packed = row_in[index];
      cache[item] = x.packed;
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const float2 f = __half22float2(x.values[pair]);
        sum += f.x * f.x + f.y * f.y;
      }
    }
  }
  __shared__ float warp_sums[BLOCK_THREADS / RMS_WARP_SIZE];
  __shared__ float inverse_rms;
  const float inv = rms_block_reduce<BLOCK_THREADS>(
      sum, hidden_dim, eps, warp_sums, &inverse_rms);
#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const size_t index =
        tid + static_cast<size_t>(item) * BLOCK_THREADS;
    if (index < vectors) {
      MusaHalf8 x, w, y;
      x.packed = cache[item];
      w.packed = scales[index];
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const float2 xf = __half22float2(x.values[pair]);
        const float2 wf = __half22float2(w.values[pair]);
        y.values[pair] = __floats2half2_rn(
            xf.x * inv * wf.x, xf.y * inv * wf.y);
      }
      row_out[index] = y.packed;
    }
  }
}

template <int BLOCK_THREADS>
__global__ void rms_scalar_float(const float* __restrict__ input,
                                 const float* __restrict__ weight,
                                 float* __restrict__ output, size_t rows,
                                 size_t hidden_dim, float eps) {
  const size_t row = blockIdx.x;
  if (row >= rows) return;
  const float* row_in = input + row * hidden_dim;
  float* row_out = output + row * hidden_dim;
  float sum = 0.0f;
  for (size_t col = threadIdx.x; col < hidden_dim; col += BLOCK_THREADS) {
    const float x = row_in[col];
    sum += x * x;
  }
  __shared__ float warp_sums[BLOCK_THREADS / RMS_WARP_SIZE];
  __shared__ float inverse_rms;
  const float inv = rms_block_reduce<BLOCK_THREADS>(
      sum, hidden_dim, eps, warp_sums, &inverse_rms);
  for (size_t col = threadIdx.x; col < hidden_dim; col += BLOCK_THREADS)
    row_out[col] = row_in[col] * inv * weight[col];
}

template <int BLOCK_THREADS>
__global__ void rms_scalar_half(const half* __restrict__ input,
                                const half* __restrict__ weight,
                                half* __restrict__ output, size_t rows,
                                size_t hidden_dim, float eps) {
  const size_t row = blockIdx.x;
  if (row >= rows) return;
  const half* row_in = input + row * hidden_dim;
  half* row_out = output + row * hidden_dim;
  float sum = 0.0f;
  for (size_t col = threadIdx.x; col < hidden_dim; col += BLOCK_THREADS) {
    const float x = __half2float(row_in[col]);
    sum += x * x;
  }
  __shared__ float warp_sums[BLOCK_THREADS / RMS_WARP_SIZE];
  __shared__ float inverse_rms;
  const float inv = rms_block_reduce<BLOCK_THREADS>(
      sum, hidden_dim, eps, warp_sums, &inverse_rms);
  for (size_t col = threadIdx.x; col < hidden_dim; col += BLOCK_THREADS)
    row_out[col] = __float2half_rn(
        __half2float(row_in[col]) * inv * __half2float(weight[col]));
}

template <int WARPS_PER_BLOCK, int ITEMS_PER_LANE>
__global__ void rms_warp_rows_float(const float* __restrict__ input,
                                    const float* __restrict__ weight,
                                    float* __restrict__ output, size_t rows,
                                    size_t hidden_dim, float eps) {
  const int warp = threadIdx.x / RMS_WARP_SIZE;
  const int lane = threadIdx.x & (RMS_WARP_SIZE - 1);
  const size_t row =
      static_cast<size_t>(blockIdx.x) * WARPS_PER_BLOCK + warp;
  if (row >= rows) return;
  const size_t vectors = hidden_dim / 4;
  const float4* row_in =
      reinterpret_cast<const float4*>(input + row * hidden_dim);
  const float4* scales = reinterpret_cast<const float4*>(weight);
  float4* row_out = reinterpret_cast<float4*>(output + row * hidden_dim);
  float4 cache[ITEMS_PER_LANE];
  float sum = 0.0f;
#pragma unroll
  for (int item = 0; item < ITEMS_PER_LANE; ++item) {
    const size_t index = lane + static_cast<size_t>(item) * RMS_WARP_SIZE;
    if (index < vectors) {
      const float4 x = row_in[index];
      cache[item] = x;
      sum += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(RMS_FULL_MASK, sum, offset);
  const float inv = __shfl_sync(
      RMS_FULL_MASK, rsqrtf(sum / static_cast<float>(hidden_dim) + eps), 0);
#pragma unroll
  for (int item = 0; item < ITEMS_PER_LANE; ++item) {
    const size_t index = lane + static_cast<size_t>(item) * RMS_WARP_SIZE;
    if (index < vectors) {
      const float4 x = cache[item];
      const float4 w = scales[index];
      float4 y;
      y.x = x.x * inv * w.x;
      y.y = x.y * inv * w.y;
      y.z = x.z * inv * w.z;
      y.w = x.w * inv * w.w;
      row_out[index] = y;
    }
  }
}

template <int WARPS_PER_BLOCK, int ITEMS_PER_LANE>
__global__ void rms_warp_rows_half(const half* __restrict__ input,
                                   const half* __restrict__ weight,
                                   half* __restrict__ output, size_t rows,
                                   size_t hidden_dim, float eps) {
  const int warp = threadIdx.x / RMS_WARP_SIZE;
  const int lane = threadIdx.x & (RMS_WARP_SIZE - 1);
  const size_t row =
      static_cast<size_t>(blockIdx.x) * WARPS_PER_BLOCK + warp;
  if (row >= rows) return;
  const size_t vectors = hidden_dim / 8;
  const float4* row_in =
      reinterpret_cast<const float4*>(input + row * hidden_dim);
  const float4* scales = reinterpret_cast<const float4*>(weight);
  float4* row_out = reinterpret_cast<float4*>(output + row * hidden_dim);
  float4 cache[ITEMS_PER_LANE];
  float sum = 0.0f;
#pragma unroll
  for (int item = 0; item < ITEMS_PER_LANE; ++item) {
    const size_t index = lane + static_cast<size_t>(item) * RMS_WARP_SIZE;
    if (index < vectors) {
      MusaHalf8 x;
      x.packed = row_in[index];
      cache[item] = x.packed;
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const float2 f = __half22float2(x.values[pair]);
        sum += f.x * f.x + f.y * f.y;
      }
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(RMS_FULL_MASK, sum, offset);
  const float inv = __shfl_sync(
      RMS_FULL_MASK, rsqrtf(sum / static_cast<float>(hidden_dim) + eps), 0);
#pragma unroll
  for (int item = 0; item < ITEMS_PER_LANE; ++item) {
    const size_t index = lane + static_cast<size_t>(item) * RMS_WARP_SIZE;
    if (index < vectors) {
      MusaHalf8 x, w, y;
      x.packed = cache[item];
      w.packed = scales[index];
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const float2 xf = __half22float2(x.values[pair]);
        const float2 wf = __half22float2(w.values[pair]);
        y.values[pair] = __floats2half2_rn(
            xf.x * inv * wf.x, xf.y * inv * wf.y);
      }
      row_out[index] = y.packed;
    }
  }
}

// Pack several short rows into one physical MUSA warp. GROUP_WIDTH is the
// number of lanes assigned to a row and must divide the native 32-lane warp.
template <typename T, int HIDDEN_DIM, int WARPS_PER_BLOCK>
__global__ void rms_subgroup_rows(const T* __restrict__ input,
                                  const T* __restrict__ weight,
                                  T* __restrict__ output, size_t rows,
                                  float eps) {
  const int vector_width = sizeof(T) == sizeof(float) ? 4 : 8;
  const int vectors_per_row = HIDDEN_DIM / vector_width;
  const int rows_per_warp = RMS_WARP_SIZE / vectors_per_row;
  const int warp = threadIdx.x / RMS_WARP_SIZE;
  const int lane = threadIdx.x & (RMS_WARP_SIZE - 1);
  const int subgroup = lane / vectors_per_row;
  const int lane_in_row = lane & (vectors_per_row - 1);
  const size_t row =
      static_cast<size_t>(blockIdx.x) * WARPS_PER_BLOCK * rows_per_warp +
      warp * rows_per_warp + subgroup;
  const bool active = row < rows;
  const size_t vector_index =
      row * vectors_per_row + static_cast<size_t>(lane_in_row);

  float4 packed = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float sum = 0.0f;
  if (active) {
    packed = reinterpret_cast<const float4*>(input)[vector_index];
    if (sizeof(T) == sizeof(float)) {
      sum = packed.x * packed.x + packed.y * packed.y +
            packed.z * packed.z + packed.w * packed.w;
    } else {
      MusaHalf8 x;
      x.packed = packed;
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const float2 f = __half22float2(x.values[pair]);
        sum += f.x * f.x + f.y * f.y;
      }
    }
  }
#pragma unroll
  for (int offset = vectors_per_row / 2; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(
        RMS_FULL_MASK, sum, offset, vectors_per_row);
  const float inv = __shfl_sync(
      RMS_FULL_MASK,
      rsqrtf(sum / static_cast<float>(HIDDEN_DIM) + eps),
      0, vectors_per_row);

  if (!active) return;
  const float4 packed_weight =
      reinterpret_cast<const float4*>(weight)[lane_in_row];
  if (sizeof(T) == sizeof(float)) {
    float4 y;
    y.x = packed.x * inv * packed_weight.x;
    y.y = packed.y * inv * packed_weight.y;
    y.z = packed.z * inv * packed_weight.z;
    y.w = packed.w * inv * packed_weight.w;
    reinterpret_cast<float4*>(output)[vector_index] = y;
  } else {
    MusaHalf8 x, w, y;
    x.packed = packed;
    w.packed = packed_weight;
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const float2 xf = __half22float2(x.values[pair]);
      const float2 wf = __half22float2(w.values[pair]);
      y.values[pair] = __floats2half2_rn(
          xf.x * inv * wf.x, xf.y * inv * wf.y);
    }
    reinterpret_cast<float4*>(output)[vector_index] = y.packed;
  }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, int HIDDEN_DIM>
__global__ void rms_static_float(const float* __restrict__ input,
                                 const float* __restrict__ weight,
                                 float* __restrict__ output, size_t rows,
                                 float eps) {
  const size_t row = blockIdx.x;
  if (row >= rows) return;
  const int tid = threadIdx.x;
  const int vectors = HIDDEN_DIM / 4;
  const float4* row_in =
      reinterpret_cast<const float4*>(input + row * HIDDEN_DIM);
  const float4* scales = reinterpret_cast<const float4*>(weight);
  float4* row_out =
      reinterpret_cast<float4*>(output + row * HIDDEN_DIM);
  float4 cache[ITEMS_PER_THREAD];
  float sum = 0.0f;
#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int index = tid + item * BLOCK_THREADS;
    if (index < vectors) {
      const float4 x = row_in[index];
      cache[item] = x;
      sum += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
    }
  }
  __shared__ float warp_sums[BLOCK_THREADS / RMS_WARP_SIZE];
  __shared__ float inverse_rms;
  const float inv = rms_block_reduce<BLOCK_THREADS>(
      sum, HIDDEN_DIM, eps, warp_sums, &inverse_rms);
#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int index = tid + item * BLOCK_THREADS;
    if (index < vectors) {
      const float4 x = cache[item];
      const float4 w = scales[index];
      float4 y;
      y.x = x.x * inv * w.x;
      y.y = x.y * inv * w.y;
      y.z = x.z * inv * w.z;
      y.w = x.w * inv * w.w;
      row_out[index] = y;
    }
  }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, int HIDDEN_DIM>
__global__ void rms_static_half(const half* __restrict__ input,
                                const half* __restrict__ weight,
                                half* __restrict__ output, size_t rows,
                                float eps) {
  const size_t row = blockIdx.x;
  if (row >= rows) return;
  const int tid = threadIdx.x;
  const int vectors = HIDDEN_DIM / 8;
  const float4* row_in =
      reinterpret_cast<const float4*>(input + row * HIDDEN_DIM);
  const float4* scales = reinterpret_cast<const float4*>(weight);
  float4* row_out =
      reinterpret_cast<float4*>(output + row * HIDDEN_DIM);
  float4 cache[ITEMS_PER_THREAD];
  float sum = 0.0f;
#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int index = tid + item * BLOCK_THREADS;
    if (index < vectors) {
      MusaHalf8 x;
      x.packed = row_in[index];
      cache[item] = x.packed;
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const float2 f = __half22float2(x.values[pair]);
        sum += f.x * f.x + f.y * f.y;
      }
    }
  }
  __shared__ float warp_sums[BLOCK_THREADS / RMS_WARP_SIZE];
  __shared__ float inverse_rms;
  const float inv = rms_block_reduce<BLOCK_THREADS>(
      sum, HIDDEN_DIM, eps, warp_sums, &inverse_rms);
#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int index = tid + item * BLOCK_THREADS;
    if (index < vectors) {
      MusaHalf8 x, w, y;
      x.packed = cache[item];
      w.packed = scales[index];
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const float2 xf = __half22float2(x.values[pair]);
        const float2 wf = __half22float2(w.values[pair]);
        y.values[pair] = __floats2half2_rn(
            xf.x * inv * wf.x, xf.y * inv * wf.y);
      }
      row_out[index] = y.packed;
    }
  }
}

// Vectorized body plus scalar prefix/tail for the non-16-byte-aligned 769
// element rows used by case 12.
__global__ void rms_mixed_769_float(const float* __restrict__ input,
                                    const float* __restrict__ weight,
                                    float* __restrict__ output, size_t rows,
                                    float eps) {
  const size_t row = blockIdx.x;
  if (row >= rows) return;
  const int tid = threadIdx.x;
  const int row_offset = row * 769;
  const int prefix = (4 - (row_offset & 3)) & 3;
  const int vectors = (769 - prefix) / 4;
  const int tail = prefix + vectors * 4;
  const float* row_in = input + row_offset;
  float* row_out = output + row_offset;
  const float4* vec_in =
      reinterpret_cast<const float4*>(row_in + prefix);
  float4* vec_out = reinterpret_cast<float4*>(row_out + prefix);

  float4 cached = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float sum = 0.0f;
  if (tid < vectors) {
    cached = vec_in[tid];
    sum = cached.x * cached.x + cached.y * cached.y +
          cached.z * cached.z + cached.w * cached.w;
  }
  if (tid < prefix) {
    const float x = row_in[tid];
    sum += x * x;
  }
  if (tail + tid < 769) {
    const float x = row_in[tail + tid];
    sum += x * x;
  }
  __shared__ float warp_sums[6];
  __shared__ float inverse_rms;
  const float inv =
      rms_block_reduce<192>(sum, 769, eps, warp_sums, &inverse_rms);

  if (tid < vectors) {
    const int col = prefix + tid * 4;
    float4 y;
    y.x = cached.x * inv * weight[col];
    y.y = cached.y * inv * weight[col + 1];
    y.z = cached.z * inv * weight[col + 2];
    y.w = cached.w * inv * weight[col + 3];
    vec_out[tid] = y;
  }
  if (tid < prefix)
    row_out[tid] = row_in[tid] * inv * weight[tid];
  if (tail + tid < 769)
    row_out[tail + tid] =
        row_in[tail + tid] * inv * weight[tail + tid];
}

__global__ void rms_mixed_769_half(const half* __restrict__ input,
                                   const half* __restrict__ weight,
                                   half* __restrict__ output, size_t rows,
                                   float eps) {
  const size_t row = blockIdx.x;
  if (row >= rows) return;
  const int tid = threadIdx.x;
  const int row_offset = row * 769;
  const int prefix = (8 - (row_offset & 7)) & 7;
  const int vectors = (769 - prefix) / 8;
  const int tail = prefix + vectors * 8;
  const half* row_in = input + row_offset;
  half* row_out = output + row_offset;
  const float4* vec_in =
      reinterpret_cast<const float4*>(row_in + prefix);
  float4* vec_out = reinterpret_cast<float4*>(row_out + prefix);

  MusaHalf8 cached;
  cached.packed = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float sum = 0.0f;
  if (tid < vectors) {
    cached.packed = vec_in[tid];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const float2 f = __half22float2(cached.values[pair]);
      sum += f.x * f.x + f.y * f.y;
    }
  }
  if (tid < prefix) {
    const float x = __half2float(row_in[tid]);
    sum += x * x;
  }
  if (tail + tid < 769) {
    const float x = __half2float(row_in[tail + tid]);
    sum += x * x;
  }
  __shared__ float warp_sums[4];
  __shared__ float inverse_rms;
  const float inv =
      rms_block_reduce<128>(sum, 769, eps, warp_sums, &inverse_rms);

  if (tid < vectors) {
    const int col = prefix + tid * 8;
    MusaHalf8 y;
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const float2 xf = __half22float2(cached.values[pair]);
      const float w0 = __half2float(weight[col + pair * 2]);
      const float w1 = __half2float(weight[col + pair * 2 + 1]);
      y.values[pair] =
          __floats2half2_rn(xf.x * inv * w0, xf.y * inv * w1);
    }
    vec_out[tid] = y.packed;
  }
  if (tid < prefix)
    row_out[tid] = __float2half_rn(
        __half2float(row_in[tid]) * inv * __half2float(weight[tid]));
  if (tail + tid < 769)
    row_out[tail + tid] = __float2half_rn(
        __half2float(row_in[tail + tid]) * inv *
        __half2float(weight[tail + tid]));
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, int HIDDEN_DIM>
__global__ void rms_static_half2_output(const half* __restrict__ input,
                                        const half* __restrict__ weight,
                                        half* __restrict__ output,
                                        size_t rows, float eps) {
  const size_t row = blockIdx.x;
  if (row >= rows) return;
  const int tid = threadIdx.x;
  const int vectors = HIDDEN_DIM / 8;
  const float4* row_in =
      reinterpret_cast<const float4*>(input + row * HIDDEN_DIM);
  const float4* scales = reinterpret_cast<const float4*>(weight);
  float4* row_out =
      reinterpret_cast<float4*>(output + row * HIDDEN_DIM);
  float4 cache[ITEMS_PER_THREAD];
  float sum = 0.0f;
#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int index = tid + item * BLOCK_THREADS;
    if (index < vectors) {
      MusaHalf8 x;
      x.packed = row_in[index];
      cache[item] = x.packed;
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const float2 f = __half22float2(x.values[pair]);
        sum += f.x * f.x + f.y * f.y;
      }
    }
  }
  __shared__ float warp_sums[BLOCK_THREADS / RMS_WARP_SIZE];
  __shared__ float inverse_rms;
  const float inv = rms_block_reduce<BLOCK_THREADS>(
      sum, HIDDEN_DIM, eps, warp_sums, &inverse_rms);
  const half2 inv2 = __float2half2_rn(inv);
#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    const int index = tid + item * BLOCK_THREADS;
    if (index < vectors) {
      MusaHalf8 x, w, y;
      x.packed = cache[item];
      w.packed = scales[index];
#pragma unroll
      for (int pair = 0; pair < 4; ++pair)
        y.values[pair] =
            __hmul2(__hmul2(x.values[pair], w.values[pair]), inv2);
      row_out[index] = y.packed;
    }
  }
}
