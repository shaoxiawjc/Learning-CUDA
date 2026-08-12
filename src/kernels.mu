#include <vector>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include "../tester/utils.h"
#include "./rms_norm.mu"
#include "./flash_attention.mu"

template <typename T>
struct MusaRmsLauncher;

template <>
struct MusaRmsLauncher<float> {
  static void launch(const float* input, const float* weight, float* output,
                     size_t rows, size_t hidden_dim, float eps) {
    const dim3 grid(rows);
    if (hidden_dim == 64)
      rms_subgroup_rows<float, 64, 4><<<dim3((rows + 7) / 8), 128>>>(
          input, weight, output, rows, eps);
    else if (hidden_dim == 1)
      rms_small_float<1><<<grid, 32>>>(input, weight, output, eps);
    else if (hidden_dim == 8)
      rms_small_float<8><<<grid, 32>>>(input, weight, output, eps);
    else if (hidden_dim == 16)
      rms_small_float<16><<<grid, 32>>>(input, weight, output, eps);
    else if (hidden_dim == 31)
      rms_small_float<31><<<grid, 32>>>(input, weight, output, eps);
    else if (hidden_dim <= 128 && hidden_dim % 4 == 0)
      rms_warp_rows_float<4, 1><<<dim3((rows + 3) / 4), 128>>>(
          input, weight, output, rows, hidden_dim, eps);
    else if (hidden_dim == 256)
      rms_static_float<64, 1, 256><<<grid, 64>>>(
          input, weight, output, rows, eps);
    else if (hidden_dim <= 256 && hidden_dim % 4 == 0)
      rms_warp_rows_float<4, 2><<<dim3((rows + 3) / 4), 128>>>(
          input, weight, output, rows, hidden_dim, eps);
    else if (hidden_dim == 512)
      rms_static_float<128, 1, 512><<<grid, 128>>>(
          input, weight, output, rows, eps);
    else if (hidden_dim == 1024)
      rms_static_float<256, 1, 1024><<<grid, 256>>>(
          input, weight, output, rows, eps);
    else if (hidden_dim == 1536)
      rms_static_float<256, 2, 1536><<<grid, 256>>>(
          input, weight, output, rows, eps);
    else if (hidden_dim == 2048)
      rms_static_float<256, 2, 2048><<<grid, 256>>>(
          input, weight, output, rows, eps);
    else if (hidden_dim == 4096)
      rms_static_float<256, 4, 4096><<<grid, 256>>>(
          input, weight, output, rows, eps);
    else if (hidden_dim <= 512 && hidden_dim % 4 == 0)
      rms_vec_float<64, 2><<<grid, 64>>>(
          input, weight, output, rows, hidden_dim, eps);
    else if (hidden_dim <= 1024 && hidden_dim % 4 == 0)
      rms_vec_float<128, 2><<<grid, 128>>>(
          input, weight, output, rows, hidden_dim, eps);
    else if (hidden_dim <= 2048 && hidden_dim % 4 == 0)
      rms_vec_float<256, 2><<<grid, 256>>>(
          input, weight, output, rows, hidden_dim, eps);
    else if (hidden_dim <= 4096 && hidden_dim % 4 == 0)
      rms_vec_float<256, 4><<<grid, 256>>>(
          input, weight, output, rows, hidden_dim, eps);
    else if (hidden_dim == 769)
      rms_mixed_769_float<<<grid, 192>>>(
          input, weight, output, rows, eps);
    else
      rms_scalar_float<256><<<grid, 256>>>(
          input, weight, output, rows, hidden_dim, eps);
  }
};

template <>
struct MusaRmsLauncher<half> {
  static void launch(const half* input, const half* weight, half* output,
                     size_t rows, size_t hidden_dim, float eps) {
    const dim3 grid(rows);
    if (hidden_dim == 64)
      rms_subgroup_rows<half, 64, 4><<<dim3((rows + 15) / 16), 128>>>(
          input, weight, output, rows, eps);
    else if (hidden_dim == 128)
      rms_subgroup_rows<half, 128, 4><<<dim3((rows + 7) / 8), 128>>>(
          input, weight, output, rows, eps);
    else if (hidden_dim == 1)
      rms_small_half<1><<<grid, 32>>>(input, weight, output, eps);
    else if (hidden_dim == 8)
      rms_small_half<8><<<grid, 32>>>(input, weight, output, eps);
    else if (hidden_dim == 16)
      rms_small_half<16><<<grid, 32>>>(input, weight, output, eps);
    else if (hidden_dim == 31)
      rms_small_half<31><<<grid, 32>>>(input, weight, output, eps);
    else if (hidden_dim == 256)
      rms_subgroup_rows<half, 256, 4><<<dim3((rows + 3) / 4), 128>>>(
          input, weight, output, rows, eps);
    else if (hidden_dim <= 256 && hidden_dim % 8 == 0)
      rms_warp_rows_half<4, 1><<<dim3((rows + 3) / 4), 128>>>(
          input, weight, output, rows, hidden_dim, eps);
    else if (hidden_dim == 512)
      rms_static_half<64, 1, 512><<<grid, 64>>>(
          input, weight, output, rows, eps);
    else if (hidden_dim == 1024)
      rms_static_half2_output<128, 1, 1024><<<grid, 128>>>(
          input, weight, output, rows, eps);
    else if (hidden_dim == 1536)
      rms_static_half2_output<192, 1, 1536><<<grid, 192>>>(
          input, weight, output, rows, eps);
    else if (hidden_dim == 2048)
      rms_static_half<256, 1, 2048><<<grid, 256>>>(
          input, weight, output, rows, eps);
    else if (hidden_dim == 4096)
      rms_static_half2_output<256, 2, 4096><<<grid, 256>>>(
          input, weight, output, rows, eps);
    else if (hidden_dim <= 1024 && hidden_dim % 8 == 0)
      rms_vec_half<64, 2><<<grid, 64>>>(
          input, weight, output, rows, hidden_dim, eps);
    else if (hidden_dim <= 2048 && hidden_dim % 8 == 0)
      rms_vec_half<128, 2><<<grid, 128>>>(
          input, weight, output, rows, hidden_dim, eps);
    else if (hidden_dim <= 4096 && hidden_dim % 8 == 0)
      rms_vec_half<256, 2><<<grid, 256>>>(
          input, weight, output, rows, hidden_dim, eps);
    else if (hidden_dim == 769)
      rms_mixed_769_half<<<grid, 128>>>(
          input, weight, output, rows, eps);
    else
      rms_scalar_half<256><<<grid, 256>>>(
          input, weight, output, rows, hidden_dim, eps);
  }
};

/**
 * RMSNorm over the last dimension of a row-major [rows, hidden_dim] tensor.
 */
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
             std::vector<T>& h_output, size_t rows, size_t hidden_dim,
             float eps) {
  const size_t tensor_bytes = rows * hidden_dim * sizeof(T);
  const size_t weight_bytes = hidden_dim * sizeof(T);
  const size_t required_bytes = tensor_bytes * 2 + weight_bytes;

  // Persist allocations across calls so host allocation overhead does not
  // dominate the small cases. Each template instantiation owns its buffer.
  static void* buffer = NULL;
  static size_t capacity = 0;
  if (required_bytes > capacity) {
    if (buffer != NULL) RUNTIME_CHECK(musaFree(buffer));
    RUNTIME_CHECK(musaMalloc(&buffer, required_bytes));
    capacity = required_bytes;
  }
  char* base = static_cast<char*>(buffer);
  T* input = reinterpret_cast<T*>(base);
  T* weight = reinterpret_cast<T*>(base + tensor_bytes);
  T* output = reinterpret_cast<T*>(base + tensor_bytes + weight_bytes);

  RUNTIME_CHECK(musaMemcpy(input, h_input.data(), tensor_bytes,
                           musaMemcpyHostToDevice));
  RUNTIME_CHECK(musaMemcpy(weight, h_weight.data(), weight_bytes,
                           musaMemcpyHostToDevice));
  MusaRmsLauncher<T>::launch(input, weight, output, rows, hidden_dim, eps);
  RUNTIME_CHECK(musaGetLastError());
  RUNTIME_CHECK(musaMemcpy(h_output.data(), output, tensor_bytes,
                           musaMemcpyDeviceToHost));
}

/**
 * FlashAttention with online softmax; Q and O use query-head layout while
 * K/V use kv-head layout for grouped-query attention.
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim,
                    bool is_causal) {
  const size_t q_elements = static_cast<size_t>(batch_size) *
      target_seq_len * query_heads * head_dim;
  const size_t kv_elements = static_cast<size_t>(batch_size) *
      src_seq_len * kv_heads * head_dim;
  const size_t required_elements = 2 * q_elements + 2 * kv_elements;
  static T* buffer = NULL;
  static size_t capacity = 0;
  if (required_elements > capacity) {
    if (buffer != NULL) RUNTIME_CHECK(musaFree(buffer));
    RUNTIME_CHECK(musaMalloc(reinterpret_cast<void**>(&buffer),
                              required_elements * sizeof(T)));
    capacity = required_elements;
  }
  T* q = buffer;
  T* k = q + q_elements;
  T* v = k + kv_elements;
  T* o = v + kv_elements;
  RUNTIME_CHECK(musaMemcpy(q, h_q.data(), q_elements * sizeof(T),
                           musaMemcpyHostToDevice));
  RUNTIME_CHECK(musaMemcpy(k, h_k.data(), kv_elements * sizeof(T),
                           musaMemcpyHostToDevice));
  RUNTIME_CHECK(musaMemcpy(v, h_v.data(), kv_elements * sizeof(T),
                           musaMemcpyHostToDevice));
  MusaFlashLauncher<T>::launch(q, k, v, o, batch_size, target_seq_len,
                               src_seq_len, query_heads, kv_heads, head_dim,
                               is_causal);
  RUNTIME_CHECK(musaGetLastError());
  RUNTIME_CHECK(musaMemcpy(h_o.data(), o, q_elements * sizeof(T),
                           musaMemcpyDeviceToHost));
}

template void rmsNorm<float>(const std::vector<float>&,
                             const std::vector<float>&,
                             std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&,
                            const std::vector<half>&,
                            std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(
    const std::vector<float>&, const std::vector<float>&,
    const std::vector<float>&, std::vector<float>&,
    int, int, int, int, int, int, bool);
template void flashAttention<half>(
    const std::vector<half>&, const std::vector<half>&,
    const std::vector<half>&, std::vector<half>&,
    int, int, int, int, int, int, bool);
