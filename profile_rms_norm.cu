#include <cuda_fp16.h>

#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

#include "src/kernels.cu"

int main(int argc, char** argv) {
  static const std::pair<size_t, size_t> cases[] = {
      {1, 1}, {1, 8}, {2, 16}, {4, 31}, {8, 64}, {16, 128}, {32, 256},
      {64, 512}, {128, 1024}, {32, 2048}, {8, 4096}, {3, 769}, {5, 1536}};

  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " CASE_NUMBER\n";
    return 2;
  }
  const int case_number = std::atoi(argv[1]);
  if (case_number < 1 || case_number > 13) {
    std::cerr << "case must be in [1, 13]\n";
    return 2;
  }

  const auto [rows, hidden_dim] = cases[case_number - 1];
  const size_t elements = rows * hidden_dim;
  std::vector<float> input_f(elements, 1.0f), weight_f(hidden_dim, 1.0f), output_f(elements);
  rmsNorm(input_f, weight_f, output_f, rows, hidden_dim, 1e-5f);

  const half one = __float2half(1.0f);
  std::vector<half> input_h(elements, one), weight_h(hidden_dim, one), output_h(elements);
  rmsNorm(input_h, weight_h, output_h, rows, hidden_dim, 1e-5f);
  return 0;
}
