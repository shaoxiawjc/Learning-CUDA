#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <cuda_fp16.h>

#include "./tester/utils.h"
#include "./src/utils.h"
#include "./src/kernels.cu"

// CPU reference: naive attention
static void ref_attention(const std::vector<half>& Q, const std::vector<half>& K,
                          const std::vector<half>& V, std::vector<half>& O,
                          int B, int T, int S, int Hq, int Hkv, int D, bool causal) {
    float scale = 1.0f / sqrtf((float)D);
    int Hg = Hq / Hkv;

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int hq = 0; hq < Hq; hq++) {
                int hkv = hq / Hg;

                float max_val = -INFINITY;
                std::vector<float> scores(S);
                for (int s = 0; s < S; s++) {
                    if (causal && s > t) { scores[s] = -INFINITY; continue; }
                    float dot = 0;
                    for (int d = 0; d < D; d++) {
                        float qv = __half2float(Q[b*T*Hq*D + t*Hq*D + hq*D + d]);
                        float kv = __half2float(K[b*S*Hkv*D + s*Hkv*D + hkv*D + d]);
                        dot += qv * kv;
                    }
                    scores[s] = dot * scale;
                    max_val = fmaxf(max_val, scores[s]);
                }

                float sum = 0;
                for (int s = 0; s < S; s++) {
                    if (causal && s > t) continue;
                    scores[s] = expf(scores[s] - max_val);
                    sum += scores[s];
                }

                for (int d = 0; d < D; d++) {
                    float oval = 0;
                    for (int s = 0; s < S; s++) {
                        if (causal && s > t) continue;
                        float p = scores[s] / sum;
                        float vv = __half2float(V[b*S*Hkv*D + s*Hkv*D + hkv*D + d]);
                        oval += p * vv;
                    }
                    O[b*T*Hq*D + t*Hq*D + hq*D + d] = __float2half_rn(oval);
                }
            }
        }
    }
}

static double now_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static float max_diff(const std::vector<half>& a, const std::vector<half>& b) {
    float max_d = 0;
    for (size_t i = 0; i < a.size(); i++) {
        float d = fabsf(__half2float(a[i]) - __half2float(b[i]));
        if (d > max_d) max_d = d;
    }
    return max_d;
}

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s B T S Hq Hkv D causal\n", prog);
    fprintf(stderr, "  B      batch_size\n");
    fprintf(stderr, "  T      target_seq_len\n");
    fprintf(stderr, "  S      src_seq_len\n");
    fprintf(stderr, "  Hq     query_heads\n");
    fprintf(stderr, "  Hkv    kv_heads (must divide Hq)\n");
    fprintf(stderr, "  D      head_dim (64 or 128)\n");
    fprintf(stderr, "  causal 0 or 1\n");
}

int main(int argc, char** argv) {
    if (argc != 8) {
        print_usage(argv[0]);
        return 1;
    }

    int B  = atoi(argv[1]);
    int T  = atoi(argv[2]);
    int S  = atoi(argv[3]);
    int Hq = atoi(argv[4]);
    int Hkv = atoi(argv[5]);
    int D  = atoi(argv[6]);
    int causal = atoi(argv[7]);

    if (Hq % Hkv != 0) {
        fprintf(stderr, "Error: query_heads must be divisible by kv_heads\n");
        return 1;
    }
    if (D != 64 && D != 128) {
        fprintf(stderr, "Error: head_dim must be 64 or 128\n");
        return 1;
    }

    size_t q_elems  = (size_t)B * T * Hq * D;
    size_t kv_elems = (size_t)B * S * Hkv * D;
    size_t q_bytes  = q_elems * sizeof(half);
    size_t kv_bytes = kv_elems * sizeof(half);

    printf("Config: B=%d T=%d S=%d Hq=%d Hkv=%d D=%d causal=%d\n", B, T, S, Hq, Hkv, D, causal);
    printf("Data:   Q=%zu elems (%.2f MB)  KV=%zu elems (%.2f MB)  O=%zu elems\n",
           q_elems, q_bytes / 1048576.0, kv_elems, kv_bytes / 1048576.0, q_elems);

    std::vector<half> Q(q_elems), K(kv_elems), V(kv_elems);
    std::vector<half> O_kernel(q_elems), O_ref(q_elems);

    srand(42);
    for (size_t i = 0; i < q_elems; i++)
        Q[i] = __float2half_rn((float)rand() / RAND_MAX * 2.0f - 1.0f);
    for (size_t i = 0; i < kv_elems; i++) {
        K[i] = __float2half_rn((float)rand() / RAND_MAX * 2.0f - 1.0f);
        V[i] = __float2half_rn((float)rand() / RAND_MAX * 2.0f - 1.0f);
    }

    // Warmup
    printf("Warming up...\n");
    std::vector<half> O_warm(q_elems);
    flashAttention<half>(Q, K, V, O_warm, B, T, S, Hq, Hkv, D, (bool)causal);

    // Timed kernel run
    printf("Running kernel...\n");
    double t0 = now_ms();
    flashAttention<half>(Q, K, V, O_kernel, B, T, S, Hq, Hkv, D, (bool)causal);
    double t1 = now_ms();
    double kernel_ms = t1 - t0;

    // CPU reference
    printf("Running CPU reference...\n");
    double t2 = now_ms();
    ref_attention(Q, K, V, O_ref, B, T, S, Hq, Hkv, D, (bool)causal);
    double t3 = now_ms();
    double ref_ms = t3 - t2;

    // Compare
    float diff = max_diff(O_kernel, O_ref);
    int n_wrong = 0;
    for (size_t i = 0; i < q_elems; i++) {
        if (fabsf(__half2float(O_kernel[i]) - __half2float(O_ref[i])) > 5e-2f)
            n_wrong++;
    }
    double wrong_pct = 100.0 * n_wrong / q_elems;

    printf("\nResults:\n");
    printf("  Kernel time: %.3f ms\n", kernel_ms);
    printf("  Ref time:    %.3f ms (CPU, for reference only)\n", ref_ms);
    printf("  Max diff:    %.6f\n", diff);
    printf("  Wrong elems: %d / %zu (%.2f%%)\n", n_wrong, q_elems, wrong_pct);

    bool pass = (diff < 1e-1f && wrong_pct < 1.0f);
    printf("  Status:      %s\n", pass ? "PASS" : "FAIL");

    return pass ? 0 : 1;
}
