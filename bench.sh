# python benchmark.py --runs 50 \
#     --operator rms_norm \
#     --baseline \
#     --baseline-file data/rms_norm_baseline_v2.json

# python benchmark.py \
#     --runs 10 \
#     --operator rms_norm \
#     --plot \
#     --plot-file rms_norm_compare.png \
#     --baseline-file data/rms_norm_baseline_v2.json

# python benchmark.py --runs 2 \
#     --operator attention \
#     --baseline \
#     --baseline-file data/attention_baseline_v4.json

# Official Dao-AILab FlashAttention baseline (fp16, cases 4-14).
# Requires third_party/flash-attention and an installed flash-attn CUDA extension.
# python benchmark_flash_attention.py \
#     --runs 10 \
#     --output data/attention_flash_attn_baseline.json

python benchmark.py \
    --runs 1 \
    --operator attention \
    --plot \
    --plot-file attention_compare.png \
    --baseline-file data/attention_baseline_v4.json