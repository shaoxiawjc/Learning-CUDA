# python benchmark.py --runs 50 \
#     --operator rms_norm \
#     --baseline \
#     --baseline-file data/rms_norm_baseline.json

python benchmark.py \
    --runs 10 \
    --operator rms_norm \
    --plot \
    --plot-file rms_norm_compare.png \
    --baseline-file data/rms_norm_baseline.json

# python benchmark.py --runs 50 \
#     --operator attention \
#     --baseline \
#     --baseline-file data/attention_baseline.json

# python benchmark.py \
#     --runs 10 \
#     --operator attention \
#     --plot \
#     --plot-file attention_compare.png \
#     --baseline-file data/attention_baseline.json