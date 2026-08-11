#!/usr/bin/env bash
set -e

NCU=$(command -v ncu || true)
NCU=${NCU:-/opt/cuda/nsight_compute/ncu}
NVCC=$(command -v nvcc || true)
NVCC=${NVCC:-/opt/cuda/bin/nvcc}


if [[ $EUID -eq 0 && -n ${SUDO_USER:-} ]]; then
    sudo -u "$SUDO_USER" "$NVCC" \
        -std=c++17 -O0 -arch=sm_80 -DPLATFORM_NVIDIA \
        profile_rms_norm.cu -o profile_rms_norm
    sudo -u "$SUDO_USER" mkdir -p ncu_profile/rms
else
    "$NVCC" -std=c++17 -O0 -arch=sm_80 -DPLATFORM_NVIDIA \
        profile_rms_norm.cu -o profile_rms_norm
    mkdir -p ncu_profile/rms
fi

for case_number in {1..13}; do
    echo "正在分析 RMSNorm case${case_number}（FP32 + FP16）..."
    "$NCU" --set full \
        --kernel-name 'regex:rms_norm' \
        --launch-count 2 \
        --force-overwrite \
        --export "ncu_profile/rms/case${case_number}" \
        ./profile_rms_norm "$case_number"
done

if [[ $EUID -eq 0 && -n ${SUDO_USER:-} ]]; then
    chown -R "$SUDO_USER:$(id -gn "$SUDO_USER")" ncu_profile/rms
fi
