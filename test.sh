#!/bin/bash
# Flash Attention test wrapper
# Usage: ./test.sh B T S Hq Hkv D causal

set -e

BIN="./test_flash_attn"
SRC="test_flash_attn.cu"
NVCC_FLAGS="-std=c++17 -O0 -arch=sm_80 -DPLATFORM_NVIDIA -I."

# Rebuild if binary is older than source or kernel files
if [[ "$BIN" -ot "$SRC" ]] || [[ "$BIN" -ot "src/kernels.cu" ]] || [[ "$BIN" -ot "src/flash_attention.cu" ]]; then
    echo "=== Rebuilding $BIN ==="
    nvcc $NVCC_FLAGS -o "$BIN" "$SRC" src/cuda_compat.o
fi

exec "$BIN" "$@"

# bash test.sh 2 256 256 32 32 64 0
