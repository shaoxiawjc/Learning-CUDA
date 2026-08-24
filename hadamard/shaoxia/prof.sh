#!/usr/bin/env bash
set -euo pipefail

# Which implementation to compile/profile: `bash prof.sh 3` (or IMPL=3 bash prof.sh).
IMPL="${1:-${IMPL:-2}}"

# Put the venv's tools (ninja) on PATH so load_inline can (re)build if needed.
export PATH="$PWD/.venv/bin:$PATH"

# Resolve values load_inline needs *before* sudo strips the environment:
#   - CUDA_HOME: torch resolves it to /opt/cuda but never exports it; load_inline
#     reads the env var directly and raises "CUDA_HOME is not set" if it must rebuild.
#   - TORCH_EXTENSIONS_DIR: pin it to *your* cache so the root process reuses the
#     already-built extension instead of rebuilding into /root/.cache.
export CUDA_HOME="${CUDA_HOME:-$(.venv/bin/python -c 'from torch.utils.cpp_extension import CUDA_HOME as h; print(h)' 2>/dev/null)}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}"

# 1) Build/refresh the extension as you (cache stays owned by you, no sudo). The
#    impl must match what prof.py will run, else it rebuilds (as root) under sudo.
.venv/bin/python -c "import sys; sys.argv=['bench.py','--impl','$IMPL']; import bench"

# 2) Profile under ncu. `sudo env` re-exports the vars sudo would otherwise strip
#    (PATH, CUDA_HOME, TORCH_EXTENSIONS_DIR). Drop `sudo` if ncu works without it
#    (preferred); re-add only on ERR_NVGPUCTRPERM.
sudo env PATH="$PATH" CUDA_HOME="$CUDA_HOME" TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR" \
    ncu --kernel-name regex:hadamard_kernel -c 1 --launch-skip 9 \
        --set full -o report -f \
        .venv/bin/python prof.py --impl "$IMPL" --dtype fp16 --rows 2048 --cols 8192 --iters 10
