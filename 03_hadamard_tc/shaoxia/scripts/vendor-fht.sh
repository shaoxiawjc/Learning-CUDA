#!/usr/bin/env bash
# Vendor the fast-hadamard-transform dependency and patch it with a minimal
# pyproject.toml declaring its real build requirements.
#
# Upstream ships only a legacy setup.py (no pyproject.toml). That setup.py
# imports torch, setuptools, wheel and packaging at build time to compile a
# CUDA extension, but declares none of them as build dependencies, so uv's
# isolated build env lacks torch and the build fails. This script writes the
# missing pyproject.toml so `uv sync` / `uv lock` can build it with default
# build isolation.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$repo_root/third-party/fast-hadamard-transform"

if [ -d "$dest/.git" ]; then
    echo "already vendored: $dest"
else
    echo "cloning fast-hadamard-transform ..."
    git clone https://github.com/Dao-AILab/fast-hadamard-transform.git "$dest"
fi

cat > "$dest/pyproject.toml" <<'EOF'
[build-system]
requires = ["setuptools", "wheel", "torch", "packaging", "ninja"]
build-backend = "setuptools.build_meta"
EOF

echo "wrote $dest/pyproject.toml"
