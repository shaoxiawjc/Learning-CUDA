#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

make bench

ROWS=2048
DIMS=1024,2048,4096,8192

for IMPL in 2; do
    ./bench --impl "$IMPL" --rows "$ROWS" --dims "$DIMS"
done

# ./bench --impl tc --rows "$ROWS" --dims "$DIMS"
