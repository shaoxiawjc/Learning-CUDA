# hadamard-bench

自研 CUDA Hadamard 内核与 [fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform) 的对比 benchmark，支持 `fp16` / `bf16`。

纯 C++ 实现：不依赖 Python / torch / virtualenv，直接用 `nvcc` 编译所有内核（自研内核 + 本地 vendored 的 fht CUDA 源码），用一个二进制完成正确性校验与计时。

正确性用 fp32 的 FWHT（`y = x @ H`，Sylvester 矩阵 `H[m][k] = (-1)^popcount(m&k)`）做参考，并以「自研 vs fht」的最大误差作为通过判据。

## 文件结构

| 文件 | 说明 |
| --- | --- |
| `hadamard.cu` | 自研 CUDA kernel（scalar / 向量化 / multi-warp / auto-select，`cols` 为 2 的幂） |
| `hadamard_tensor_core.cu` | 自研 tensor-core kernel |
| `hadacore.cu` | tensor-core kernel（原 torch 绑定版，已改为纯 CUDA） |
| `bench.cu` | 正确性 + 计时 host 程序（curand 生成输入、CUDA event 计时） |
| `fht/` | vendored 的 fast-hadamard-transform CUDA 源码（已去除 torch/c10 依赖） |
| `utils.h` | `cp.async` 等设备侧辅助函数 |
| `cuda_check.h` | CUDA 错误检查宏 |
| `dtype.h` | 共享的 `DType` 枚举 |
| `Makefile` | 构建脚本 |
| `bench.sh` | 运行 `impl 1/2/3` 与 `tc` 的便捷脚本 |

## 前置条件

- NVIDIA GPU 与驱动（CUDA-capable）
- CUDA toolchain（`nvcc`）与 `libcurand`，`sm_80` 及以上的架构（本仓库目标 `sm_80` / A800）

`nvcc` 需在 `PATH` 中，或通过 `NVCC=/path/to/nvcc` 指定。

## 构建

```bash
make          # 生成 ./bench
make clean
```

可在 `Makefile` 中修改 `ARCH`（默认 `sm_80`）以适配其它架构。

## 运行

```bash
./bench --impl 2 --rows 1024,2048,4096 --dims 1024,2048,4096,8192
```

常用参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--impl` | `2` | 自研内核实现：`1`(scalar) `2`(向量化) `3`(multi-warp) `4`(auto) `tc`(tensor-core) |
| `--rows` | `small` | 逗号分隔的行数（`small`/`middle`/`large`/`all` 或具体整数） |
| `--dims` | `small` | 逗号分隔的列数（同上） |
| `--dtype` | `all` | `fp16` / `bf16` / `all` |
| `--no-check` | 关 | 跳过正确性校验 |
| `--no-bench` | 关 | 跳过计时 |
| `--iters` | `100` | 每次计时的 kernel 启动次数 |
| `--warmup-ms` | `200` | 计时前的 GPU 预热时长 |
| `--seed` | `0` | 随机数种子 |

输出分为两部分：

1. **correctness** —— 与 fp32 FWHT 参考值对比，并校验「自研 vs fht」与「hadacore vs fht」的最大误差。自研非 tc 内核需 `≤ 1e-2`（fp16）/ `5e-2`（bf16）；tensor-core 内核用更宽松的 `8·eps·√cols` 容差。
2. **timing** —— 各内核延迟（ms）、吞吐（GFLOP/s）与相对 fht 的加速比。

一次跑完多个实现：

```bash
bash bench.sh
```

## 说明

- fht 仅支持 `dim ≤ 32768`；对 `dim < 8`（2、4）会零填充到 8 再计算（等价于原 Python 包装的 padding 行为）。
- 正确性校验只对 `dim ≤ 8192` 进行（fp32 参考的规模限制），更大的 `dim` 只计时。
- 自研 `hadamard.cu` 各 impl 的 `cols` 支持范围不同：`1/2/3` 为 `1024..8192`，`4` 与 `tc` 为 `2..32768`。
