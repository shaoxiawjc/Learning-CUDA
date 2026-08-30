# hadamard-bench

自研 CUDA Hadamard 内核（`hadamard.cu`）与 [fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform) 的对比 benchmark，支持 `fp16` / `bf16`。

用 Sylvester 矩阵 `H[m][k] = (-1)^popcount(m & k)`（即 `y = x @ H`）做正确性校验，并对比两者吞吐与延迟。

## 文件结构

| 文件 | 说明 |
| --- | --- |
| `hadamard.cu` | 自研 CUDA kernel（scalar + 向量化，`cols ∈ {32..1024}` 的 2 次幂） |
| `bench.py` | 正确性 + 性能对比脚本（用 `load_inline` 现场编译自研 kernel） |
| `pyproject.toml` | 项目与依赖配置 |
| `scripts/vendor-fht.sh` | 拉取并修补 vendored 依赖（见「安装」） |
| `third-party/fast-hadamard-transform` | vendored 依赖（**被 gitignore，由脚本拉取**） |

## 前置条件

- NVIDIA GPU 与驱动（CUDA-capable）
- Python 3.10+
- [uv](https://docs.astral.sh/uv/)（推荐，也可用 pip）

编译 `fast-hadamard-transform` 的 CUDA 扩展需要 `nvcc`（CUDA 编译器）。它由系统 CUDA toolchain 提供，不在 pip 安装的 CUDA 运行时库（`nvidia-*` / `cuda-toolkit`）里；请确保 `nvcc` 在 `PATH` 中，或设置 `CUDA_HOME` / `CUDA_PATH` 指向 CUDA 安装目录（例如 `/opt/cuda`）。

## 安装

### 第 1 步：拉取 vendored 依赖

`third-party/fast-hadamard-transform` 被 `.gitignore` 忽略，不会随仓库一起克隆。直接运行脚本即可（它会 clone 上游仓库，并写入一个 `pyproject.toml` 声明其构建依赖，见下）：

```bash
bash scripts/vendor-fht.sh
```

等价的手动步骤是：

```bash
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git \
    third-party/fast-hadamard-transform

cat > third-party/fast-hadamard-transform/pyproject.toml <<'EOF'
[build-system]
requires = ["setuptools", "wheel", "torch", "packaging", "ninja"]
build-backend = "setuptools.build_meta"
EOF
```

### 第 2 步：安装依赖

**方式 A（推荐，uv 项目模式，用 `uv.lock` 保证可复现）：**

```bash
uv sync
```

**方式 B（pip 风格）：**

```bash
uv pip install .
```

> fast-hadamard-transform 只提供 `setup.py`（没有 `pyproject.toml`），其
> `setup.py` 在构建期 `import torch` 编译 CUDA 扩展，却不声明 torch 为构建依赖。
> uv 的隔离构建环境因此缺少 torch，构建会直接失败。第 1 步的脚本给它写入一个
> `pyproject.toml`，把 `torch`、`setuptools`、`wheel`、`packaging`、`ninja`
> 声明为构建依赖，这样 `uv sync` 就能用默认的构建隔离正确编译。

## 运行

```bash
# 方式 A
uv run python bench.py

# 方式 B
.venv/bin/python bench.py
```

首次运行会用 `load_inline` 现场编译自研 kernel（缓存于 `~/.cache/torch_extensions`），会慢一些；之后走缓存。

输出分为两部分：

1. **correctness** —— 与 fp32 `x @ H` 参考值对比，并校验「自研 vs fht」的最大误差（fp16 ≤ 1e-2，bf16 ≤ 5e-2）。
2. **timing** —— 两者延迟（ms）、带宽（GB/s）与自研相对 fht 的加速比。

## CUDA / torch 版本说明

`uv.lock` 锁定了 torch 2.13.0（CUDA 13 构建）。若新机器需要固定某个 CUDA 变体（例如 `cu130`），按 `pyproject.toml` 末尾注释的写法添加 PyTorch index 并固定 `torch==2.13.0+cu130` 即可：

```toml
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu130" }
```
