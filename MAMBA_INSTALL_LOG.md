# Mamba (mamba\_ssm) 源码编译安装复现实验记录

## 目标

在 Linux 上配置一个可复现的 conda 环境：

- Python **3.10**
- PyTorch **2.7.x**（CUDA 12.8 wheels）
- 从源码编译安装 **state-spaces/mamba**（mamba\_ssm），并用 `Mamba2` 代码片段进行验收。

> 说明：本文档记录了实际执行的命令、关键输出、遇到的问题与解决方式，便于你在同类机器上 1:1 复现。

---

## 运行目录

- 工作目录：`/mnt/data_nas/lcy/project_MS2predict/5.tools/MS2Int`
- 本记录文件：`MAMBA_INSTALL_LOG.md`

---

## 预检信息（Preflight）

> 注：在本机运行 shell 命令时，`/bin/bash` 会反复打印一条无害告警：  
> `/bin/bash: ...libtinfo.so.6: no version information available ...`  
> 不影响后续安装与编译。

执行命令：

```bash
date
uname -a
conda --version && which conda
git --version
gcc --version
which nvcc && nvcc --version
```

关键输出（节选）：

```text
Thu Feb  5 04:15:23 PM UTC 2026
Linux user 6.8.0-90-generic #91-Ubuntu SMP PREEMPT_DYNAMIC Tue Nov 18 14:14:30 UTC 2025 x86_64 GNU/Linux
conda 25.11.1
/data/home/cliang/miniconda3/bin/conda
git version 2.43.0
gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
/usr/local/cuda-13.0/bin/nvcc
Cuda compilation tools, release 13.0, V13.0.88
```

CUDA 备注：

- 系统同时存在 `/usr/local/cuda-12.8` 与 `/usr/local/cuda-13.0`。
- PyTorch 2.7 官方 wheels 对应 **CUDA 12.8**（cu128）。因此在编译 CUDA 扩展时，优先让 `CUDA_HOME=/usr/local/cuda-12.8`。

---

## 实验流程（逐步记录）

### Step 1. 创建 conda 环境（Python 3.10）

先查看已有环境（可选，但建议做，避免覆盖同名环境）：

```bash
conda info --envs
```

输出要点：本机已有一个名为 `mamba` 的环境，但**没有** `mamba_dev`。

创建新环境：

```bash
conda create -n mamba_dev python=3.10 -y
```

关键输出（节选）：

```text
environment location: /data/home/cliang/miniconda3/envs/mamba_dev
python-3.10.19-...
To activate this environment, use
    $ conda activate mamba_dev
```

### Step 2. 安装 PyTorch 2.7（cu128）

激活环境并安装 PyTorch 2.7（CUDA 12.8 wheels）：

```bash
conda activate mamba_dev
pip install --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

安装完成后验证版本与 CUDA 可用性：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device0:", torch.cuda.get_device_name(0))
PY
```

关键输出（节选）：

```text
torch: 2.7.0+cu128
torch.version.cuda: 12.8
torch.cuda.is_available: True
device0: NVIDIA RTX PRO 6000 Blackwell Server Edition
```

遇到的小问题：

- 运行验证脚本时出现警告：`No module named 'numpy'`（环境里尚未安装 numpy）。  
  该警告不影响 CUDA 可用性，但建议在后续依赖安装步骤中补上 `numpy`。

### Step 3. 安装编译依赖（ninja / packaging / causal-conv1d 等）

安装常用依赖（ninja + numpy）。其中 numpy 不是 mamba_ssm 的硬依赖，但 PyTorch 在某些功能会尝试导入 numpy（否则会出现 warning）。

```bash
conda activate mamba_dev
pip install --no-cache-dir numpy ninja
```

#### 可选：安装 causal-conv1d（注意：可能需要源码编译）

`Mamba2` 的默认 mem-efficient 路径会尝试使用 `causal-conv1d`。在本机环境中，`pip install causal-conv1d` 多次出现网络 SSL 抖动 / build 卡住的情况，因此我最终选择 **不依赖 causal-conv1d**，改用 `use_mem_eff_path=False` 的兼容路径完成验收（见 Step 5）。

本机遇到的具体“卡住”表现（供排障复现）：

- **尝试 1（默认 build isolation）**：

```bash
pip install --no-cache-dir 'causal-conv1d>=1.4.0'
```

现象：长时间停留在 `Installing build dependencies: started`，超过 10 分钟无进展，因此手动 `kill` 终止。

- **尝试 2（禁用 build isolation）**：

```bash
pip install --no-cache-dir --no-build-isolation 'causal-conv1d>=1.4.0'
```

现象：可进入 `Building wheel for causal-conv1d (pyproject.toml): started`，但随后长时间无输出/无进展，因此终止。

- **网络抖动**：多次看到 `SSLEOFError` 重试警告（pip 会自动 retry）。如果经常出现，建议重复执行或切换更稳定的镜像源。

如果你希望强制启用 mem-efficient 路径，可自行尝试（可能需要更长编译时间）：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
pip install --no-cache-dir --no-build-isolation causal-conv1d==1.6.0
```

若出现类似错误（CUDA 扩展 ABI 不匹配）：

```text
ImportError: ... causal_conv1d_cuda...so: undefined symbol: ...c10_cuda_check_implementation...
```

建议优先：

- 确认 `torch==2.7.*` 与 `causal-conv1d` 的版本兼容性
- 用 `--no-build-isolation` 触发本机针对当前 torch 的重新编译
- 或在无法解决时使用 `use_mem_eff_path=False` 作为功能验收路径

### Step 4. 克隆源码并编译安装 mamba\_ssm

克隆源码（在当前目录生成 `mamba_src/`）：

```bash
cd /mnt/data_nas/lcy/project_MS2predict/5.tools/MS2Int
git clone https://github.com/state-spaces/mamba.git mamba_src
```

切换到 releases 对应的 tag（本次使用 `v2.3.0`，来自 [state-spaces/mamba Releases](https://github.com/state-spaces/mamba/releases)）：

```bash
cd /mnt/data_nas/lcy/project_MS2predict/5.tools/MS2Int/mamba_src
git fetch --tags
git checkout v2.3.0
git rev-parse --short HEAD
```

遇到的问题：`fatal: detected dubious ownership in repository ...`

这是 Git 的安全机制（safe.directory）。按提示去改 **global git config** 虽然能解决，但这里为了可控复现，我采用 **单次命令覆盖**（不改配置）：

```bash
git -c safe.directory="/mnt/data_nas/lcy/project_MS2predict/5.tools/MS2Int/mamba_src" fetch --tags
git -c safe.directory="/mnt/data_nas/lcy/project_MS2predict/5.tools/MS2Int/mamba_src" checkout v2.3.0
git -c safe.directory="/mnt/data_nas/lcy/project_MS2predict/5.tools/MS2Int/mamba_src" rev-parse --short HEAD
```

输出要点（节选）：

```text
HEAD is now at f1493ff ...
f1493ff
```

源码编译安装（建议明确使用 CUDA 12.8 工具链）：

```bash
conda activate mamba_dev
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
cd /mnt/data_nas/lcy/project_MS2predict/5.tools/MS2Int/mamba_src
pip install --no-cache-dir --no-build-isolation .
```

关键输出（节选）：

```text
Successfully built mamba_ssm
Successfully installed ... mamba_ssm-2.3.0 ...
```

备注：

- 安装过程中若出现 `SSLEOFError`，通常是网络抖动导致的 pip 重试；多执行一次往往可恢复。

### Step 5. 验收（Mamba2 最小样例）

#### 5.1 按用户最小样例测试（第一次失败）

你给的最小样例（仅参数不同于官方 README：你用 `d_state=16`）：

```python
from mamba_ssm import Mamba2
batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba2(d_model=16, d_state=16, d_conv=4, expand=2).to("cuda")
```

在 `mamba_ssm==2.3.0` 中默认 `headdim=64`，而 `d_inner = expand*d_model = 32`，导致：

```text
AssertionError: assert self.d_ssm % self.headdim == 0
```

解决方式：让 `headdim` 能整除 `d_inner`（例如 `headdim=16`），或增大 `d_model`。

#### 5.2 Blackwell GPU + Triton 版本不支持（第二次失败）

修正 `headdim` 后，在本机 Blackwell GPU（compute capability 12.0 / cuda:120）上，PyTorch 2.7 自带的 `triton==3.3.0` 会在 JIT 编译阶段报错：

```text
Assertion `false && "computeCapability not supported"' failed.
... target=cuda:120 ...
RuntimeError: PassManager::run failed
```

解决方式：升级 Triton 到支持 Blackwell 的版本。这里直接升级到 `3.6.0`：

```bash
conda activate mamba_dev
pip install --no-cache-dir --upgrade --force-reinstall triton==3.6.0
```

> 注意：这会与 `torch==2.7.0+cu128` 的 `triton==3.3.0` 依赖声明冲突（pip 会提示 incompatible），但在本机 Blackwell 上这是让 Triton kernel 正常工作的必要条件。  
> 背景可参考 PyTorch 对 “PyTorch 2.7 / Triton 3.3 需要升级以支持 Blackwell” 的跟踪 issue：  
> [pytorch/pytorch#146518](https://github.com/pytorch/pytorch/issues/146518)

#### 5.3 最终通过的验收代码（可直接复制）

为避免依赖 `causal-conv1d`（可选包，且本机安装/ABI 兼容性存在不确定性），我使用 `use_mem_eff_path=False` 的兼容路径完成验收：

```bash
conda activate mamba_dev
python - <<'PY'
import torch
from mamba_ssm import Mamba2

batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba2(
    d_model=16,
    d_state=16,
    d_conv=4,
    expand=2,
    headdim=16,
    use_mem_eff_path=False,
).to("cuda")
y = model(x)
assert y.shape == x.shape
print("ACCEPTANCE PASS:", y.shape)
PY
```

---

## 补充：安装 causal-conv1d（用于 `use_mem_eff_path=True`）

### 背景：直接安装的 wheel 可能与当前 torch ABI 不匹配

在本机 `torch==2.7.0+cu128` 环境下，如果拿到的是旧版本 torch 编译的预编译 wheel，可能出现导入报错（典型症状为 undefined symbol）：

```text
ImportError: .../causal_conv1d_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb
```

这种情况通常需要“按当前环境的 torch/CUDA 重新编译安装”（仓库 issues 里常见建议：从源码安装、关闭 build isolation、必要时强制编译）。

### 1) 克隆并切到 release tag

```bash
cd /mnt/data_nas/lcy/project_MS2predict/5.tools/MS2Int
git clone https://github.com/Dao-AILab/causal-conv1d.git causal_conv1d_src
cd causal_conv1d_src
git -c safe.directory="$(pwd)" checkout v1.6.0
```

### 2) 在 `mamba_dev` 中从源码编译安装

```bash
conda activate mamba_dev

# 先卸载旧的（如果存在）
pip uninstall -y causal-conv1d

# 指定 CUDA 12.8（与 torch==2.7.0+cu128 匹配）
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# issues 常见建议：强制走本地编译 + 关闭 build isolation，避免拉到不匹配的构建依赖
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
pip install --no-cache-dir --no-build-isolation .
```

### 3) 重要：Blackwell 需要升级 Triton（覆盖 torch 的 pin）

`pip install causal-conv1d` 过程中可能把 `triton` 拉回 `3.3.0`（torch 默认 pin），但本机 Blackwell（`sm_120`）需要更高版本 Triton 才能正常 JIT：

```bash
pip install --no-cache-dir --upgrade --force-reinstall triton==3.6.0
```

### 4) 验收

#### 4.1 导入验证

```bash
python -c "import causal_conv1d; from causal_conv1d import causal_conv1d_fn; print('causal_conv1d', getattr(causal_conv1d, '__version__', 'unknown')); print('causal_conv1d_fn', causal_conv1d_fn)"
```

实际输出（节选）：

```text
causal_conv1d 1.6.0
causal_conv1d_fn <function causal_conv1d_fn at 0x...>
```

#### 4.2 Mamba2 mem-eff 路径验收（显式 `use_mem_eff_path=True`）

```bash
python - <<'PY'
import torch
from mamba_ssm import Mamba2

batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim, device="cuda")
model = Mamba2(
    d_model=16,
    d_state=16,
    d_conv=4,
    expand=2,
    headdim=16,
    use_mem_eff_path=True,
).to("cuda")
y = model(x)
assert y.shape == x.shape
print("MAMBA2 MEM_EFF PASS:", y.shape)
PY
```

实际输出：

```text
MAMBA2 MEM_EFF PASS: torch.Size([2, 64, 16])
```


实际输出：

```text
ACCEPTANCE PASS: torch.Size([2, 64, 16])
```

---

## 一键复现脚本（从零开始）

下面脚本按“本次最终可跑通”的路径整理（不依赖 causal-conv1d，Triton 升级以支持 Blackwell）：

```bash
cd /mnt/data_nas/lcy/project_MS2predict/5.tools/MS2Int

# 1) conda env
conda create -n mamba_dev python=3.10 -y
conda activate mamba_dev

# 2) torch 2.7 (cu128)
pip install --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# 3) deps
pip install --no-cache-dir numpy ninja packaging

# 4) clone + build mamba_ssm
git clone https://github.com/state-spaces/mamba.git mamba_src
cd mamba_src
git -c safe.directory="$(pwd)" fetch --tags
git -c safe.directory="$(pwd)" checkout v2.3.0

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
pip install --no-cache-dir --no-build-isolation .

# 5) Triton upgrade for Blackwell (override torch pin)
pip install --no-cache-dir --upgrade --force-reinstall triton==3.6.0

# 6) acceptance
python - <<'PY'
import torch
from mamba_ssm import Mamba2
batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba2(d_model=16, d_state=16, d_conv=4, expand=2, headdim=16, use_mem_eff_path=False).to("cuda")
y = model(x)
assert y.shape == x.shape
print("ACCEPTANCE PASS:", y.shape)
PY
```
