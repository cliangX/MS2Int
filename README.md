
<p align="center">
  <img src="assets/logo.png" alt="MS2Int Logo" width="280"/>
</p>

---

<p align="center">
  <strong>MS2Int leverages internal fragment ions to advance peptide tandem mass spectrum prediction</strong>
</p>

![Python](https://img.shields.io/badge/Python-3.10-3776ab?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-ee4c2c?logo=pytorch)
[![Mamba](https://img.shields.io/badge/Mamba-SSM-44a833?logo=github)](https://github.com/state-spaces/mamba)
[![GitHub Stars](https://img.shields.io/github/stars/YOUR_ORG/MS2Int.svg?style=social&label=Stars)](https://github.com/YOUR_ORG/MS2Int)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

<details>
<summary><strong>Table of Contents</strong></summary>

1. [About The Project](#about-the-project)
   * [Applications](#applications)
2. [Getting Started](#getting-started)
   * [Prerequisites](#prerequisites)
   * [Dependencies](#dependencies)
   * [Installation](#installation)
3. [Usage](#usage)
   * [Inference](#1-inference)
   * [VAT Training / Fine-tuning](#2-vat-training--fine-tuning)
   * [FLR Pipeline](#3-flr-pipeline-phosphoproteomics-qc)
   * [Model Weights & Data](#model-weights--data)
4. [Troubleshooting](#troubleshooting)
5. [Citation](#citation)
6. [License](#license)
7. [Contact](#contact)

</details>

---

## About The Project

MS2Int is a deep learning framework that integrates internal fragment ions to enable full-spectrum MS/MS intensity prediction. Built on a bidirectional **Mamba** state space backbone and trained with **Virtual Adversarial Training (VAT)**, MS2Int jointly predicts terminal **b/y** ions and internal **m** ions. Trained on ~7.4 million precursors, MS2Int improves downstream proteomics workflows including DDA rescoring, DIA library search, HLA immunopeptidomics, and phosphosite localization.

<p align="center">
  <img src="assets/model.png" alt="MS2Int model overview" width="900"/>
</p>

<p align="right">(<a href="#ms2int">back to top</a>)</p>

### Applications

* **DDA rescoring**: Prediction-assisted rescoring for data-dependent acquisition workflows.
* **DIA library search**: Predicted spectra for spectral library construction and DIA searching.
* **HLA immunopeptidomics**: Improved identification and rescoring for immunopeptidomics datasets.
* **Phosphosite localization**: Phosphorylation-site localization support with an FLR QC pipeline.

<p align="center">
  <img src="assets/task.png" alt="Downstream applications of MS2Int" width="900"/>
</p>

<p align="right">(<a href="#ms2int">back to top</a>)</p>

---

## Getting Started

To get MS2Int up and running locally, follow these steps.

### Prerequisites

Ensure you have the following before installation:

* Python 3.10+
* GPU with CUDA support (recommended for training and inference acceleration)
* Conda or Miniconda

### Dependencies

* Python 3.10+
* PyTorch 2.7+ (tested with 2.7.0; see `environment.yml` or [Installation](#installation))
* mamba-ssm, mamba-ssm2, h5py, numpy, pandas, tqdm, einops


### Installation

#### One-Click Reproduction Script (From Scratch)

该脚本按“本机已跑通”的路径整理（安装 `causal-conv1d` 以启用 Mamba2 fast path；Blackwell 需要升级/恢复 Triton）：

**Step 1: Create conda environment**
```bash
conda create -n mamba_dev python=3.10 -y
conda activate mamba_dev
```

**Step 2: Install PyTorch 2.7 (cu128)**
```bash
pip install --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

**Step 3: Install core dependencies**
```bash
pip install --no-cache-dir numpy ninja packaging
```

**Step 4: Clone and build mamba_ssm**
```bash
git clone https://github.com/state-spaces/mamba.git mamba_src
cd mamba_src
git -c safe.directory="$(pwd)" fetch --tags
git -c safe.directory="$(pwd)" checkout v2.3.0

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
pip install --no-cache-dir --no-build-isolation .
```

**Step 5：安装 causal-conv1d（启用 Mamba2 fast path）**
```bash
cd ..
git clone https://github.com/Dao-AILab/causal-conv1d.git causal_conv1d_src
cd causal_conv1d_src
git -c safe.directory="$(pwd)" checkout v1.6.0

# 强制从源码编译，避免预编译 wheel 与当前 torch ABI 不匹配
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
pip install --no-cache-dir --no-build-isolation .
```

**Step 6：升级/恢复 Triton（支持 Blackwell）**
```bash
# 注意：安装 causal-conv1d 过程中可能会把 triton 拉回 torch pin 的 3.3.0；Blackwell 需要更高版本
pip install --no-cache-dir --upgrade --force-reinstall triton==3.6.0
```

**Step 7: Acceptance test**
```python
import torch
from mamba_ssm import Mamba2
batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba2(d_model=16, d_state=16, d_conv=4, expand=2, headdim=16, use_mem_eff_path=False).to("cuda")
y = model(x)
assert y.shape == x.shape
print("ACCEPTANCE PASS:", y.shape)
```

#### Alternative: Using docker

If you prefer using `docker`:

coming soon

<p align="right">(<a href="#ms2int">back to top</a>)</p>

---

## Usage

### 1) Inference (from MaxQuant)

从 MaxQuant 的 `msms.txt` + 对应的 `mzML` 生成 MS2Int 推理输入 H5（包含 `train_data`），再用模型写出 `Intpredict`。

**Step 1：从 MaxQuant 生成推理输入 H5（默认输出到 `data/MS2Int_input.h5`）**

```sh
python "spectrum_processing/unmodficaiton/run.py" \
  --msms "data/msms.txt" \
  --mzml-dir "data/mzml" \
  --dataset-name "MS2Int_input" \
  --output-dir "data"
```

**Step 2：运行 MS2Int 推理（把预测强度写入 `Intpredict`）**

```sh
python "MS2Int/predict.py" \
  --ckpt "/mnt/data_nas/lcy/project_MS2predict/5.tools/dia/2.pick.NCE/3.finetune_out/best_epoch_5_val_loss_0.3569_0105_062132.pth" \
  --input "data/MS2Int_input.h5" \
  --output "data/MS2Int_input.h5"
```

**Notes:**

* `run.py` 参数默认值：`--num-workers 32`、`--batch-size 400`（小样本建议 `--num-workers 1`）。
* `run.py` 的 `--final-h5` 默认值是 `data/MS2Int_input.h5`（可显式指定其他路径；如需关闭重命名可传空字符串 `--final-h5 ""`）。
* `Fragmentation`（HCD/CID）优先从 mzML 的 activation short string 提取；`collision_energy` 优先从 mzML 的 precursor meta value `collision energy` 提取。
* `predict.py` 的 `--output` 可以与 `--input` 相同，此时会在同一个 H5 中新增/覆盖 `Intpredict`。
* `predict.py` 默认只对长度 ≤30 的肽段做推理；更长的样本在输出中会用 0 填充。

### 2) VAT Training / Fine-tuning

**Training from scratch (example, adjust parameters according to your data and environment):**

```sh
python MS2Int/main.py --train_data_pth "/path/to/train.h5"
```

**Fine-tuning (initialize with pre-trained weights `--pth`):**

```sh
python MS2Int/fine_tune.py \
  --pth "/path/to/pretrained.pth" \
  --train_data_pth "/path/to/train.h5" \
  --checkpoint_path "checkpoints/ms2int_vat/" \
  --log_path "logs/train.log"
```

**VAT-related hyperparameters (optional):**

| Parameter   | Description                      |
|-------------|----------------------------------|
| `--vat_alpha` | VAT loss weight                  |
| `--vat_eps`   | Perturbation radius              |
| `--vat_xi`    | Initial perturbation scale       |
| `--vat_ip`    | Power iteration iterations       |

### 3) FLR Pipeline (Phosphoproteomics QC)

Execution entry point: `MS2Int_FLR/run_pipeline.sh`, requires model checkpoint path:

```sh
MODEL_CKPT="/path/to/model.pth" \
bash MS2Int_FLR/run_pipeline.sh "/path/to/PROJECT_ROOT"
```

**Expected input structure for `PROJECT_ROOT`:**

```
PROJECT_ROOT/
├── txt/
│   ├── msms.txt
│   └── Phospho (STY)Sites.txt   # Optional: skip Step8 if not present
└── mzml/
    ├── raw1.mzML
    └── raw2.mzML
```

Output will be written to `PROJECT_ROOT/mambaflr/` (including `step7_flr_curve.csv`, `step8_phosphosites.csv`, etc.).

### Model Weights & Data

This repository **does not directly provide large model weights and large datasets** (for GitHub distribution purposes). It is recommended to host weights and data on Zenodo or Hugging Face, and specify paths via `MODEL_CKPT` or command-line arguments.

<p align="right">(<a href="#ms2int">back to top</a>)</p>

---

## Troubleshooting

### Common Issues

**Environment / Dependencies**

* **mamba-ssm / causal-conv1d build issues**: If compilation fails with `SSLEOFError` or network timeouts, repeat the pip install command - pip will auto-retry. 如果你需要 `causal-conv1d`（启用 Mamba2 fast path），建议按 [Installation](#installation) 的 Step 5 从源码编译安装（`CAUSAL_CONV1D_FORCE_BUILD=TRUE` + `--no-build-isolation`），避免预编译 wheel 与当前 torch ABI 不匹配；并注意它可能把 triton 拉回 3.3.0，Blackwell 需再安装 `triton==3.6.0`。也可以直接跳过 `causal-conv1d`（MS2Int 兼容 `use_mem_eff_path=False`）。

* **Git safe.directory error**: If you see `fatal: detected dubious ownership in repository`, use `git -c safe.directory="$(pwd)"` prefix for fetch/checkout commands, or add the directory to git safe directories.

* **Mamba2 headdim assertion error**: If you get `AssertionError: assert self.d_ssm % self.headdim == 0`, ensure `headdim` divides `d_inner` evenly. For `d_model=16, expand=2`, use `headdim=16` (since `d_inner = 32`).

* **Triton not supported on Blackwell**: If you see `computeCapability not supported' failed` with `target=cuda:120`, upgrade Triton: `pip install --upgrade --force-reinstall triton==3.6.0`. This conflicts with PyTorch's triton==3.3.0 dependency but is necessary for Blackwell GPUs.

* **CUDA version mismatch**: When both CUDA 12.8 and 13.0 exist, explicitly set `CUDA_HOME=/usr/local/cuda-12.8` (PyTorch 2.7 wheels use CUDA 12.8) before building mamba_ssm.

* **FLR pipeline Step3**: Requires `pyopenms`, install separately via `conda install -c conda-forge pyopenms`.

**Inference / Training**

* **All-zero intensities**: During inference, samples with intensity all zeros likely exceed the maximum supported peptide length (≤30 amino acids).

* **Out of Memory (OOM)**: Reduce batch size.

**Docker**

* **Building Docker image**: The Dockerfile uses `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` as base. Ensure `CUDA_HOME=/usr/local/cuda-12.8` is set during build if compiling from source.

<p align="right">(<a href="#ms2int">back to top</a>)</p>

---

## Citation

**Coming soon.** The manuscript is currently under review. Citation information will be updated upon publication.

<p align="right">(<a href="#ms2int">back to top</a>)</p>

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

<p align="right">(<a href="#ms2int">back to top</a>)</p>

---

## Contact

Project Link: [https://github.com/YOUR_ORG/MS2Int](https://github.com/YOUR_ORG/MS2Int)

<p align="right">(<a href="#ms2int">back to top</a>)</p>
