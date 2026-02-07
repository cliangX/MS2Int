# mamba_dev 环境：pyopenms 安装与自检

## 结论
- 通过 `pip install pyopenms==3.4.0` 在 `conda(mamba_dev)` 中安装成功。
- 自检：`import pyopenms` 成功，基本对象 `MSExperiment/MzMLFile/PeakFileOptions` 可构造。
- GPU 自检：`from mamba_ssm import Mamba2` 的 CUDA acceptance test 通过。

## 关键版本（来自 mamba_dev）
- `pyopenms==3.4.0`
- `numpy==2.2.6`
- `pandas==2.3.3`
- `tqdm==4.67.3`

## 记录位置
- 仓库根目录 `requirement.txt` 记录了本次新增安装的包与版本。

## 备注
- 之前尝试用 `conda install` 安装（含 `pyopenms/pandas/h5py/pytables`）在依赖求解阶段耗时较长，可能需要 `--solver libmamba` 或增加超时/耐心等待。