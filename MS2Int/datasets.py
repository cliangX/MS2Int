import h5py
import numpy as np

try:
    from .preprocess import data_read
except ImportError:  # pragma: no cover
    from preprocess import data_read

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """H5 数据集，支持 preload 模式将整个文件一次性加载到内存。"""

    def __init__(self, file_path, include_train: bool = True, preload: bool = False):
        self.file_path = file_path
        self.include_train = include_train
        self.preload = preload
        self._h5_handle = None
        self._mem_data = None

        if preload:
            self._mem_data = self._load_all(file_path, include_train)
            self.length = len(self._mem_data["Sequence"])
        else:
            with h5py.File(file_path, "r") as f:
                self.length = len(f["Sequence"])

    @staticmethod
    def _load_all(file_path: str, include_train: bool) -> dict:
        """将 H5 所有需要的数据集一次性读入内存。"""
        data = {}
        with h5py.File(file_path, "r") as f:
            # 序列字段：优先 annotate
            if "annotate" in f:
                data["annotate"] = f["annotate"][:]
            if "Sequence" in f:
                data["Sequence"] = f["Sequence"][:]
            data["Length"] = f["Length"][:]
            data["Charge"] = f["Charge"][:]
            data["collision_energy"] = f["collision_energy"][:]
            data["Fragmentation"] = f["Fragmentation"][:]
            if include_train and "train_data" in f:
                data["train_data"] = f["train_data"][:]
        return data

    def __len__(self):
        return self.length

    def _get_h5_handle(self):
        if self._h5_handle is None:
            # 每个 worker 在首次取样时各自打开一次 H5，后续复用同一句柄。
            self._h5_handle = h5py.File(self.file_path, "r")
        return self._h5_handle

    def close(self):
        if self._h5_handle is not None:
            self._h5_handle.close()
            self._h5_handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getitem__(self, idx):
        if self.preload:
            return data_read(self._mem_data, idx, include_train=self.include_train)
        return data_read(self._get_h5_handle(), idx, include_train=self.include_train)
