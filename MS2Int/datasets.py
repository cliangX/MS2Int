import h5py
from preprocess import data_read
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, file_path, include_train: bool = True):
        self.file_path = file_path
        self.include_train = include_train
        self._h5_handle = None
        with h5py.File(file_path, "r") as f:
            self.length = len(f["Sequence"])

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
        return data_read(self._get_h5_handle(), idx, include_train=self.include_train)
