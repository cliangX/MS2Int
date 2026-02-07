import h5py
from preprocess import data_read
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, file_path, include_train: bool = True):
        """
        通用 H5 数据集封装。

        参数：
        - file_path: H5 文件路径
        - include_train: 是否在 __getitem__ 中一并读取 train_data
          * 训练 / 评估 loss 时需要 True（返回 6 元组，包括 train_data）
          * 纯推理预测时建议 False，避免不必要的 I/O
        """
        self.file_path = file_path
        self.include_train = include_train
        with h5py.File(file_path, "r") as f:
            self.length = len(f["Sequence"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return data_read(self.file_path, idx, include_train=self.include_train)


# 这样做有助于数据直接进入模型处理，而不需在每个批次中手动转换
