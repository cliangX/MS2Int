import h5py
from preprocess import data_read
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, file_path, include_train: bool = True):
        self.file_path = file_path
        self.include_train = include_train
        with h5py.File(file_path, "r") as f:
            self.length = len(f["Sequence"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return data_read(self.file_path, idx, include_train=self.include_train)
