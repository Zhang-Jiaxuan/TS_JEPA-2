import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FordADataset(Dataset):
    def __init__(self, path, ratio_patches, regression=False):
        df = pd.read_csv(path, sep='\t', header=None)
        labels = df.iloc[:, 0].values
        data   = df.iloc[:, 1:].values.astype(np.float32)

        # 标签 -1 → 0（二分类标准化）
        labels = np.where(labels == -1, 0, 1)

        # Z-score 归一化（per-sample）
        mean = data.mean(axis=1, keepdims=True)
        std  = data.std(axis=1, keepdims=True) + 1e-8
        data = (data - mean) / std

        # 切分 patches
        seq_len = data.shape[1]           # 500
        patch_size = seq_len // ratio_patches  # 500/10 = 50
        data = data.reshape(len(data), ratio_patches, patch_size)

        self.data   = torch.tensor(data)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.n_classes = 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_eval_loaders(path_train, path_test, batch_size,
                     ratio_patches, regression=False, **kwargs):
    train_ds = FordADataset(path_train, ratio_patches, regression)
    test_ds  = FordADataset(path_test,  ratio_patches, regression)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
