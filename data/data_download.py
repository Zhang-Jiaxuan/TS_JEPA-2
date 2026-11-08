from aeon.datasets import load_classification

X_train, y_train = load_classification("FordB", split="train")
X_test,  y_test  = load_classification("FordB", split="test")

# 保存为 tsv（与仓库格式一致）
import numpy as np, pandas as pd

def save_tsv(X, y, path):
    # X shape: (n_samples, 1, seq_len) → (n_samples, seq_len)
    X2d = X[:, 0, :]
    df = pd.DataFrame(X2d)
    df.insert(0, "label", y)
    df.to_csv(path, sep='\t', header=False, index=False)

save_tsv(X_train, y_train, "data/fordB/fordB_TRAIN.tsv")
save_tsv(X_test,  y_test,  "data/fordB/fordB_TEST.tsv")
print("Done!")
