import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}


def get_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    val_ratio: float = 0.9,
    batch_size: int = 32,
):

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).to(**tkwargs)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).to(**tkwargs)
    if isinstance(X_test, np.ndarray):
        X_test = torch.from_numpy(X_test).to(**tkwargs)
    if isinstance(y_test, np.ndarray):
        y_test = torch.from_numpy(y_test).to(**tkwargs)

    tensor_dataset = TensorDataset(X, y)
    lengths = [
        int(val_ratio * len(tensor_dataset)),
        len(tensor_dataset) - int(val_ratio * len(tensor_dataset)),
    ]
    train_dataset, val_dataset = random_split(tensor_dataset, lengths)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 4, shuffle=False, drop_last=False
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size * 4, shuffle=False, drop_last=False
    )

    return train_loader, val_loader, test_loader


def spearman_correlation(x, y):
    n = x.size(0)
    _, rank_x = x.sort(0)
    _, rank_y = y.sort(0)

    d = rank_x - rank_y
    d_squared_sum = (d**2).sum(0).float()

    rho = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
    return rho


task2fullname = {
    "zdt1": "ZDT1-Exact-v0",
    "re21": "RE21-Exact-v0",
    "dtlz1": "DTLZ1-Exact-v0",
}
