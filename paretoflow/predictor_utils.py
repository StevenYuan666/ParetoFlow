"""
This file contains utility functions for the predictor module.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}


def spearman_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the Spearman correlation
    :param x: torch.Tensor: the first tensor
    :param y: torch.Tensor: the second tensor
    :return: torch.Tensor: the Spearman correlation
    """
    n = x.size(0)
    _, rank_x = x.sort(0)
    _, rank_y = y.sort(0)

    d = rank_x - rank_y
    d_squared_sum = (d**2).sum(0).float()

    rho = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
    return rho


def compute_pcc(valid_preds: torch.Tensor, valid_labels: torch.Tensor) -> torch.Tensor:
    """
    Compute the Pearson correlation coefficient
    :param valid_preds: torch.Tensor: the predicted values
    :param valid_labels: torch.Tensor: the true values
    :return: torch.Tensor: the Pearson correlation coefficient
    """
    vx = valid_preds - torch.mean(valid_preds)
    vy = valid_labels - torch.mean(valid_labels)
    pcc = torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(vx**2) + 1e-12) * torch.sqrt(torch.sum(vy**2) + 1e-12)
    )
    return pcc


def get_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.9,
    batch_size: int = 32,
):

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).to(**tkwargs)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).to(**tkwargs)

    tensor_dataset = TensorDataset(X, y)
    lengths = [
        int(val_ratio * len(tensor_dataset)),
        len(tensor_dataset) - int(val_ratio * len(tensor_dataset)),
    ]
    train_dataset, val_dataset = random_split(tensor_dataset, lengths)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 4, shuffle=False, drop_last=False
    )

    return train_loader, val_loader
