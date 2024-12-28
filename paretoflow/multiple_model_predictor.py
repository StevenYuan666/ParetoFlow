"""
This file contains the training of the proxies for the objectives.
"""

import os

import numpy as np
import torch

from paretoflow.multiple_model_predictor_net import (
    MultipleModels,
    SingleModelBaseTrainer,
)
from paretoflow.predictor_utils import get_dataloader

tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}


def train_proxies(
    X: np.ndarray,
    y: np.ndarray,
    name: str,
    save_dir: str = "saved_proxies/",
    tkwargs: dict = tkwargs,
    proxy_lr: float = 1e-3,
    proxy_lr_decay: float = 0.98,
    n_epochs: int = 200,
    proxies_val_ratio: float = 0.1,
    proxies_batch_size: int = 32,
):
    """
    Train the proxies for the objectives.
    :param X: np.ndarray: the input data, with shape (n_samples, n_dim)
    :param y: np.ndarray: the output data, with shape (n_samples, n_obj)
    :param name: str: the name of the model
    :param save_dir: str: the path to save the model
    :param tkwargs: dict: the keyword arguments for the model
    :param proxy_lr: float: the learning rate for the proxies
    :param proxy_lr_decay: float: the learning rate decay for the proxies
    :param n_epochs: int: the number of epochs
    :param proxies_val_ratio: float: the validation ratio
    :param proxies_batch_size: int: the batch size
    :return: str: the path to the model, torch.nn.Module: the model
    """
    n_obj = y.shape[1]
    data_size, n_dim = tuple(X.shape)
    model_save_dir = save_dir
    os.makedirs(model_save_dir, exist_ok=True)

    model = MultipleModels(
        n_dim=n_dim,
        n_obj=n_obj,
        train_mode="Vanilla",
        hidden_size=[2048, 2048],
        save_dir=save_dir,
        save_prefix=f"MultipleModels-Vanilla-{name}",
    )
    model.set_kwargs(**tkwargs)

    trainer_func = SingleModelBaseTrainer

    for which_obj in range(n_obj):

        y0 = y[:, which_obj].copy().reshape(-1, 1)

        trainer = trainer_func(
            model=list(model.obj2model.values())[which_obj],
            which_obj=which_obj,
            proxy_lr=proxy_lr,
            proxy_lr_decay=proxy_lr_decay,
            n_epochs=n_epochs,
        )

        (train_loader, val_loader) = get_dataloader(
            X,
            y0,
            val_ratio=(
                1 - proxies_val_ratio
            ),  # means 0.9 for training and 0.1 for validation
            batch_size=proxies_batch_size,
        )

        trainer.launch(train_loader, val_loader)

    return (
        os.path.join(save_dir, f"MultipleModels-Vanilla-{name}-{which_obj}.pt"),
        model,
    )
