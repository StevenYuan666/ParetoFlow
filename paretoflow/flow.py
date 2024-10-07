"""
This module contains the functions to train the flow matching model.
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from paretoflow.flow_net import FlowMatching, VectorFieldNet
from paretoflow.flow_utils import DesignDataset, training_fm


def train_flow_matching(
    all_x: np.ndarray,
    device: torch.device,
    name: str,
    store_path: str = "saved_fm_models/",
    validation_size: int = None,
    batch_size: int = 64,
    lr: float = 1e-3,
    fm_prob_path: str = "icfm",
    fm_sampling_steps: int = 1000,
    fm_sigma: float = 0.0,
    hidden_size: int = 512,
    patience: int = 20,
    epochs: int = 1000,
):
    """
    Train the flow matching model.
    :param all_x: np.ndarray: the input data
    :param device: torch.device: the device to run the model
    :param name: str: the name of the model
    :param store_path: str: the path to store the model
    :param validation_size: int: the size of the validation set
    :param batch_size: int: the batch size
    :param lr: float: the learning rate
    :param fm_prob_path: str: the flow matching probability path
    :param fm_sampling_steps: int: the flow matching sampling steps
    :param fm_sigma: float: the flow matching sigma
    :param hidden_size: int: the hidden size
    :param patience: int: the patience
    :param epochs: int: the number of epochs
    :return: np.ndarray: the negative log-likelihood,
             str: the path to the model, torch.nn.Module: the model
    """
    # Use a subset of the data
    if validation_size is not None:
        data_size = int(all_x.shape[0] - validation_size)
        X_test = all_x[data_size:]
        X_train = all_x[:data_size]

    # Obtain the number of data points and the number of dimensions
    data_size, n_dim = tuple(X_train.shape)

    # Create datasets
    training_dataset = DesignDataset(X_train)
    val_dataset = DesignDataset(X_test)

    # Create dataloaders
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create the model
    model_store_dir = store_path
    if not (os.path.exists(model_store_dir)):
        os.makedirs(model_store_dir)

    net = VectorFieldNet(n_dim, hidden_size)
    net = net.to(device)
    model = FlowMatching(
        net, fm_sigma, n_dim, fm_sampling_steps, prob_path=fm_prob_path
    )
    model = model.to(device)

    # OPTIMIZER
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    # Training procedure
    nll_val = training_fm(
        name=model_store_dir + name,
        max_patience=patience,
        num_epochs=epochs,
        model=model,
        optimizer=optimizer,
        training_loader=training_loader,
        val_loader=val_loader,
        device=device,
    )

    return nll_val, model_store_dir + name + ".model", model
