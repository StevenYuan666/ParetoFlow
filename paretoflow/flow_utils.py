"""
This module contains utility functions for the flow matching models.
"""

import os

import numpy as np
import torch
from flow import FlowMatching, VectorFieldNet
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def evaluation_fm(
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    name: str = None,
    model_best: torch.nn.Module = None,
    epoch: int = None,
) -> float:
    """
    Evaluate the model on the test set.
    :param test_loader: torch.utils.data.DataLoader: the test data loader
    :param device: torch.device: the device to run the model
    :param name: str: the name of the model
    :param model_best: torch.nn.Module: the best model
    :param epoch: int: the epoch number
    :return: float: the negative log-likelihood
    """
    # EVALUATION
    if model_best is None:
        # load best performing model
        model_best = torch.load(name + ".model")

    model_best.eval()
    loss = 0.0
    N = 0.0
    # use tqdm for progress bar
    with tqdm(total=len(test_loader), desc="Validation", unit="batch") as pbar:
        for indx_batch, test_batch in enumerate(test_loader):
            test_batch = test_batch.float()
            test_batch = test_batch.to(device)
            loss_t = -model_best.log_prob(test_batch, reduction="sum")
            loss = loss + loss_t.item()
            N = N + test_batch.shape[0]
            pbar.update(1)

    loss = loss / N

    if epoch is None:
        print(f"FINAL LOSS: nll={loss}")
    else:
        print(f"Epoch: {epoch}, val nll={loss}")

    return loss


def training_fm(
    name: str,
    max_patience: int,
    num_epochs: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Train the model.
    :param name: str: the name of the model
    :param max_patience: int: the maximum patience
    :param num_epochs: int: the number of epochs
    :param model: torch.nn.Module: the model
    :param optimizer: torch.optim.Optimizer: the optimizer
    :param training_loader: torch.utils.data.DataLoader: the training data loader
    :param val_loader: torch.utils.data.DataLoader: the validation data loader
    :param device: torch.device: the device to run the model
    :return: np.ndarray: the negative log-likelihood
    """
    nll_val = []
    best_nll = float("inf")
    patience = 0

    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        # use tqdm for progress bar
        epoch_loss = 0
        with tqdm(
            total=len(training_loader),
            desc=f"Training {e + 1}/{num_epochs}",
            unit="batch",
        ) as pbar:
            for indx_batch, batch in enumerate(training_loader):
                batch = batch.float()
                batch = batch.to(device)
                loss = model(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss = epoch_loss + loss.item()
                pbar.set_postfix({"loss": epoch_loss / (indx_batch + 1)})
                pbar.update(1)
        print(f"Epoch: {e}, train nll={epoch_loss / len(training_loader)}")

        # Validation
        loss_val = evaluation_fm(val_loader, model_best=model, epoch=e)
        nll_val.append(loss_val)  # save for plotting

        if e == 0:
            print("saved!")
            torch.save(model, name + ".model")
            best_nll = loss_val
        else:
            if loss_val < best_nll:
                print("saved!")
                torch.save(model, name + ".model")
                best_nll = loss_val
                patience = 0
            else:
                patience = patience + 1

        if patience > max_patience:
            print(f"Early stopping at epoch {e + 1}!")
            break

    nll_val = np.asarray(nll_val)

    return nll_val


class DesignDataset(Dataset):
    """
    Dataset class for the designs.
    """

    def __init__(self, designs: np.ndarray):
        """
        Initialize the dataset.
        :param designs: np.ndarray: the designs
        """
        self.X = designs

    def __len__(self) -> int:
        """
        Get the length of the dataset.
        :return: int: the length of the dataset
        """
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.X[idx]


def train_flow_matching(
    all_x: np.ndarray,
    device: torch.device,
    name: str,
    store_path: str = "saved_models/",
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
    )

    return nll_val, model_store_dir + name + ".model", model
